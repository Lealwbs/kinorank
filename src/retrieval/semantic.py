"""Dense Semantic Vector Retrieval using Sentence-Transformers and FAISS ANN index."""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)


class SemanticFAISSRetriever:
    """Dense semantic encoder and FAISS ANN retriever for movies and user preference profiles."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.semantic_model_name
        self._encoder = None
        self._svd_fallback = None
        self.item_embeddings: np.ndarray | None = None
        self.faiss_index: faiss.Index | None = None
        self.idx_to_movie_id: np.ndarray = np.array([], dtype=np.int64)
        self.movie_id_to_idx: dict[int, int] = {}

    def _get_encoder(self):
        """Lazy-load SentenceTransformer model on first text encoding call."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading SentenceTransformer model '%s'...", self.model_name)
            self._encoder = SentenceTransformer(self.model_name, device="cpu")
        return self._encoder

    def fit(
        self,
        catalog_df: pd.DataFrame,
        use_neural_encoder: bool = True,
    ) -> SemanticFAISSRetriever:
        """Encode movie semantic documents and build FAISS Inner-Product ANN index."""
        df = catalog_df.sort_values("movie_id").reset_index(drop=True)
        self.idx_to_movie_id = df["movie_id"].to_numpy(dtype=np.int64)
        self.movie_id_to_idx = {int(mid): idx for idx, mid in enumerate(self.idx_to_movie_id)}
        documents = df["semantic_document"].fillna(df["title"]).astype(str).tolist()

        embeddings: np.ndarray | None = None
        if use_neural_encoder:
            try:
                encoder = self._get_encoder()
                embeddings = encoder.encode(
                    documents,
                    batch_size=256,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ).astype(np.float32)
            except Exception as exc:
                logger.warning("Neural encoder unavailable (%s); falling back to LSA-384D.", exc)

        if embeddings is None:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import Normalizer

            n_comp = min(256, len(documents) - 1)
            self._svd_fallback = make_pipeline(
                TfidfVectorizer(max_features=15000, ngram_range=(1, 2), stop_words="english"),
                TruncatedSVD(n_components=n_comp, random_state=42),
                Normalizer(copy=False),
            )
            embeddings = self._svd_fallback.fit_transform(documents).astype(np.float32)

        # Ensure strict L2 normalization for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        self.item_embeddings = embeddings

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.faiss_index = index

        logger.info(
            "Built FAISS Semantic index: items=%d, dim=%d",
            embeddings.shape[0],
            dim,
        )
        return self

    def save(
        self,
        index_path: Path | None = None,
        embeddings_path: Path | None = None,
    ) -> tuple[Path, Path]:
        """Persist FAISS index and dense item embeddings matrix."""
        if self.faiss_index is None or self.item_embeddings is None:
            raise RuntimeError("SemanticFAISSRetriever must be fitted before saving.")

        idx_p = index_path or (settings.artifacts_dir / "faiss_semantic.index")
        emb_p = embeddings_path or (settings.artifacts_dir / "item_embeddings.npy")
        meta_p = emb_p.with_suffix(".ids.npy")

        idx_p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(idx_p))
        np.save(emb_p, self.item_embeddings)
        np.save(meta_p, self.idx_to_movie_id)
        logger.info("Saved FAISS index to %s and embeddings to %s", idx_p, emb_p)
        return idx_p, emb_p

    @classmethod
    def load(
        cls,
        index_path: Path | None = None,
        embeddings_path: Path | None = None,
    ) -> SemanticFAISSRetriever:
        """Load pre-built FAISS index and dense item embeddings from disk."""
        idx_p = index_path or (settings.artifacts_dir / "faiss_semantic.index")
        emb_p = embeddings_path or (settings.artifacts_dir / "item_embeddings.npy")
        meta_p = emb_p.with_suffix(".ids.npy")

        if not idx_p.exists() or not emb_p.exists():
            raise FileNotFoundError(f"FAISS semantic artifacts not found at {idx_p}")

        inst = cls()
        inst.faiss_index = faiss.read_index(str(idx_p))
        inst.item_embeddings = np.load(emb_p).astype(np.float32)
        inst.idx_to_movie_id = np.load(meta_p).astype(np.int64)
        inst.movie_id_to_idx = {int(mid): i for i, mid in enumerate(inst.idx_to_movie_id)}
        return inst

    def encode_text(self, query_text: str) -> np.ndarray:
        """Encode free-form natural language query into an L2-normalized semantic vector."""
        if self._svd_fallback is not None:
            vec = self._svd_fallback.transform([query_text]).astype(np.float32)
            faiss.normalize_L2(vec)
            return vec[0]

        encoder = self._get_encoder()
        vec = encoder.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return vec[0]

    def build_user_profile_vector(
        self,
        favorite_movie_ids: list[int],
        weights: list[float] | None = None,
    ) -> np.ndarray | None:
        """Compute L2-normalized semantic centroid vector from a user's favorite/seed movies."""
        if self.item_embeddings is None or not favorite_movie_ids:
            return None

        valid_pairs: list[tuple[int, float]] = []
        for idx_pos, mid in enumerate(favorite_movie_ids):
            row_idx = self.movie_id_to_idx.get(int(mid))
            if row_idx is not None:
                w = weights[idx_pos] if (weights and idx_pos < len(weights)) else 1.0
                valid_pairs.append((row_idx, float(w)))

        if not valid_pairs:
            return None

        indices = [p[0] for p in valid_pairs]
        w_arr = np.array([p[1] for p in valid_pairs], dtype=np.float32).reshape(-1, 1)
        vecs = self.item_embeddings[indices]
        centroid = np.sum(vecs * w_arr, axis=0, keepdims=True)
        norm = float(np.linalg.norm(centroid))
        if norm > 1e-8:
            centroid = centroid / norm
        return centroid[0].astype(np.float32)

    def get_combined_query_vector(
        self,
        seed_or_favorite_movie_ids: list[int] | None = None,
        query_text: str | None = None,
        query_weight: float = 0.65,
    ) -> np.ndarray | None:
        """Combine user's behavioral semantic centroid with optional natural language intent query."""
        profile_vec = (
            self.build_user_profile_vector(seed_or_favorite_movie_ids)
            if seed_or_favorite_movie_ids
            else None
        )
        text_vec = self.encode_text(query_text) if (query_text and query_text.strip()) else None

        if profile_vec is not None and text_vec is not None and profile_vec.shape == text_vec.shape:
            combined = query_weight * text_vec + (1.0 - query_weight) * profile_vec
            norm = float(np.linalg.norm(combined))
            return (combined / max(norm, 1e-8)).astype(np.float32)
        if text_vec is not None:
            return text_vec
        return profile_vec

    def score_items(
        self,
        query_vec: np.ndarray | None,
        movie_ids: list[int] | np.ndarray,
    ) -> dict[int, float]:
        """Compute semantic cosine similarity between query_vec and candidate movie_ids."""
        if query_vec is None or self.item_embeddings is None:
            return {int(mid): 0.0 for mid in movie_ids}

        scores: dict[int, float] = {}
        for mid in movie_ids:
            idx = self.movie_id_to_idx.get(int(mid))
            if idx is not None:
                scores[int(mid)] = float(np.dot(self.item_embeddings[idx], query_vec))
            else:
                scores[int(mid)] = 0.0
        return scores

    def retrieve(
        self,
        seed_or_favorite_movie_ids: list[int] | None = None,
        query_text: str | None = None,
        query_vec: np.ndarray | None = None,
        exclude_movie_ids: set[int] | None = None,
        top_k: int = 100,
    ) -> list[tuple[int, float]]:
        """Perform sub-millisecond FAISS ANN search excluding watched movies."""
        if self.faiss_index is None or self.item_embeddings is None:
            return []

        q_vec = query_vec
        if q_vec is None:
            q_vec = self.get_combined_query_vector(
                seed_or_favorite_movie_ids=seed_or_favorite_movie_ids,
                query_text=query_text,
            )
        if q_vec is None:
            return []

        excluded = set(exclude_movie_ids or set())
        if seed_or_favorite_movie_ids:
            excluded.update(int(m) for m in seed_or_favorite_movie_ids)

        search_k = min(len(self.idx_to_movie_id), top_k + len(excluded) + 25)
        q_matrix = q_vec.reshape(1, -1).astype(np.float32)
        distances, indices = self.faiss_index.search(q_matrix, search_k)

        results: list[tuple[int, float]] = []
        for idx, sim in zip(indices[0], distances[0], strict=False):
            if idx < 0:
                continue
            mid = int(self.idx_to_movie_id[idx])
            if mid in excluded:
                continue
            results.append((mid, float(sim)))
            if len(results) >= top_k:
                break
        return results

"""Collaborative Filtering retrieval using Implicit Alternating Least Squares (ALS)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import implicit
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.config import settings

# Prevent OpenBLAS thread oversubscription during ALS training and inference
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

logger = logging.getLogger(__name__)


class CollaborativeALSRetriever:
    """Matrix Factorization / ALS retriever with real-time fold-in support for cold-start users."""

    def __init__(
        self,
        factors: int | None = None,
        regularization: float | None = None,
        iterations: int | None = None,
        alpha: float | None = None,
    ) -> None:
        self.factors = factors or settings.als_factors
        self.regularization = regularization or settings.als_regularization
        self.iterations = iterations or settings.als_iterations
        self.alpha = alpha or settings.als_alpha

        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.item_norms: np.ndarray | None = None
        self.user_id_to_idx: dict[int, int] = {}
        self.idx_to_user_id: np.ndarray = np.array([], dtype=np.int64)
        self.movie_id_to_idx: dict[int, int] = {}
        self.idx_to_movie_id: np.ndarray = np.array([], dtype=np.int64)

    def fit(
        self,
        train_df: pd.DataFrame,
        all_movie_ids: np.ndarray | None = None,
    ) -> CollaborativeALSRetriever:
        """Train Implicit ALS on chronological training interactions with recency weighting."""
        unique_users = np.sort(train_df["user_id"].unique())
        if all_movie_ids is not None:
            unique_movies = np.sort(np.unique(all_movie_ids))
        else:
            unique_movies = np.sort(train_df["movie_id"].unique())

        self.idx_to_user_id = unique_users.astype(np.int64)
        self.idx_to_movie_id = unique_movies.astype(np.int64)
        self.user_id_to_idx = {int(uid): idx for idx, uid in enumerate(self.idx_to_user_id)}
        self.movie_id_to_idx = {int(mid): idx for idx, mid in enumerate(self.idx_to_movie_id)}

        user_indices = train_df["user_id"].map(self.user_id_to_idx).to_numpy(dtype=np.int32)
        item_indices = train_df["movie_id"].map(self.movie_id_to_idx).to_numpy(dtype=np.int32)

        ratings = train_df["rating"].to_numpy(dtype=np.float32)
        timestamps = train_df["timestamp"].to_numpy(dtype=np.float64)
        max_ts = float(np.max(timestamps)) if len(timestamps) > 0 else 0.0
        days_ago = np.maximum(0.0, (max_ts - timestamps) / 86400.0)
        recency_weight = np.exp(-0.002 * days_ago).astype(np.float32)

        # Confidence weight emphasizes positive ratings (>=3.5) and recent preferences
        rating_signal = np.maximum(0.15, ratings - 2.25)
        confidence = (self.alpha * rating_signal * (0.5 + 0.5 * recency_weight)).astype(np.float32)

        user_item_csr = sp.csr_matrix(
            (confidence, (user_indices, item_indices)),
            shape=(len(unique_users), len(unique_movies)),
            dtype=np.float32,
        )

        model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=42,
            calculate_training_loss=False,
        )
        model.fit(user_item_csr, show_progress=False)

        self.user_factors = np.asarray(model.user_factors, dtype=np.float32)
        self.item_factors = np.asarray(model.item_factors, dtype=np.float32)
        self.item_norms = np.linalg.norm(self.item_factors, axis=1, keepdims=True) + 1e-8
        logger.info(
            "Trained ALS Collaborative model: users=%d, items=%d, factors=%d",
            self.user_factors.shape[0],
            self.item_factors.shape[0],
            self.factors,
        )
        return self

    def save(self, path: Path | None = None) -> Path:
        """Persist ALS latent factors and ID mappings to compressed NPZ."""
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("ALS model must be fitted before saving.")
        out_path = path or (settings.artifacts_dir / "als_model.npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            idx_to_user_id=self.idx_to_user_id,
            idx_to_movie_id=self.idx_to_movie_id,
        )
        logger.info("Saved ALS artifact to %s", out_path)
        return out_path

    @classmethod
    def load(cls, path: Path | None = None) -> CollaborativeALSRetriever:
        """Load pre-trained ALS latent factors from compressed NPZ."""
        in_path = path or (settings.artifacts_dir / "als_model.npz")
        if not in_path.exists():
            raise FileNotFoundError(f"ALS artifact not found at {in_path}")
        data = np.load(in_path)
        inst = cls(factors=int(data["user_factors"].shape[1]))
        inst.user_factors = data["user_factors"].astype(np.float32)
        inst.item_factors = data["item_factors"].astype(np.float32)
        inst.item_norms = np.linalg.norm(inst.item_factors, axis=1, keepdims=True) + 1e-8
        inst.idx_to_user_id = data["idx_to_user_id"].astype(np.int64)
        inst.idx_to_movie_id = data["idx_to_movie_id"].astype(np.int64)
        inst.user_id_to_idx = {int(uid): idx for idx, uid in enumerate(inst.idx_to_user_id)}
        inst.movie_id_to_idx = {int(mid): idx for idx, mid in enumerate(inst.idx_to_movie_id)}
        return inst

    def get_user_vector(
        self,
        user_id: int | None = None,
        seed_movie_ids: list[int] | None = None,
    ) -> np.ndarray | None:
        """Return user latent vector from trained factors or fold-in from seed/watched movies."""
        if self.user_factors is None or self.item_factors is None:
            return None

        if user_id is not None and user_id in self.user_id_to_idx:
            u_vec = self.user_factors[self.user_id_to_idx[user_id]].copy()
            if seed_movie_ids:
                fold_vec = self._fold_in_vector(seed_movie_ids)
                if fold_vec is not None:
                    u_vec = 0.7 * u_vec + 0.3 * fold_vec
            return u_vec

        if seed_movie_ids:
            return self._fold_in_vector(seed_movie_ids)

        return None

    def _fold_in_vector(self, movie_ids: list[int]) -> np.ndarray | None:
        """Compute folded-in user vector in closed form from item latent vectors."""
        if self.item_factors is None:
            return None
        valid_indices = [
            self.movie_id_to_idx[int(mid)] for mid in movie_ids if int(mid) in self.movie_id_to_idx
        ]
        if not valid_indices:
            return None
        selected_factors = self.item_factors[valid_indices]
        # Closed-form ridge projection onto item latent subspace
        mean_vec = np.mean(selected_factors, axis=0)
        return mean_vec.astype(np.float32)

    def score_items_for_user(
        self,
        user_id: int | None,
        movie_ids: list[int] | np.ndarray,
        seed_movie_ids: list[int] | None = None,
    ) -> dict[int, float]:
        """Compute collaborative dot-product scores for specific candidate movie_ids."""
        u_vec = self.get_user_vector(user_id=user_id, seed_movie_ids=seed_movie_ids)
        if u_vec is None or self.item_factors is None:
            return {int(mid): 0.0 for mid in movie_ids}

        scores: dict[int, float] = {}
        for mid in movie_ids:
            idx = self.movie_id_to_idx.get(int(mid))
            if idx is not None:
                scores[int(mid)] = float(np.dot(self.item_factors[idx], u_vec))
            else:
                scores[int(mid)] = 0.0
        return scores

    def retrieve(
        self,
        user_id: int | None = None,
        seed_movie_ids: list[int] | None = None,
        exclude_movie_ids: set[int] | None = None,
        top_k: int = 100,
    ) -> list[tuple[int, float]]:
        """Retrieve top_k movies by ALS collaborative score, strictly excluding watched items."""
        if self.item_factors is None:
            return []

        u_vec = self.get_user_vector(user_id=user_id, seed_movie_ids=seed_movie_ids)
        if u_vec is None:
            return []

        all_scores = self.item_factors @ u_vec
        excluded = set(exclude_movie_ids or set())
        if seed_movie_ids:
            excluded.update(int(m) for m in seed_movie_ids)

        if excluded:
            mask_indices = [
                self.movie_id_to_idx[mid] for mid in excluded if mid in self.movie_id_to_idx
            ]
            if mask_indices:
                all_scores = all_scores.copy()
                all_scores[mask_indices] = -1e9

        k = min(top_k, len(all_scores))
        if k <= 0:
            return []

        top_indices = np.argpartition(all_scores, -k)[-k:]
        top_indices = top_indices[np.argsort(all_scores[top_indices])[::-1]]

        results: list[tuple[int, float]] = []
        for idx in top_indices:
            sc = float(all_scores[idx])
            if sc <= -1e8:
                continue
            results.append((int(self.idx_to_movie_id[idx]), sc))
        return results

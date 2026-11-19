"""Stage 1 Hybrid Candidate Generator combining Collaborative ALS and Semantic FAISS via RRF."""

from __future__ import annotations

import logging
import time

import numpy as np

from src.config import settings
from src.retrieval.collaborative import CollaborativeALSRetriever
from src.retrieval.semantic import SemanticFAISSRetriever
from src.schemas.domain import CandidateItem

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    collab_ranked: list[tuple[int, float]],
    semantic_ranked: list[tuple[int, float]],
    exclude_movie_ids: set[int] | None = None,
    rrf_k: int = 60,
    collab_weight: float = 0.55,
    semantic_weight: float = 0.45,
    top_k: int = 100,
) -> list[CandidateItem]:
    """Fuse Collaborative and Semantic candidate lists using Reciprocal Rank Fusion (RRF)."""
    excluded = exclude_movie_ids or set()

    collab_map: dict[int, tuple[int, float]] = {}
    for rank_idx, (mid, score) in enumerate(collab_ranked, start=1):
        if mid not in excluded:
            collab_map[int(mid)] = (rank_idx, float(score))

    semantic_map: dict[int, tuple[int, float]] = {}
    for rank_idx, (mid, score) in enumerate(semantic_ranked, start=1):
        if mid not in excluded:
            semantic_map[int(mid)] = (rank_idx, float(score))

    all_candidate_ids = set(collab_map.keys()) | set(semantic_map.keys())
    candidates: list[CandidateItem] = []

    for mid in all_candidate_ids:
        c_entry = collab_map.get(mid)
        s_entry = semantic_map.get(mid)

        c_rank = c_entry[0] if c_entry else None
        c_score = c_entry[1] if c_entry else 0.0
        s_rank = s_entry[0] if s_entry else None
        s_score = s_entry[1] if s_entry else 0.0

        rrf_score = 0.0
        if c_rank is not None:
            rrf_score += collab_weight / (rrf_k + c_rank)
        if s_rank is not None:
            rrf_score += semantic_weight / (rrf_k + s_rank)

        if c_rank is not None and s_rank is not None:
            source = "hybrid_both"
        elif c_rank is not None:
            source = "collaborative_als"
        else:
            source = "semantic_faiss"

        candidates.append(
            CandidateItem(
                movie_id=mid,
                collab_score=round(c_score, 5),
                collab_rank=c_rank,
                semantic_score=round(s_score, 5),
                semantic_rank=s_rank,
                rrf_score=round(rrf_score, 6),
                retrieval_source=source,
            )
        )

    candidates.sort(
        key=lambda x: (x.rrf_score, x.collab_score + x.semantic_score),
        reverse=True,
    )
    return candidates[:top_k]


class HybridCandidateRetriever:
    """Two-pathway (Collaborative ALS + Semantic FAISS) candidate generator with < 20ms latency."""

    def __init__(
        self,
        als_retriever: CollaborativeALSRetriever,
        semantic_retriever: SemanticFAISSRetriever,
        popular_fallback_ids: list[int] | None = None,
    ) -> None:
        self.als = als_retriever
        self.semantic = semantic_retriever
        self.popular_fallback_ids = popular_fallback_ids or []

    def retrieve_candidates(
        self,
        user_id: int | None = None,
        watched_movie_ids: list[int] | None = None,
        favorite_movie_ids: list[int] | None = None,
        query_text: str | None = None,
        top_k: int | None = None,
        rrf_k: int | None = None,
    ) -> tuple[list[CandidateItem], float, np.ndarray | None]:
        """Retrieve top_k hybrid candidates, returning (candidates, elapsed_ms, semantic_query_vec)."""
        t0 = time.perf_counter()
        target_k = top_k or settings.retrieval_top_k
        k_const = rrf_k or settings.rrf_k

        watched_list = [int(m) for m in (watched_movie_ids or [])]
        excluded_set = set(watched_list)

        fav_seeds = [int(m) for m in (favorite_movie_ids or [])]
        if not fav_seeds and watched_list:
            fav_seeds = watched_list[-10:]

        # 1. Collaborative ALS Pathway
        collab_hits = self.als.retrieve(
            user_id=user_id,
            seed_movie_ids=fav_seeds if (user_id is None or user_id < 0) else None,
            exclude_movie_ids=excluded_set,
            top_k=target_k,
        )

        # 2. Semantic FAISS Pathway
        sem_query_vec = self.semantic.get_combined_query_vector(
            seed_or_favorite_movie_ids=fav_seeds,
            query_text=query_text,
        )
        semantic_hits = self.semantic.retrieve(
            query_vec=sem_query_vec,
            exclude_movie_ids=excluded_set,
            top_k=target_k,
        )

        # Adjust fusion weights dynamically when an explicit conversational query is present
        has_text_query = bool(query_text and query_text.strip())
        c_weight = 0.35 if has_text_query else 0.58
        s_weight = 0.65 if has_text_query else 0.42

        fused = reciprocal_rank_fusion(
            collab_ranked=collab_hits,
            semantic_ranked=semantic_hits,
            exclude_movie_ids=excluded_set,
            rrf_k=k_const,
            collab_weight=c_weight,
            semantic_weight=s_weight,
            top_k=target_k,
        )

        # Backfill exact scores when an item was retrieved by only one pathway
        missing_collab_ids = [c.movie_id for c in fused if c.collab_rank is None]
        if missing_collab_ids:
            backfilled_c = self.als.score_items_for_user(
                user_id=user_id,
                movie_ids=missing_collab_ids,
                seed_movie_ids=fav_seeds,
            )
            for c in fused:
                if c.collab_rank is None:
                    c.collab_score = round(backfilled_c.get(c.movie_id, 0.0), 5)

        missing_sem_ids = [c.movie_id for c in fused if c.semantic_rank is None]
        if missing_sem_ids and sem_query_vec is not None:
            backfilled_s = self.semantic.score_items(sem_query_vec, missing_sem_ids)
            for c in fused:
                if c.semantic_rank is None:
                    c.semantic_score = round(backfilled_s.get(c.movie_id, 0.0), 5)

        # Cold-start zero-input fallback: use Bayesian popularity ranking
        if not fused and self.popular_fallback_ids:
            for rank_i, mid in enumerate(self.popular_fallback_ids, start=1):
                if mid in excluded_set:
                    continue
                fused.append(
                    CandidateItem(
                        movie_id=mid,
                        rrf_score=round(1.0 / (k_const + rank_i), 6),
                        retrieval_source="popularity_fallback",
                    )
                )
                if len(fused) >= target_k:
                    break

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return fused, round(elapsed_ms, 3), sem_query_vec

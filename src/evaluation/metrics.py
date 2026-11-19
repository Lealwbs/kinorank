"""Information Retrieval & Recommender Systems Ranking Metrics (NDCG, Recall, MAP, Coverage, HitRate, ILD)."""

from __future__ import annotations

import math

import numpy as np


def ndcg_at_k(
    recommended_ids: list[int],
    relevant_set: set[int],
    graded_relevance: dict[int, float] | None = None,
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at K (NDCG@K)."""
    if not relevant_set or k <= 0:
        return 0.0

    top_k = recommended_ids[:k]
    dcg = 0.0
    for idx, mid in enumerate(top_k):
        if mid in relevant_set:
            gain = float(graded_relevance.get(mid, 1.0)) if graded_relevance else 1.0
            dcg += gain / math.log2(idx + 2.0)

    if dcg == 0.0:
        return 0.0

    if graded_relevance:
        ideal_gains = sorted(
            [float(graded_relevance.get(mid, 1.0)) for mid in relevant_set],
            reverse=True,
        )[:k]
    else:
        ideal_gains = [1.0] * min(k, len(relevant_set))

    idcg = sum(g / math.log2(idx + 2.0) for idx, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0.0 else 0.0


def recall_at_k(
    recommended_ids: list[int],
    relevant_set: set[int],
    k: int = 20,
) -> float:
    """Compute Recall@K: fraction of relevant future items retrieved in top-K."""
    if not relevant_set or k <= 0:
        return 0.0
    hits = sum(1 for mid in recommended_ids[:k] if mid in relevant_set)
    return hits / len(relevant_set)


def average_precision_at_k(
    recommended_ids: list[int],
    relevant_set: set[int],
    k: int = 10,
) -> float:
    """Compute Average Precision at K (AP@K), aggregated across users to form MAP@K."""
    if not relevant_set or k <= 0:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for idx, mid in enumerate(recommended_ids[:k], start=1):
        if mid in relevant_set:
            hits += 1
            sum_precisions += hits / idx

    if hits == 0:
        return 0.0
    return sum_precisions / min(k, len(relevant_set))


def hit_rate_at_k(
    recommended_ids: list[int],
    relevant_set: set[int],
    k: int = 10,
) -> float:
    """Compute binary HitRate@K (1.0 if at least one relevant item is in top-K, else 0.0)."""
    if not relevant_set or k <= 0:
        return 0.0
    for mid in recommended_ids[:k]:
        if mid in relevant_set:
            return 1.0
    return 0.0


def catalog_coverage_pct(
    all_recommended_lists: list[list[int]],
    total_catalog_size: int,
    k: int = 10,
) -> float:
    """Compute percentage of active catalog recommended across all evaluated users."""
    if total_catalog_size <= 0:
        return 0.0
    unique_items: set[int] = set()
    for rec_list in all_recommended_lists:
        unique_items.update(rec_list[:k])
    return (len(unique_items) / total_catalog_size) * 100.0


def intra_list_diversity(
    recommended_ids: list[int],
    item_embeddings: np.ndarray,
    movie_id_to_idx: dict[int, int],
    k: int = 10,
) -> float:
    """Compute average pairwise cosine dissimilarity (1 - cosine_sim) within top-K list."""
    top_ids = [mid for mid in recommended_ids[:k] if mid in movie_id_to_idx]
    if len(top_ids) < 2:
        return 0.0
    indices = [movie_id_to_idx[mid] for mid in top_ids]
    vecs = item_embeddings[indices]
    sim_matrix = np.clip(vecs @ vecs.T, -1.0, 1.0)
    n = len(top_ids)
    upper_tri_sum = float(np.sum(np.triu(1.0 - sim_matrix, k=1)))
    num_pairs = n * (n - 1) / 2.0
    return upper_tri_sum / num_pairs

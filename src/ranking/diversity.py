"""Maximal Marginal Relevance (MMR) post-processing for relevance vs. diversity trade-off."""

from __future__ import annotations

import numpy as np

from src.retrieval.semantic import SemanticFAISSRetriever
from src.schemas.domain import RankedRecommendation


def apply_mmr_diversity(
    ranked_items: list[RankedRecommendation],
    semantic_retriever: SemanticFAISSRetriever | None = None,
    mmr_lambda: float = 0.75,
    top_k: int = 15,
) -> list[RankedRecommendation]:
    """Re-order ranked candidates using Maximal Marginal Relevance (MMR) to prevent filter bubbles.

    MMR(i) = lambda * Rel(i) - (1 - lambda) * max_{j in Selected} Sim(i, j)
    """
    if not ranked_items or len(ranked_items) <= 1:
        return ranked_items[:top_k]

    target_k = min(top_k, len(ranked_items))
    if mmr_lambda >= 0.999:
        trimmed = ranked_items[:target_k]
        for idx, item in enumerate(trimmed, start=1):
            item.rank = idx
        return trimmed

    # Min-Max normalize LightGBM relevance scores into [0, 1]
    raw_scores = np.array([item.lgbm_score for item in ranked_items], dtype=np.float32)
    s_min = float(np.min(raw_scores))
    s_max = float(np.max(raw_scores))
    if s_max - s_min > 1e-6:
        norm_rel = (raw_scores - s_min) / (s_max - s_min)
    else:
        norm_rel = np.ones_like(raw_scores)

    n = len(ranked_items)
    # Build pairwise similarity matrix combining dense semantic cosine similarity + genre Jaccard
    sim_matrix = np.zeros((n, n), dtype=np.float32)

    has_embeddings = (
        semantic_retriever is not None and semantic_retriever.item_embeddings is not None
    )
    if has_embeddings:
        emb_dim = semantic_retriever.item_embeddings.shape[1]
        vecs = np.zeros((n, emb_dim), dtype=np.float32)
        for i, item in enumerate(ranked_items):
            row_idx = semantic_retriever.movie_id_to_idx.get(item.movie_id)
            if row_idx is not None:
                vecs[i] = semantic_retriever.item_embeddings[row_idx]
        sem_sim = np.clip(vecs @ vecs.T, 0.0, 1.0)
        sim_matrix += 0.65 * sem_sim

    genre_sets = [set(item.genres) for item in ranked_items]
    for i in range(n):
        for j in range(i + 1, n):
            gi, gj = genre_sets[i], genre_sets[j]
            g_jacc = len(gi & gj) / len(gi | gj) if (gi or gj) else 0.0
            dir_same = (
                0.15
                if (
                    ranked_items[i].director != "Unknown"
                    and ranked_items[i].director == ranked_items[j].director
                )
                else 0.0
            )
            extra_sim = (0.35 if has_embeddings else 1.0) * g_jacc + dir_same
            sim_matrix[i, j] = min(1.0, sim_matrix[i, j] + extra_sim)
            sim_matrix[j, i] = sim_matrix[i, j]

    selected_indices: list[int] = []
    remaining_indices = list(range(n))

    # First item is always the highest relevance item
    first_idx = int(np.argmax(norm_rel))
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    ranked_items[first_idx].mmr_score = round(float(norm_rel[first_idx]), 5)
    ranked_items[first_idx].final_score = ranked_items[first_idx].mmr_score

    while len(selected_indices) < target_k and remaining_indices:
        best_idx = -1
        best_mmr = -1e9

        for cand_i in remaining_indices:
            max_sim_to_selected = float(np.max(sim_matrix[cand_i, selected_indices]))
            mmr_val = (
                mmr_lambda * float(norm_rel[cand_i]) - (1.0 - mmr_lambda) * max_sim_to_selected
            )
            if mmr_val > best_mmr:
                best_mmr = mmr_val
                best_idx = cand_i

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        ranked_items[best_idx].mmr_score = round(float(best_mmr), 5)
        ranked_items[best_idx].final_score = ranked_items[best_idx].mmr_score

    final_list: list[RankedRecommendation] = []
    for new_rank, idx in enumerate(selected_indices, start=1):
        item = ranked_items[idx].model_copy()
        item.rank = new_rank
        final_list.append(item)

    return final_list

"""Automated tests for Stage 1 Hybrid Retrieval (< 20ms), Feature Store, LightGBM Ranker, and MMR."""

from __future__ import annotations

import time

from src.ranking.diversity import apply_mmr_diversity
from src.ranking.feature_store import FeatureStore
from src.ranking.lgbm_ranker import LightGBMRanker
from src.retrieval.collaborative import CollaborativeALSRetriever
from src.retrieval.hybrid import HybridCandidateRetriever
from src.retrieval.semantic import SemanticFAISSRetriever
from src.schemas.agent import UserIntent


def test_hybrid_retrieval_latency_and_zero_watched_leakage() -> None:
    """Ensure Stage 1 Hybrid Retrieval (ALS + FAISS + RRF) runs in < 20ms and excludes watched items."""
    als = CollaborativeALSRetriever.load()
    sem = SemanticFAISSRetriever.load()
    fs = FeatureStore(try_redis=False)
    hybrid = HybridCandidateRetriever(
        als_retriever=als,
        semantic_retriever=sem,
        popular_fallback_ids=fs.popular_movie_ids,
    )

    profile = fs.get_user_profile(user_id=1)
    watched_set = set(profile.watched_movie_ids)
    assert len(watched_set) > 10

    # Warm-up call
    hybrid.retrieve_candidates(user_id=1, watched_movie_ids=profile.watched_movie_ids, top_k=100)

    # Timed benchmark over 10 users
    latencies: list[float] = []
    for uid in range(1, 11):
        u_prof = fs.get_user_profile(user_id=uid)
        u_watched = set(u_prof.watched_movie_ids)
        t0 = time.perf_counter()
        candidates, _, _ = hybrid.retrieve_candidates(
            user_id=uid,
            watched_movie_ids=u_prof.watched_movie_ids,
            top_k=100,
        )
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        assert len(candidates) == 100
        retrieved_ids = {c.movie_id for c in candidates}
        assert len(retrieved_ids & u_watched) == 0, f"Watched movies leaked for user {uid}!"

    median_ms = sorted(latencies)[len(latencies) // 2]
    assert median_ms < 20.0, f"Expected p50 retrieval latency < 20ms, got {median_ms:.2f}ms"


def test_lightgbm_ranker_shap_and_hard_filters() -> None:
    """Validate LightGBM LambdaRank scoring, TreeSHAP attributions, and hard intent filters."""
    als = CollaborativeALSRetriever.load()
    sem = SemanticFAISSRetriever.load()
    fs = FeatureStore(try_redis=False)
    ranker = LightGBMRanker.load()
    hybrid = HybridCandidateRetriever(als, sem, fs.popular_movie_ids)

    profile = fs.get_user_profile(user_id=15)
    candidates, _, _ = hybrid.retrieve_candidates(
        user_id=15,
        watched_movie_ids=profile.watched_movie_ids,
        top_k=100,
    )

    intent = UserIntent(
        raw_query="Sci-Fi or Thriller after 1995 under 150 minutes without Horror",
        semantic_search_query="psychological sci-fi thriller",
        required_genres=["Sci-Fi", "Thriller"],
        excluded_genres=["Horror"],
        min_year=1995,
        max_runtime_minutes=150,
    )

    ranked = ranker.rank(
        user_profile=profile,
        candidates=candidates,
        feature_store=fs,
        intent=intent,
        compute_shap=True,
    )
    assert len(ranked) > 0
    for item in ranked:
        assert item.movie_id not in set(profile.watched_movie_ids)
        assert "Horror" not in item.genres
        assert any(g in ("Sci-Fi", "Thriller") for g in item.genres)
        if item.release_year is not None:
            assert item.release_year >= 1995
        assert item.runtime_minutes <= 150
        assert len(item.feature_contributions) >= 3


def test_mmr_diversity_reordering() -> None:
    """Verify Maximal Marginal Relevance (MMR) executes and preserves top_k bounds."""
    als = CollaborativeALSRetriever.load()
    sem = SemanticFAISSRetriever.load()
    fs = FeatureStore(try_redis=False)
    ranker = LightGBMRanker.load()
    hybrid = HybridCandidateRetriever(als, sem, fs.popular_movie_ids)

    profile = fs.get_user_profile(user_id=42)
    candidates, _, _ = hybrid.retrieve_candidates(
        user_id=42,
        watched_movie_ids=profile.watched_movie_ids,
        top_k=100,
    )
    ranked = ranker.rank(profile, candidates, fs, compute_shap=False)

    mmr_items = apply_mmr_diversity(
        ranked_items=ranked[:40],
        semantic_retriever=sem,
        mmr_lambda=0.65,
        top_k=10,
    )
    assert len(mmr_items) == 10
    # First item must always be the top-1 LightGBM item
    assert mmr_items[0].movie_id == ranked[0].movie_id
    assert [it.rank for it in mmr_items] == list(range(1, 11))


def test_cold_start_seed_movies_retrieval_and_ranking() -> None:
    """Test interactive cold-start flow using 3 seed movies (Fight Club, Se7en, The Matrix)."""
    als = CollaborativeALSRetriever.load()
    sem = SemanticFAISSRetriever.load()
    fs = FeatureStore(try_redis=False)
    ranker = LightGBMRanker.load()
    hybrid = HybridCandidateRetriever(als, sem, fs.popular_movie_ids)

    seeds = [2959, 47, 2571]  # Fight Club, Se7en, The Matrix
    cold_profile = fs.synthesize_cold_start_profile(seed_movie_ids=seeds)
    assert cold_profile.total_interactions == 3
    assert set(seeds).issubset(set(cold_profile.watched_movie_ids))

    candidates, _, _ = hybrid.retrieve_candidates(
        user_id=None,
        watched_movie_ids=seeds,
        favorite_movie_ids=seeds,
        top_k=50,
    )
    assert len(candidates) == 50
    for c in candidates:
        assert c.movie_id not in set(seeds)

    ranked = ranker.rank(cold_profile, candidates, fs)
    assert len(ranked) >= 15

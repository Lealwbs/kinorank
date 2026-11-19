"""Tests for Stage 3 Agentic GenAI Tools, Structured Outputs, and Anti-Hallucination Guardrails."""

from __future__ import annotations

from src.agent.guardrails import enforce_catalog_guardrail
from src.agent.llm_client import deterministic_parse_intent
from src.agent.tools import KinoRankAgentOrchestrator
from src.schemas.agent import AgentRerankOutput, RerankedItemExplanation
from src.schemas.domain import RankedRecommendation, UserProfile


def test_parse_user_intent_extracts_hard_and_soft_constraints() -> None:
    """Verify natural language intent parser extracts genres, exclusions, runtime, year, and auteur."""
    query = "thriller psicológico estilo Fincher, < 130 min, depois de 1994, sem terror"
    intent = deterministic_parse_intent(query)

    assert "Thriller" in intent.required_genres
    assert "Horror" in intent.excluded_genres
    assert intent.max_runtime_minutes == 130
    assert intent.min_year == 1994
    assert "David Fincher" in intent.reference_directors


def test_anti_hallucination_guardrail_blocks_out_of_catalog_and_watched_movies() -> None:
    """Ensure guardrail strictly strips any movie_id outside the Stage 2 candidate pool or already watched."""
    user_profile = UserProfile(
        user_id=99,
        total_interactions=20,
        mean_rating=4.2,
        top_genres=["Thriller", "Crime"],
        watched_movie_ids=[
            2959,
            50,
        ],  # User already watched 2959 (Fight Club) and 50 (Usual Suspects)
        recent_favorites=["Fight Club (1999)"],
    )

    stage2_candidates = [
        RankedRecommendation(
            rank=1,
            movie_id=47,
            title="Seven (a.k.a. Se7en)",
            release_year=1995,
            genres=["Mystery", "Thriller"],
            director="David Fincher",
            runtime_minutes=127,
            bayesian_rating=4.08,
            lgbm_score=1.42,
            final_score=1.0,
        ),
        RankedRecommendation(
            rank=2,
            movie_id=4226,
            title="Memento",
            release_year=2000,
            genres=["Mystery", "Thriller"],
            director="Christopher Nolan",
            runtime_minutes=113,
            bayesian_rating=4.12,
            lgbm_score=1.31,
            final_score=0.92,
        ),
    ]

    # Simulate an LLM hallucinating movie_id=9999999 and attempting to recommend watched movie_id=2959
    hallucinated_output = AgentRerankOutput(
        ordered_items=[
            RerankedItemExplanation(
                movie_id=9999999,
                explanation="Hallucinated movie outside catalog.",
            ),
            RerankedItemExplanation(
                movie_id=2959,
                explanation="Already watched movie that should never be recommended.",
            ),
            RerankedItemExplanation(
                movie_id=4226,
                explanation="Memento aligns with your affinity for non-linear psychological thrillers like Fight Club.",
            ),
        ],
        agent_reasoning_summary="Test guardrail run.",
    )

    verified, rejected = enforce_catalog_guardrail(
        stage2_candidates=stage2_candidates,
        agent_output=hallucinated_output,
        user_profile=user_profile,
        top_k=5,
    )

    assert 9999999 in rejected
    assert 2959 in rejected
    verified_ids = [item.movie_id for item in verified]
    assert 9999999 not in verified_ids
    assert 2959 not in verified_ids
    # Memento (4226) should be ranked #1, and Se7en (47) backfilled at #2
    assert verified_ids == [4226, 47]
    assert all(len(item.explanation) > 15 for item in verified)


def test_end_to_end_agent_orchestrator_with_telemetry() -> None:
    """Verify full 3-stage KinoRankAgentOrchestrator execution and structured telemetry output."""
    orchestrator = KinoRankAgentOrchestrator()
    response = orchestrator.recommend(
        user_id=1,
        query="sci-fi or action movie after 1980 under 150 minutes without horror",
        top_k=8,
    )

    assert len(response.recommendations) == 8
    assert response.parsed_intent is not None
    assert "Horror" in response.parsed_intent.excluded_genres
    assert response.telemetry.retrieval_ms > 0.0
    assert response.telemetry.ranking_ms > 0.0
    assert response.telemetry.total_ms > 0.0
    assert len(response.telemetry.guardrail_rejected_ids) == 0

    watched_set = set(response.user_profile.watched_movie_ids)
    for rec in response.recommendations:
        assert rec.movie_id not in watched_set
        assert "Horror" not in rec.genres
        assert len(rec.explanation) > 20

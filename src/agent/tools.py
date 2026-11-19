"""Agentic Orchestrator and Structured Tools (parse_user_intent, retrieve_and_rank, rerank_and_explain)."""

from __future__ import annotations

import time

from src.agent.guardrails import build_grounded_explanation, enforce_catalog_guardrail
from src.agent.llm_client import MultiProviderLLMClient
from src.agent.telemetry import create_telemetry_context, log_telemetry
from src.config import settings
from src.ranking.diversity import apply_mmr_diversity
from src.ranking.feature_store import FeatureStore
from src.ranking.lgbm_ranker import LightGBMRanker
from src.retrieval.collaborative import CollaborativeALSRetriever
from src.retrieval.hybrid import HybridCandidateRetriever
from src.retrieval.semantic import SemanticFAISSRetriever
from src.schemas.agent import StageTelemetry, UserIntent
from src.schemas.api import RecommendResponse
from src.schemas.domain import RankedRecommendation, UserProfile


class KinoRankAgentOrchestrator:
    """Three-stage Hybrid Recommender + Agentic GenAI Orchestrator."""

    def __init__(
        self,
        feature_store: FeatureStore | None = None,
        als_retriever: CollaborativeALSRetriever | None = None,
        semantic_retriever: SemanticFAISSRetriever | None = None,
        lgbm_ranker: LightGBMRanker | None = None,
        llm_client: MultiProviderLLMClient | None = None,
    ) -> None:
        self.feature_store = feature_store or FeatureStore(try_redis=True)
        self.als = als_retriever or CollaborativeALSRetriever.load()
        self.semantic = semantic_retriever or SemanticFAISSRetriever.load()
        self.ranker = lgbm_ranker or LightGBMRanker.load()
        self.hybrid = HybridCandidateRetriever(
            als_retriever=self.als,
            semantic_retriever=self.semantic,
            popular_fallback_ids=self.feature_store.popular_movie_ids,
        )
        self.llm_client = llm_client or MultiProviderLLMClient()

    # -------------------------------------------------------------------------
    # Tool 1: parse_user_intent
    # -------------------------------------------------------------------------
    def parse_user_intent(
        self,
        query: str | None,
        telemetry: StageTelemetry,
    ) -> UserIntent | None:
        """Tool 1: Parse natural language query into hard and soft recommendation constraints."""
        if not query or not query.strip():
            return None
        return self.llm_client.parse_intent(raw_query=query, telemetry=telemetry)

    # -------------------------------------------------------------------------
    # Tool 2: retrieve_and_rank
    # -------------------------------------------------------------------------
    def retrieve_and_rank(
        self,
        user_profile: UserProfile,
        intent: UserIntent | None = None,
        seed_movie_ids: list[int] | None = None,
        mmr_lambda: float = 0.75,
        stage2_pool_size: int = 15,
        telemetry: StageTelemetry | None = None,
    ) -> list[RankedRecommendation]:
        """Tool 2: Execute Stage 1 Hybrid Retrieval (RRF) + Stage 2 LightGBM Ranker + MMR."""
        search_query = intent.semantic_search_query if intent else None
        fav_ids = list(seed_movie_ids or [])

        # Stage 1: Hybrid Candidate Generation (< 20ms)
        candidates, retrieval_ms, _ = self.hybrid.retrieve_candidates(
            user_id=user_profile.user_id if user_profile.user_id > 0 else None,
            watched_movie_ids=user_profile.watched_movie_ids,
            favorite_movie_ids=fav_ids if fav_ids else None,
            query_text=search_query,
            top_k=settings.retrieval_top_k,
        )
        if telemetry is not None:
            telemetry.retrieval_ms = retrieval_ms
            telemetry.candidates_retrieved = len(candidates)

        # Stage 2: LightGBM LambdaRank + Hard Constraints + MMR Diversity
        t_rank = time.perf_counter()
        ranked_all = self.ranker.rank(
            user_profile=user_profile,
            candidates=candidates,
            feature_store=self.feature_store,
            intent=intent,
            exclude_movie_ids=set(user_profile.watched_movie_ids),
            compute_shap=True,
        )
        if telemetry is not None:
            telemetry.candidates_after_filters = len(ranked_all)

        diversified = apply_mmr_diversity(
            ranked_items=ranked_all[:45],
            semantic_retriever=self.semantic,
            mmr_lambda=mmr_lambda,
            top_k=max(stage2_pool_size, 15),
        )
        ranking_ms = (time.perf_counter() - t_rank) * 1000.0
        if telemetry is not None:
            telemetry.ranking_ms = round(ranking_ms, 3)

        return diversified

    # -------------------------------------------------------------------------
    # Tool 3: rerank_and_explain
    # -------------------------------------------------------------------------
    def rerank_and_explain(
        self,
        stage2_candidates: list[RankedRecommendation],
        user_profile: UserProfile,
        intent: UserIntent | None,
        telemetry: StageTelemetry,
        top_k: int = 10,
        enable_agentic_rerank: bool = True,
    ) -> tuple[list[RankedRecommendation], str]:
        """Tool 3: Contextual Reranking, Factual Grounding, and Strict Anti-Hallucination Guardrail."""
        t_llm = time.perf_counter()

        if not enable_agentic_rerank:
            telemetry.llm_provider_used = "disabled"
            final_items: list[RankedRecommendation] = []
            for idx, item in enumerate(stage2_candidates[:top_k], start=1):
                copy_item = item.model_copy()
                copy_item.rank = idx
                copy_item.explanation = build_grounded_explanation(copy_item, user_profile, intent)
                final_items.append(copy_item)
            telemetry.llm_ms = round((time.perf_counter() - t_llm) * 1000.0, 3)
            return final_items, "Stage 2 LightGBM + MMR ranking (Stage 3 agentic rerank bypassed)."

        agent_output = self.llm_client.rerank_and_explain(
            candidates=stage2_candidates[:15],
            user_profile=user_profile,
            intent=intent,
            telemetry=telemetry,
            top_k=top_k,
        )

        verified_items, rejected_ids = enforce_catalog_guardrail(
            stage2_candidates=stage2_candidates,
            agent_output=agent_output,
            user_profile=user_profile,
            intent=intent,
            top_k=top_k,
        )
        telemetry.guardrail_rejected_ids = rejected_ids
        telemetry.llm_ms = round((time.perf_counter() - t_llm) * 1000.0, 3)
        return verified_items, agent_output.agent_reasoning_summary

    # -------------------------------------------------------------------------
    # End-to-End Recommendation Pipeline Entrypoint
    # -------------------------------------------------------------------------
    def recommend(
        self,
        user_id: int | None = 1,
        query: str | None = None,
        seed_movie_ids: list[int] | None = None,
        custom_user_profile: UserProfile | None = None,
        top_k: int = 10,
        mmr_lambda: float = 0.75,
        enable_agentic_rerank: bool = True,
    ) -> RecommendResponse:
        """Execute full 3-stage pipeline and return RecommendResponse with telemetry."""
        telemetry = create_telemetry_context(feature_store_backend=self.feature_store.backend_name)

        # Resolve UserProfile
        if custom_user_profile is not None:
            user_profile = custom_user_profile
        else:
            user_profile = self.feature_store.get_user_profile(
                user_id=user_id,
                seed_movie_ids=seed_movie_ids,
            )

        # Tool 1: Parse Conversational Intent (if query provided)
        t_intent0 = time.perf_counter()
        intent = self.parse_user_intent(query=query, telemetry=telemetry)
        intent_ms = (time.perf_counter() - t_intent0) * 1000.0

        eff_mmr = intent.diversity_preference if (intent and query) else mmr_lambda

        # Tool 2: Hybrid Retrieval + LightGBM Ranking + MMR
        stage2_candidates = self.retrieve_and_rank(
            user_profile=user_profile,
            intent=intent,
            seed_movie_ids=seed_movie_ids,
            mmr_lambda=eff_mmr,
            stage2_pool_size=max(15, top_k),
            telemetry=telemetry,
        )

        # Tool 3: Agentic Reranking, Explanation & Anti-Hallucination Guardrail
        final_recommendations, agent_summary = self.rerank_and_explain(
            stage2_candidates=stage2_candidates,
            user_profile=user_profile,
            intent=intent,
            telemetry=telemetry,
            top_k=top_k,
            enable_agentic_rerank=enable_agentic_rerank,
        )
        telemetry.llm_ms = round(telemetry.llm_ms + intent_ms, 3)
        telemetry.final_returned = len(final_recommendations)
        log_telemetry(telemetry)

        return RecommendResponse(
            user_id=user_profile.user_id,
            user_profile=user_profile,
            parsed_intent=intent,
            recommendations=final_recommendations,
            agent_summary=agent_summary,
            telemetry=telemetry,
        )

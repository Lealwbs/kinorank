"""Production FastAPI service for KinoRank Hybrid Recommender & Agentic AI."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.agent.telemetry import RECENT_TELEMETRY_EVENTS
from src.agent.tools import KinoRankAgentOrchestrator
from src.data.letterboxd import LetterboxdCatalogMatcher, parse_letterboxd_payload
from src.schemas.api import (
    ColdStartRequest,
    HealthResponse,
    RecommendRequest,
    RecommendResponse,
)


@lru_cache(maxsize=1)
def get_orchestrator() -> KinoRankAgentOrchestrator:
    """Singleton loader for the 3-stage KinoRankAgentOrchestrator."""
    return KinoRankAgentOrchestrator()


@lru_cache(maxsize=1)
def get_letterboxd_matcher() -> LetterboxdCatalogMatcher:
    """Singleton loader for the Letterboxd catalog matcher."""
    return LetterboxdCatalogMatcher.from_default_catalog()


app = FastAPI(
    title="KinoRank Hybrid Recommendation & Agentic AI API",
    description=(
        "Three-Stage Production Recommender: "
        "Stage 1 (ALS Collaborative + FAISS Semantic ANN + RRF) -> "
        "Stage 2 (LightGBM LambdaRank + Redis/DuckDB Feature Store + MMR) -> "
        "Stage 3 (Agentic GenAI Intent Parser, Reranker & Anti-Hallucination Guardrail)."
    ),
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Return service health, catalog size, Feature Store backend, and artifact readiness."""
    orch = get_orchestrator()
    provider_name, _ = orch.llm_client._resolve_active_provider()
    return HealthResponse(
        status="ok",
        version=__version__,
        catalog_size=len(orch.feature_store.items_by_id),
        feature_store_backend=orch.feature_store.backend_name,
        llm_provider_configured=provider_name,
        artifacts_loaded=orch.ranker.booster is not None,
    )


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendation"])
def recommend_endpoint(req: RecommendRequest) -> RecommendResponse:
    """Personalized 3-stage recommendation endpoint for existing or seed-based users."""
    orch = get_orchestrator()
    return orch.recommend(
        user_id=req.user_id,
        query=req.query,
        seed_movie_ids=req.seed_movie_ids,
        top_k=req.top_k,
        mmr_lambda=req.mmr_lambda,
        enable_agentic_rerank=req.enable_agentic_rerank,
    )


@app.post("/chat", response_model=RecommendResponse, tags=["Agentic GenAI"])
def conversational_recommend_endpoint(req: RecommendRequest) -> RecommendResponse:
    """Conversational recommendation endpoint enforcing structured intent parsing and catalog guardrails."""
    if not req.query or not req.query.strip():
        raise HTTPException(
            status_code=422, detail="Conversational /chat requires a non-empty 'query'."
        )
    orch = get_orchestrator()
    return orch.recommend(
        user_id=req.user_id,
        query=req.query,
        seed_movie_ids=req.seed_movie_ids,
        top_k=req.top_k,
        mmr_lambda=req.mmr_lambda,
        enable_agentic_rerank=True,
    )


@app.post("/cold-start", response_model=RecommendResponse, tags=["Cold-Start"])
def cold_start_endpoint(req: ColdStartRequest) -> RecommendResponse:
    """Interactive cold-start recommendation from 3-5 favorite seed films and/or natural language."""
    if not req.seed_movie_ids and not req.natural_language_preferences.strip():
        raise HTTPException(
            status_code=422,
            detail="Provide at least one seed movie ID or a natural language preference.",
        )
    orch = get_orchestrator()
    cold_profile = orch.feature_store.synthesize_cold_start_profile(req.seed_movie_ids)
    return orch.recommend(
        user_id=None,
        query=req.natural_language_preferences or None,
        seed_movie_ids=req.seed_movie_ids,
        custom_user_profile=cold_profile,
        top_k=req.top_k,
        mmr_lambda=req.mmr_lambda,
        enable_agentic_rerank=True,
    )


@app.post("/import-letterboxd", tags=["Letterboxd"])
async def import_letterboxd_endpoint(
    file: UploadFile = File(
        ..., description="Letterboxd export (.zip, ratings.csv, watched.csv, watchlist.csv)"
    ),
    query: str | None = Form(default=None),
    top_k: int = Form(default=10),
    mmr_lambda: float = Form(default=0.75),
) -> dict[str, Any]:
    """Import official Letterboxd CSV or ZIP export, match catalog IDs, and return recommendations."""
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded Letterboxd file is empty.")

    matcher = get_letterboxd_matcher()
    import_result, lb_profile = parse_letterboxd_payload(
        file_bytes=raw_bytes,
        filename=file.filename or "ratings.csv",
        matcher=matcher,
    )
    orch = get_orchestrator()
    rec_response = orch.recommend(
        user_id=None,
        query=query,
        seed_movie_ids=lb_profile.watched_movie_ids[-15:],
        custom_user_profile=lb_profile,
        top_k=top_k,
        mmr_lambda=mmr_lambda,
        enable_agentic_rerank=True,
    )
    return {
        "import_summary": import_result.model_dump(),
        "recommendation_response": rec_response.model_dump(),
    }


@app.get("/catalog/search", tags=["Catalog"])
def search_catalog(
    q: str = Query(..., min_length=1, description="Title search substring"),
    limit: int = Query(default=15, ge=1, le=50),
) -> list[dict[str, Any]]:
    """Fast title search over enriched catalog for cold-start movie picker."""
    orch = get_orchestrator()
    q_low = q.lower().strip()
    matches: list[dict[str, Any]] = []
    for item in orch.feature_store.items_by_id.values():
        title = str(item.get("clean_title") or item.get("title") or "")
        if q_low in title.lower():
            matches.append(
                {
                    "movie_id": int(item["movie_id"]),
                    "title": title,
                    "release_year": item.get("release_year"),
                    "genres": item.get("genres", []),
                    "director": item.get("director", "Unknown"),
                    "bayesian_rating": round(float(item.get("bayesian_rating") or 3.5), 2),
                    "rating_count": int(item.get("rating_count") or 0),
                }
            )
    matches.sort(key=lambda x: (-x["rating_count"], -x["bayesian_rating"]))
    return matches[:limit]


@app.get("/telemetry/recent", tags=["Observability"])
def get_recent_telemetry(limit: int = Query(default=20, ge=1, le=100)) -> list[dict[str, Any]]:
    """Return recent structured JSON telemetry records across all stages."""
    return [t.model_dump() for t in list(RECENT_TELEMETRY_EVENTS)[:limit]]

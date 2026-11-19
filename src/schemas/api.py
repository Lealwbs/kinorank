"""FastAPI request and response contract schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.schemas.agent import StageTelemetry, UserIntent
from src.schemas.domain import RankedRecommendation, UserProfile


class RecommendRequest(BaseModel):
    """Standard personalized recommendation request for an existing or cold-start user."""

    user_id: int | None = Field(
        default=1, description="MovieLens User ID (optional if seed_movie_ids provided)."
    )
    top_k: int = Field(default=10, ge=1, le=50)
    query: str | None = Field(
        default=None, description="Optional natural language constraint or mood."
    )
    seed_movie_ids: list[int] = Field(
        default_factory=list, description="Seed movie IDs for cold-start."
    )
    exclude_watched: bool = True
    mmr_lambda: float = Field(default=0.75, ge=0.0, le=1.0)
    enable_agentic_rerank: bool = True


class ColdStartRequest(BaseModel):
    """Interactive Cold-Start request combining 3-5 favorite movies and/or natural language."""

    seed_movie_ids: list[int] = Field(default_factory=list)
    natural_language_preferences: str = ""
    top_k: int = Field(default=10, ge=1, le=30)
    mmr_lambda: float = Field(default=0.75, ge=0.0, le=1.0)


class RecommendResponse(BaseModel):
    """Full recommendation response including ranked items, parsed intent, and telemetry."""

    user_id: int | None = None
    user_profile: UserProfile | None = None
    parsed_intent: UserIntent | None = None
    recommendations: list[RankedRecommendation]
    agent_summary: str = ""
    telemetry: StageTelemetry


class HealthResponse(BaseModel):
    """System health check response."""

    status: str
    version: str
    catalog_size: int
    feature_store_backend: str
    llm_provider_configured: str
    artifacts_loaded: bool

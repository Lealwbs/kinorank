"""Structured Output schemas for Stage 3 Agentic GenAI tool calling and observability."""

from __future__ import annotations

from pydantic import BaseModel, Field


class UserIntent(BaseModel):
    """Structured representation of hard and soft constraints extracted from natural language."""

    raw_query: str
    semantic_search_query: str = Field(
        description="Dense semantic query capturing mood, plot, themes, and stylistic references."
    )
    required_genres: list[str] = Field(
        default_factory=list,
        description="Genres explicitly requested by the user (normalized to MovieLens genres).",
    )
    excluded_genres: list[str] = Field(
        default_factory=list,
        description="Genres the user explicitly wants to avoid.",
    )
    min_year: int | None = Field(
        default=None,
        description="Minimum release year constraint if specified (e.g. 'after 2010' -> 2010).",
    )
    max_year: int | None = Field(
        default=None,
        description="Maximum release year constraint if specified (e.g. '90s or earlier' -> 1999).",
    )
    max_runtime_minutes: int | None = Field(
        default=None,
        description="Maximum movie duration in minutes if requested (e.g. '< 120 min' -> 120).",
    )
    reference_directors: list[str] = Field(
        default_factory=list,
        description="Directors mentioned as stylistic anchors (e.g. 'Fincher', 'Nolan').",
    )
    reference_titles: list[str] = Field(
        default_factory=list,
        description="Specific movies mentioned as positive anchors.",
    )
    diversity_preference: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="MMR lambda weight: 1.0 = pure relevance, 0.5 = high exploration/diversity.",
    )


class RerankedItemExplanation(BaseModel):
    """Single item in the LLM reranked output, strictly bound to a candidate movie_id."""

    movie_id: int = Field(description="Exact movie_id from the provided candidate list.")
    explanation: str = Field(
        description="Concise, factual 1-2 sentence explanation anchored in user history and query."
    )


class AgentRerankOutput(BaseModel):
    """Structured output returned by the rerank_and_explain tool."""

    ordered_items: list[RerankedItemExplanation] = Field(
        default_factory=list,
        description="Reordered candidates from most relevant to least relevant.",
    )
    agent_reasoning_summary: str = Field(
        default="",
        description="High-level summary of how constraints and user profile were balanced.",
    )


class StageTelemetry(BaseModel):
    """Observability payload logged as structured JSON on every recommendation request."""

    request_id: str
    retrieval_ms: float = 0.0
    ranking_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0
    candidates_retrieved: int = 0
    candidates_after_filters: int = 0
    final_returned: int = 0
    feature_store_backend: str = "duckdb_memory"
    feature_store_hit: bool = True
    llm_provider_used: str = "deterministic"
    llm_fallback_triggered: bool = False
    fallback_reason: str | None = None
    guardrail_rejected_ids: list[int] = Field(default_factory=list)
    estimated_prompt_tokens: int = 0
    estimated_completion_tokens: int = 0

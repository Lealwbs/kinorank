"""Structured JSON observability and telemetry logger for AI stages."""

from __future__ import annotations

import json
import logging
import uuid
from collections import deque

from src.schemas.agent import StageTelemetry

logger = logging.getLogger("kinorank.telemetry")

# Ring buffer of recent telemetry events for real-time inspection in UI and API
RECENT_TELEMETRY_EVENTS: deque[StageTelemetry] = deque(maxlen=100)


def create_telemetry_context(feature_store_backend: str = "duckdb_memory") -> StageTelemetry:
    """Initialize a fresh StageTelemetry record with a unique request_id."""
    return StageTelemetry(
        request_id=str(uuid.uuid4())[:12],
        feature_store_backend=feature_store_backend,
    )


def estimate_tokens(text: str) -> int:
    """Fast heuristic token estimator (~4 chars per token for English/Portuguese)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def log_telemetry(telemetry: StageTelemetry) -> None:
    """Emit structured JSON log line and store in recent ring buffer."""
    telemetry.total_ms = round(
        telemetry.retrieval_ms + telemetry.ranking_ms + telemetry.llm_ms,
        3,
    )
    RECENT_TELEMETRY_EVENTS.appendleft(telemetry)
    payload = telemetry.model_dump()
    logger.info(json.dumps({"event": "kinorank_recommendation_telemetry", **payload}))


def format_telemetry_badge(t: StageTelemetry) -> str:
    """Format a compact editorial telemetry bar string for the UI."""
    fallback_tag = " (Deterministic Fallback)" if t.llm_fallback_triggered else ""
    return (
        f"Retrieval {t.retrieval_ms:.1f}ms · "
        f"LightGBM+MMR {t.ranking_ms:.1f}ms · "
        f"Agent Rerank {t.llm_ms:.1f}ms [{t.llm_provider_used}{fallback_tag}] · "
        f"Total {t.total_ms:.1f}ms · "
        f"FeatureStore: {t.feature_store_backend}"
    )

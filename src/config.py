"""Centralized configuration using Pydantic Settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Normalize NO_PROXY / no_proxy for httpx's get_environment_proxies() parser
for _proxy_var in ("NO_PROXY", "no_proxy"):
    _val = os.environ.get(_proxy_var, "")
    if "::1" in _val:
        _parts = [
            p.strip() for p in _val.split(",") if p.strip() and p.strip() not in ("::1", "[::1]")
        ]
        os.environ[_proxy_var] = ",".join(_parts)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Production configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Dataset & Storage Paths
    movielens_variant: Literal["ml-latest-small", "ml-25m", "ml-32m"] = Field(
        default="ml-latest-small",
        alias="MOVIELENS_VARIANT",
    )
    data_dir: Path = Field(default=PROJECT_ROOT / "data", alias="DATA_DIR")
    artifacts_dir: Path = Field(default=PROJECT_ROOT / "artifacts", alias="ARTIFACTS_DIR")

    # Temporal Split Fractions (per-user chronological split)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    positive_rating_threshold: float = 3.5

    # Stage 1: Hybrid Candidate Retrieval
    als_factors: int = 64
    als_regularization: float = 0.05
    als_iterations: int = 20
    als_alpha: float = 15.0
    semantic_model_name: str = "BAAI/bge-small-en-v1.5"
    retrieval_top_k: int = 100
    rrf_k: int = 60

    # Stage 2: Ranking & Diversity
    ranking_top_k: int = 15
    mmr_lambda: float = 0.75
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # TMDB Enrichment
    tmdb_api_key: str = Field(default="", alias="TMDB_API_KEY")
    tmdb_max_enrich_items: int = Field(default=300, alias="TMDB_MAX_ENRICH_ITEMS")

    # Stage 3: Agentic GenAI Layer
    llm_provider: Literal["gemini", "openai", "groq", "deterministic"] = Field(
        default="gemini",
        alias="LLM_PROVIDER",
    )
    llm_model: str = Field(default="gemini-2.5-flash", alias="LLM_MODEL")
    llm_timeout_seconds: float = Field(default=8.0, alias="LLM_TIMEOUT_SECONDS")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")

    # Distribution (Hugging Face & Kaggle)
    hf_token: str = Field(default="", alias="HF_TOKEN")
    hf_username: str = Field(default="Lealwbs", alias="HF_USERNAME")
    hf_model_repo: str = Field(
        default="Lealwbs/kinorank-hybrid-recsys",
        alias="HF_MODEL_REPO",
    )
    hf_space_repo: str = Field(
        default="Lealwbs/kinorank",
        alias="HF_SPACE_REPO",
    )

    def model_post_init(self, __context: object) -> None:
        if not self.data_dir.is_absolute():
            self.data_dir = (PROJECT_ROOT / self.data_dir).resolve()
        if not self.artifacts_dir.is_absolute():
            self.artifacts_dir = (PROJECT_ROOT / self.artifacts_dir).resolve()

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def bronze_dir(self) -> Path:
        return self.data_dir / "bronze"

    @property
    def silver_dir(self) -> Path:
        return self.data_dir / "silver"

    @property
    def gold_dir(self) -> Path:
        return self.data_dir / "gold"

    @property
    def metrics_dir(self) -> Path:
        return self.artifacts_dir / "metrics"

    def ensure_dirs(self) -> None:
        """Ensure all Medallion and Artifact directories exist."""
        for directory in (
            self.raw_dir,
            self.bronze_dir,
            self.silver_dir,
            self.gold_dir,
            self.artifacts_dir,
            self.metrics_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()

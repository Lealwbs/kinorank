"""Pytest fixtures shared across KinoRank test suites."""

from __future__ import annotations

import pytest

from src.config import settings
from src.data.features_duckdb import run_full_data_pipeline


@pytest.fixture(scope="session", autouse=True)
def ensure_medallion_data() -> None:
    """Ensure Bronze, Silver, and Gold Parquet files exist before running tests."""
    required = [
        settings.bronze_dir / "movies.parquet",
        settings.silver_dir / "catalog.parquet",
        settings.silver_dir / "interactions_train.parquet",
        settings.silver_dir / "interactions_val.parquet",
        settings.silver_dir / "interactions_test.parquet",
        settings.gold_dir / "item_features.parquet",
        settings.gold_dir / "user_features.parquet",
    ]
    if not all(p.exists() for p in required):
        run_full_data_pipeline()

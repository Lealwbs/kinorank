"""Tests for DuckDB analytical SQL features and PySpark distributed ETL parity."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import settings
from src.data.features_pyspark import (
    compute_item_features_spark,
    compute_user_window_features_spark,
    get_spark_session,
)


def test_gold_item_features_integrity() -> None:
    """Validate Gold item_features schema, Bayesian rating bounds, and window ranks."""
    df = pd.read_parquet(settings.gold_dir / "item_features.parquet")
    assert len(df) == 9742

    required_cols = {
        "movie_id",
        "clean_title",
        "release_year",
        "decade",
        "genres",
        "director",
        "runtime_minutes",
        "bayesian_rating",
        "recent_popularity_score",
        "popularity_rank",
        "decade_avg_bayesian_rating",
        "semantic_document",
    }
    assert required_cols.issubset(set(df.columns))
    assert df["bayesian_rating"].between(0.5, 5.0).all()
    assert (df["popularity_rank"] >= 1).all()
    assert df["poster_url"].str.len().min() > 15
    assert df["semantic_document"].str.len().min() > 30


def test_gold_user_features_and_window_functions() -> None:
    """Validate Gold user_features computed via LAG, AVG OVER, ROW_NUMBER, and Time Decay."""
    df = pd.read_parquet(settings.gold_dir / "user_features.parquet")
    assert len(df) == 610

    for row in df.head(25).to_dict(orient="records"):
        assert row["total_interactions"] > 0
        assert 0.5 <= row["mean_rating"] <= 5.0
        assert 0.5 <= row["latest_rolling_5_rating"] <= 5.0
        assert row["avg_inter_event_hours"] >= 0.0
        aff = json.loads(row["genre_affinities_json"])
        assert isinstance(aff, dict)
        assert len(row["profile_summary_text"]) > 30


def test_duckdb_and_pyspark_numerical_parity() -> None:
    """Verify that the PySpark distributed feature pipeline produces identical values to DuckDB SQL."""
    spark = get_spark_session("KinoRank-Test-Parity")
    try:
        train_sdf = spark.read.parquet(
            (settings.silver_dir / "interactions_train.parquet").as_posix()
        )
        catalog_sdf = spark.read.parquet((settings.silver_dir / "catalog.parquet").as_posix())

        spark_items = (
            compute_item_features_spark(spark, train_sdf, catalog_sdf)
            .select("movie_id", "rating_count", "bayesian_rating", "recent_popularity_score")
            .toPandas()
            .sort_values("movie_id")
            .reset_index(drop=True)
        )
        duck_items = (
            pd.read_parquet(settings.gold_dir / "item_features.parquet")[
                ["movie_id", "rating_count", "bayesian_rating", "recent_popularity_score"]
            ]
            .sort_values("movie_id")
            .reset_index(drop=True)
        )

        np.testing.assert_array_equal(spark_items["movie_id"].values, duck_items["movie_id"].values)
        np.testing.assert_array_equal(
            spark_items["rating_count"].values, duck_items["rating_count"].values
        )
        np.testing.assert_allclose(
            spark_items["bayesian_rating"].values,
            duck_items["bayesian_rating"].values,
            rtol=1e-4,
            atol=1e-4,
        )

        spark_users = (
            compute_user_window_features_spark(spark, train_sdf)
            .toPandas()
            .sort_values("user_id")
            .reset_index(drop=True)
        )
        duck_users = (
            pd.read_parquet(settings.gold_dir / "user_features.parquet")
            .sort_values("user_id")
            .reset_index(drop=True)
        )
        np.testing.assert_allclose(
            spark_users["mean_rating"].values,
            duck_users["mean_rating"].values,
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            spark_users["latest_rolling_5_rating"].values,
            duck_users["latest_rolling_5_rating"].values,
            rtol=1e-4,
            atol=1e-4,
        )
    finally:
        spark.stop()

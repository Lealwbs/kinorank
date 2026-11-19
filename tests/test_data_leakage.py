"""Automated anti-data-leakage tests verifying strict chronological splitting."""

from __future__ import annotations

import duckdb
import pandas as pd

from src.config import settings
from src.data.temporal_split import verify_zero_temporal_leakage


def test_zero_temporal_leakage_report() -> None:
    """Verify that per-user max(train_ts) <= min(val_ts) <= min(test_ts) across all 610 users."""
    report = verify_zero_temporal_leakage()
    assert report["train_count"] > 50000
    assert report["val_count"] > 10000
    assert report["test_count"] > 10000
    assert report["temporal_violations"] == 0
    assert report["pair_overlap_violations"] == 0


def test_user_watched_history_excludes_future_splits() -> None:
    """Ensure Gold user_features.watched_movie_ids contains ONLY training interactions, never val/test."""
    user_df = pd.read_parquet(settings.gold_dir / "user_features.parquet")
    val_df = pd.read_parquet(settings.silver_dir / "interactions_val.parquet")
    test_df = pd.read_parquet(settings.silver_dir / "interactions_test.parquet")

    val_pairs = set(zip(val_df["user_id"], val_df["movie_id"], strict=False))
    test_pairs = set(zip(test_df["user_id"], test_df["movie_id"], strict=False))

    leaked_val = 0
    leaked_test = 0
    for row in user_df.to_dict(orient="records"):
        uid = int(row["user_id"])
        for mid in row["watched_movie_ids"]:
            if (uid, int(mid)) in val_pairs:
                leaked_val += 1
            if (uid, int(mid)) in test_pairs:
                leaked_test += 1

    assert leaked_val == 0, (
        f"Found {leaked_val} validation interactions leaked into Gold user features!"
    )
    assert leaked_test == 0, (
        f"Found {leaked_test} test interactions leaked into Gold user features!"
    )


def test_every_test_user_present_in_train() -> None:
    """Confirm no orphan users exist in validation or test sets without training history."""
    con = duckdb.connect(database=":memory:")
    train_pq = (settings.silver_dir / "interactions_train.parquet").as_posix()
    val_pq = (settings.silver_dir / "interactions_val.parquet").as_posix()
    test_pq = (settings.silver_dir / "interactions_test.parquet").as_posix()

    orphan_users = con.execute(
        f"""
        SELECT COUNT(DISTINCT user_id) FROM (
            SELECT user_id FROM '{val_pq}'
            UNION
            SELECT user_id FROM '{test_pq}'
        )
        WHERE user_id NOT IN (SELECT DISTINCT user_id FROM '{train_pq}')
        """
    ).fetchone()[0]
    con.close()
    assert orphan_users == 0

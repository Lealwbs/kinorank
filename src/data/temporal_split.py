"""Strict per-user chronological train/val/test split and anti-data-leakage validator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb

from src.config import settings

logger = logging.getLogger(__name__)


def perform_temporal_split(
    bronze_dir: Path | None = None,
    silver_dir: Path | None = None,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
) -> dict[str, Any]:
    """Split user interactions chronologically per user into train, val, and test Parquet files."""
    b_dir = bronze_dir or settings.bronze_dir
    s_dir = silver_dir or settings.silver_dir
    s_dir.mkdir(parents=True, exist_ok=True)

    tr_ratio = train_ratio if train_ratio is not None else settings.train_ratio
    vl_ratio = val_ratio if val_ratio is not None else settings.val_ratio

    ratings_pq = (b_dir / "ratings.parquet").as_posix()
    train_pq = (s_dir / "interactions_train.parquet").as_posix()
    val_pq = (s_dir / "interactions_val.parquet").as_posix()
    test_pq = (s_dir / "interactions_test.parquet").as_posix()

    con = duckdb.connect(database=":memory:")

    con.execute(
        f"""
        CREATE TEMP TABLE ranked_interactions AS
        WITH deduplicated AS (
            SELECT
                user_id,
                movie_id,
                rating,
                timestamp,
                interaction_dt,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id, movie_id
                    ORDER BY timestamp DESC
                ) AS dup_rn
            FROM '{ratings_pq}'
        ),
        user_ordered AS (
            SELECT
                user_id,
                movie_id,
                rating,
                timestamp,
                interaction_dt,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY timestamp ASC, movie_id ASC
                ) AS chron_rn,
                COUNT(*) OVER (PARTITION BY user_id) AS user_total
            FROM deduplicated
            WHERE dup_rn = 1
        )
        SELECT
            user_id,
            movie_id,
            rating,
            timestamp,
            interaction_dt,
            chron_rn,
            user_total,
            CASE
                WHEN user_total < 5 THEN 'train'
                WHEN chron_rn <= CAST(FLOOR(user_total * {tr_ratio}) AS INTEGER) THEN 'train'
                WHEN chron_rn <= CAST(FLOOR(user_total * ({tr_ratio} + {vl_ratio})) AS INTEGER) THEN 'val'
                ELSE 'test'
            END AS split_bucket
        FROM user_ordered;
        """
    )

    for bucket, out_path in [("train", train_pq), ("val", val_pq), ("test", test_pq)]:
        con.execute(
            f"""
            COPY (
                SELECT
                    user_id,
                    movie_id,
                    rating,
                    timestamp,
                    interaction_dt,
                    chron_rn,
                    user_total
                FROM ranked_interactions
                WHERE split_bucket = '{bucket}'
                ORDER BY user_id, chron_rn
            ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
            """
        )

    leakage_report = verify_zero_temporal_leakage(silver_dir=s_dir, con=con)
    con.close()

    logger.info("Temporal split complete and verified: %s", leakage_report)
    return leakage_report


def verify_zero_temporal_leakage(
    silver_dir: Path | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[str, Any]:
    """Mathematically verify zero temporal or user-item overlap leakage across splits."""
    s_dir = silver_dir or settings.silver_dir
    train_pq = (s_dir / "interactions_train.parquet").as_posix()
    val_pq = (s_dir / "interactions_val.parquet").as_posix()
    test_pq = (s_dir / "interactions_test.parquet").as_posix()

    own_con = con is None
    db = con or duckdb.connect(database=":memory:")

    # 1. Verify chronological boundary per user: max(train.ts) <= min(val.ts) <= min(test.ts)
    temporal_violations = db.execute(
        f"""
        WITH tr AS (
            SELECT user_id, MAX(timestamp) AS max_train_ts, MAX(chron_rn) AS max_train_rn
            FROM '{train_pq}' GROUP BY user_id
        ),
        vl AS (
            SELECT user_id, MIN(timestamp) AS min_val_ts, MAX(timestamp) AS max_val_ts,
                   MIN(chron_rn) AS min_val_rn, MAX(chron_rn) AS max_val_rn
            FROM '{val_pq}' GROUP BY user_id
        ),
        te AS (
            SELECT user_id, MIN(timestamp) AS min_test_ts, MIN(chron_rn) AS min_test_rn
            FROM '{test_pq}' GROUP BY user_id
        )
        SELECT COUNT(*)
        FROM tr
        LEFT JOIN vl ON tr.user_id = vl.user_id
        LEFT JOIN te ON tr.user_id = te.user_id
        WHERE (vl.min_val_ts IS NOT NULL AND tr.max_train_ts > vl.min_val_ts)
           OR (te.min_test_ts IS NOT NULL AND vl.max_val_ts IS NOT NULL AND vl.max_val_ts > te.min_test_ts)
           OR (te.min_test_ts IS NOT NULL AND tr.max_train_ts > te.min_test_ts)
           OR (vl.min_val_rn IS NOT NULL AND tr.max_train_rn >= vl.min_val_rn)
           OR (te.min_test_rn IS NOT NULL AND vl.max_val_rn IS NOT NULL AND vl.max_val_rn >= te.min_test_rn)
        """
    ).fetchone()[0]

    # 2. Verify zero duplicate (user_id, movie_id) across buckets
    pair_overlap_violations = db.execute(
        f"""
        SELECT COUNT(*) FROM (
            SELECT user_id, movie_id FROM '{train_pq}'
            INTERSECT
            SELECT user_id, movie_id FROM '{val_pq}'
            UNION ALL
            SELECT user_id, movie_id FROM '{train_pq}'
            INTERSECT
            SELECT user_id, movie_id FROM '{test_pq}'
            UNION ALL
            SELECT user_id, movie_id FROM '{val_pq}'
            INTERSECT
            SELECT user_id, movie_id FROM '{test_pq}'
        )
        """
    ).fetchone()[0]

    counts = {
        "train_count": db.execute(f"SELECT COUNT(*) FROM '{train_pq}'").fetchone()[0],
        "val_count": db.execute(f"SELECT COUNT(*) FROM '{val_pq}'").fetchone()[0],
        "test_count": db.execute(f"SELECT COUNT(*) FROM '{test_pq}'").fetchone()[0],
        "temporal_violations": int(temporal_violations),
        "pair_overlap_violations": int(pair_overlap_violations),
    }

    if own_con:
        db.close()

    if counts["temporal_violations"] > 0 or counts["pair_overlap_violations"] > 0:
        raise AssertionError(f"Data leakage detected! Report: {counts}")

    return counts

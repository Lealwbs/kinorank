"""Distributed ETL equivalent of Gold Feature Engineering written in PySpark."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import settings

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)


def get_spark_session(app_name: str = "KinoRank-Medallion-Features") -> SparkSession:
    """Create or retrieve a local/cluster PySpark session configured for Parquet analytics."""
    import os

    from pyspark.sql import SparkSession

    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")

    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )


def compute_item_features_spark(
    spark: SparkSession,
    interactions_train_df: DataFrame,
    catalog_df: DataFrame,
    bayesian_prior_m: float = 15.0,
    half_life_days: float = 180.0,
) -> DataFrame:
    """Compute Bayesian average rating, time-decayed popularity, and rank in PySpark."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    decay_lambda = 0.69314718056 / max(half_life_days, 1.0)

    global_stats = interactions_train_df.agg(
        F.avg("rating").alias("global_c"),
        F.max("timestamp").alias("max_train_ts"),
    ).collect()[0]

    global_c = float(global_stats["global_c"] or 3.5)
    max_ts = int(global_stats["max_train_ts"] or 0)

    decayed = interactions_train_df.withColumn(
        "days_elapsed",
        F.greatest(F.lit(0.0), (F.lit(max_ts) - F.col("timestamp")) / F.lit(86400.0)),
    ).withColumn(
        "time_decay_weight",
        F.exp(F.lit(-decay_lambda) * F.col("days_elapsed")),
    )

    item_agg = decayed.groupBy("movie_id").agg(
        F.count("*").alias("rating_count"),
        F.avg("rating").alias("raw_mean_rating"),
        F.coalesce(F.stddev_samp("rating"), F.lit(0.0)).alias("rating_stddev"),
        F.sum(F.col("time_decay_weight") * (F.col("rating") / F.lit(5.0))).alias(
            "recent_popularity_score"
        ),
    )

    joined = (
        catalog_df.join(item_agg, on="movie_id", how="left")
        .withColumn("rating_count", F.coalesce(F.col("rating_count"), F.lit(0)))
        .withColumn("raw_mean_rating", F.coalesce(F.col("raw_mean_rating"), F.lit(global_c)))
        .withColumn("rating_stddev", F.coalesce(F.col("rating_stddev"), F.lit(0.0)))
        .withColumn(
            "bayesian_rating",
            (
                F.col("rating_count") * F.col("raw_mean_rating")
                + F.lit(bayesian_prior_m * global_c)
            )
            / (F.col("rating_count") + F.lit(bayesian_prior_m)),
        )
        .withColumn(
            "recent_popularity_score",
            F.coalesce(F.col("recent_popularity_score"), F.lit(0.0)),
        )
    )

    pop_window = Window.orderBy(
        F.col("recent_popularity_score").desc(),
        F.col("bayesian_rating").desc(),
        F.col("movie_id").asc(),
    )
    decade_window = Window.partitionBy("decade")

    return (
        joined.withColumn("popularity_rank", F.rank().over(pop_window))
        .withColumn("decade_avg_bayesian_rating", F.avg("bayesian_rating").over(decade_window))
        .orderBy("movie_id")
    )


def compute_user_window_features_spark(
    spark: SparkSession,
    interactions_train_df: DataFrame,
) -> DataFrame:
    """Compute per-user Window features (LAG inter-event gap, rolling AVG, activity velocity) in PySpark."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    user_chron_w = Window.partitionBy("user_id").orderBy(
        F.col("timestamp").asc(), F.col("movie_id").asc()
    )
    user_rolling_w = user_chron_w.rowsBetween(-4, 0)
    user_recency_w = Window.partitionBy("user_id").orderBy(
        F.col("timestamp").desc(), F.col("movie_id").desc()
    )

    windowed = (
        interactions_train_df.withColumn("prev_timestamp", F.lag("timestamp", 1).over(user_chron_w))
        .withColumn(
            "inter_event_hours",
            F.when(
                F.col("prev_timestamp").isNotNull(),
                (F.col("timestamp") - F.col("prev_timestamp")) / F.lit(3600.0),
            ),
        )
        .withColumn("rolling_5_avg_rating", F.avg("rating").over(user_rolling_w))
        .withColumn("recency_rn", F.row_number().over(user_recency_w))
    )

    return (
        windowed.groupBy("user_id")
        .agg(
            F.count("*").alias("total_interactions"),
            F.round(F.avg("rating"), 4).alias("mean_rating"),
            F.round(F.coalesce(F.stddev_samp("rating"), F.lit(0.5)), 4).alias("rating_stddev"),
            F.round(
                F.max(
                    F.when(F.col("recency_rn") == 1, F.col("rolling_5_avg_rating"))
                ),
                4,
            ).alias("latest_rolling_5_rating"),
            F.round(F.coalesce(F.avg("inter_event_hours"), F.lit(0.0)), 2).alias(
                "avg_inter_event_hours"
            ),
        )
        .orderBy("user_id")
    )


def run_pyspark_feature_job(
    silver_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, int]:
    """Execute PySpark feature job over Silver Parquet files."""
    s_dir = silver_dir or settings.silver_dir
    out_dir = output_dir or (settings.gold_dir / "spark_verify")
    out_dir.mkdir(parents=True, exist_ok=True)

    spark = get_spark_session()
    try:
        train_df = spark.read.parquet((s_dir / "interactions_train.parquet").as_posix())
        catalog_df = spark.read.parquet((s_dir / "catalog.parquet").as_posix())

        items_df = compute_item_features_spark(spark, train_df, catalog_df)
        users_df = compute_user_window_features_spark(spark, train_df)

        item_count = items_df.count()
        user_count = users_df.count()
        return {"spark_items": item_count, "spark_users": user_count}
    finally:
        spark.stop()

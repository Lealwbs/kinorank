"""Advanced analytical SQL Feature Engineering in DuckDB (CTEs, Window Functions, Time Decay)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import duckdb

from src.config import settings
from src.data.ingestion import run_ingestion
from src.data.temporal_split import perform_temporal_split
from src.data.tmdb_enricher import enrich_catalog

logger = logging.getLogger(__name__)


def build_gold_features_duckdb(
    silver_dir: Path | None = None,
    gold_dir: Path | None = None,
    bayesian_prior_m: float = 15.0,
    half_life_days: float = 180.0,
) -> dict[str, Any]:
    """Execute Gold layer feature engineering using DuckDB analytical SQL over training split."""
    s_dir = silver_dir or settings.silver_dir
    g_dir = gold_dir or settings.gold_dir
    g_dir.mkdir(parents=True, exist_ok=True)

    train_pq = (s_dir / "interactions_train.parquet").as_posix()
    catalog_pq = (s_dir / "catalog.parquet").as_posix()

    item_features_pq = (g_dir / "item_features.parquet").as_posix()
    user_features_pq = (g_dir / "user_features.parquet").as_posix()
    cooccurrence_pq = (g_dir / "cooccurrence_pairs.parquet").as_posix()

    decay_lambda = 0.69314718056 / max(half_life_days, 1.0)

    con = duckdb.connect(database=":memory:")

    # -------------------------------------------------------------------------
    # 1. ITEM FEATURES (Bayesian Average, Time-Decayed Popularity, Window RANK)
    # -------------------------------------------------------------------------
    con.execute(
        f"""
        COPY (
            WITH global_horizon AS (
                SELECT
                    AVG(rating) AS global_mean_rating,
                    MAX(timestamp) AS max_train_ts
                FROM '{train_pq}'
            ),
            decayed_interactions AS (
                SELECT
                    t.movie_id,
                    t.user_id,
                    t.rating,
                    t.timestamp,
                    g.global_mean_rating,
                    EXP(
                        -{decay_lambda} * GREATEST(0.0, (g.max_train_ts - t.timestamp) / 86400.0)
                    ) AS time_decay_weight
                FROM '{train_pq}' t
                CROSS JOIN global_horizon g
            ),
            item_agg AS (
                SELECT
                    movie_id,
                    COUNT(*) AS rating_count,
                    AVG(rating) AS raw_mean_rating,
                    COALESCE(STDDEV_SAMP(rating), 0.0) AS rating_stddev,
                    MAX(global_mean_rating) AS global_c,
                    SUM(time_decay_weight * (rating / 5.0)) AS recent_popularity_score,
                    SUM(CASE WHEN rating >= 4.0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS positive_ratio
                FROM decayed_interactions
                GROUP BY movie_id
            ),
            global_fallback AS (
                SELECT COALESCE(AVG(rating), 3.5) AS fallback_c FROM '{train_pq}'
            ),
            scored_items AS (
                SELECT
                    c.movie_id,
                    c.tmdb_id,
                    c.imdb_id,
                    c.title,
                    c.clean_title,
                    c.release_year,
                    c.decade,
                    c.genres,
                    c.tags,
                    c.director,
                    c.cast_top,
                    c.runtime_minutes,
                    c.overview,
                    c.poster_url,
                    c.semantic_document,
                    COALESCE(a.rating_count, 0) AS rating_count,
                    COALESCE(a.raw_mean_rating, gf.fallback_c) AS raw_mean_rating,
                    COALESCE(a.rating_stddev, 0.0) AS rating_stddev,
                    COALESCE(
                        (a.rating_count * a.raw_mean_rating + {bayesian_prior_m} * a.global_c)
                        / (a.rating_count + {bayesian_prior_m}),
                        gf.fallback_c
                    ) AS bayesian_rating,
                    COALESCE(a.recent_popularity_score, 0.0) AS recent_popularity_score,
                    COALESCE(a.positive_ratio, 0.5) AS positive_ratio
                FROM '{catalog_pq}' c
                LEFT JOIN item_agg a ON c.movie_id = a.movie_id
                CROSS JOIN global_fallback gf
            )
            SELECT
                *,
                RANK() OVER (
                    ORDER BY recent_popularity_score DESC, bayesian_rating DESC, movie_id ASC
                ) AS popularity_rank,
                AVG(bayesian_rating) OVER (PARTITION BY decade) AS decade_avg_bayesian_rating
            FROM scored_items
            ORDER BY movie_id ASC
        ) TO '{item_features_pq}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    # -------------------------------------------------------------------------
    # 2. USER FEATURES (Window LAG, Rolling AVG OVER, Time-Decayed Genre/Decade/Director Affinity)
    # -------------------------------------------------------------------------
    con.execute(
        f"""
        CREATE TEMP TABLE user_windowed AS
        WITH user_horizon AS (
            SELECT
                user_id,
                MAX(timestamp) AS user_max_ts
            FROM '{train_pq}'
            GROUP BY user_id
        )
        SELECT
            t.user_id,
            t.movie_id,
            t.rating,
            t.timestamp,
            c.clean_title,
            c.release_year,
            c.decade,
            c.genres,
            c.director,
            LAG(t.timestamp) OVER (
                PARTITION BY t.user_id
                ORDER BY t.timestamp ASC, t.movie_id ASC
            ) AS prev_timestamp,
            AVG(t.rating) OVER (
                PARTITION BY t.user_id
                ORDER BY t.timestamp ASC, t.movie_id ASC
                ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
            ) AS rolling_5_avg_rating,
            ROW_NUMBER() OVER (
                PARTITION BY t.user_id
                ORDER BY t.timestamp DESC, t.movie_id DESC
            ) AS recency_rn,
            ROW_NUMBER() OVER (
                PARTITION BY t.user_id
                ORDER BY t.rating DESC, t.timestamp DESC, t.movie_id ASC
            ) AS favorite_rn,
            EXP(
                -{decay_lambda} * GREATEST(0.0, (uh.user_max_ts - t.timestamp) / 86400.0)
            ) AS user_decay_weight
        FROM '{train_pq}' t
        JOIN user_horizon uh ON t.user_id = uh.user_id
        JOIN '{catalog_pq}' c ON t.movie_id = c.movie_id;
        """
    )

    # Genre affinity with exponential time decay and rating centering
    con.execute(
        """
        CREATE TEMP TABLE user_genre_affinity AS
        WITH exploded_genres AS (
            SELECT
                user_id,
                UNNEST(genres) AS genre,
                rating,
                user_decay_weight
            FROM user_windowed
        ),
        genre_scores AS (
            SELECT
                user_id,
                genre,
                COUNT(*) AS genre_watches,
                SUM(user_decay_weight * (rating - 2.75)) AS raw_affinity
            FROM exploded_genres
            WHERE genre IS NOT NULL AND genre != ''
            GROUP BY user_id, genre
        ),
        ranked_genres AS (
            SELECT
                user_id,
                genre,
                genre_watches,
                ROUND(raw_affinity, 4) AS affinity_score,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY raw_affinity DESC, genre_watches DESC, genre ASC
                ) AS genre_rn
            FROM genre_scores
        )
        SELECT
            user_id,
            LIST(genre ORDER BY genre_rn) FILTER (WHERE genre_rn <= 4 AND affinity_score > 0) AS top_genres,
            LIST(STRUCT_PACK(genre := genre, score := affinity_score) ORDER BY genre_rn) AS genre_affinities_list
        FROM ranked_genres
        GROUP BY user_id;
        """
    )

    # Favorite decades per user
    con.execute(
        """
        CREATE TEMP TABLE user_decade_affinity AS
        WITH decade_scores AS (
            SELECT
                user_id,
                decade,
                SUM(user_decay_weight * (rating - 2.5)) AS dec_affinity,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY SUM(user_decay_weight * (rating - 2.5)) DESC, COUNT(*) DESC
                ) AS dec_rn
            FROM user_windowed
            WHERE decade IS NOT NULL
            GROUP BY user_id, decade
        )
        SELECT
            user_id,
            LIST(decade ORDER BY dec_rn) FILTER (WHERE dec_rn <= 3) AS favorite_decades
        FROM decade_scores
        GROUP BY user_id;
        """
    )

    # Top directors per user
    con.execute(
        """
        CREATE TEMP TABLE user_director_affinity AS
        WITH dir_scores AS (
            SELECT
                user_id,
                director,
                SUM(user_decay_weight * (rating - 2.75)) AS dir_affinity,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY SUM(user_decay_weight * (rating - 2.75)) DESC, COUNT(*) DESC
                ) AS dir_rn
            FROM user_windowed
            WHERE director IS NOT NULL AND director != 'Unknown'
            GROUP BY user_id, director
        )
        SELECT
            user_id,
            LIST(director ORDER BY dir_rn) FILTER (WHERE dir_rn <= 3 AND dir_affinity > 0) AS top_directors
        FROM dir_scores
        GROUP BY user_id;
        """
    )

    # Assemble final user_features table
    user_df = con.execute(
        """
        WITH base_user AS (
            SELECT
                user_id,
                COUNT(*) AS total_interactions,
                AVG(rating) AS mean_rating,
                COALESCE(STDDEV_SAMP(rating), 0.5) AS rating_stddev,
                MAX(CASE WHEN recency_rn = 1 THEN rolling_5_avg_rating END) AS latest_rolling_5_rating,
                COALESCE(
                     AVG(CASE WHEN prev_timestamp IS NOT NULL THEN (timestamp - prev_timestamp) / 3600.0 END),
                    0.0
                ) AS avg_inter_event_hours,
                COUNT(*) * 1.0 / GREATEST(1.0, (MAX(timestamp) - MIN(timestamp)) / 86400.0 + 1.0) AS activity_velocity,
                LIST(movie_id ORDER BY timestamp ASC, movie_id ASC) AS watched_movie_ids,
                LIST(
                    clean_title || ' (' || COALESCE(CAST(release_year AS VARCHAR), 'N/A') || ')'
                    ORDER BY favorite_rn ASC
                ) FILTER (WHERE favorite_rn <= 5 AND rating >= 3.5) AS recent_favorites
            FROM user_windowed
            GROUP BY user_id
        )
        SELECT
            b.user_id,
            b.total_interactions,
            ROUND(b.mean_rating, 4) AS mean_rating,
            ROUND(b.rating_stddev, 4) AS rating_stddev,
            ROUND(COALESCE(b.latest_rolling_5_rating, b.mean_rating), 4) AS latest_rolling_5_rating,
            ROUND(b.avg_inter_event_hours, 2) AS avg_inter_event_hours,
            ROUND(b.activity_velocity, 4) AS activity_velocity,
            COALESCE(g.top_genres, []::VARCHAR[]) AS top_genres,
            COALESCE(g.genre_affinities_list, []) AS genre_affinities_raw,
            COALESCE(d.favorite_decades, []::INTEGER[]) AS favorite_decades,
            COALESCE(dr.top_directors, []::VARCHAR[]) AS top_directors,
            b.watched_movie_ids,
            COALESCE(b.recent_favorites, []::VARCHAR[]) AS recent_favorites
        FROM base_user b
        LEFT JOIN user_genre_affinity g ON b.user_id = g.user_id
        LEFT JOIN user_decade_affinity d ON b.user_id = d.user_id
        LEFT JOIN user_director_affinity dr ON b.user_id = dr.user_id
        ORDER BY b.user_id ASC
        """
    ).df()

    # Format JSON affinities and human-readable behavioral summary for Stage 3 Agent
    affinities_json_list: list[str] = []
    summaries: list[str] = []

    for row in user_df.to_dict(orient="records"):
        raw_aff = row.get("genre_affinities_raw")
        aff_dict: dict[str, float] = {}
        if raw_aff is not None:
            for item in list(raw_aff):
                if isinstance(item, dict) and "genre" in item:
                    aff_dict[str(item["genre"])] = float(item["score"])
        # Normalize positive affinities to [0, 1] scale for LightGBM ranker lookup
        max_aff = max(aff_dict.values(), default=1.0)
        if max_aff <= 0:
            max_aff = 1.0
        norm_aff = {k: round(max(0.0, v / max_aff), 4) for k, v in aff_dict.items()}
        affinities_json_list.append(json.dumps(norm_aff))

        top_g = list(row["top_genres"]) if row["top_genres"] is not None else []
        fav_dec = [f"{int(d)}s" for d in (row["favorite_decades"] if row["favorite_decades"] is not None else [])]
        top_dir = list(row["top_directors"]) if row["top_directors"] is not None else []
        fav_titles = list(row["recent_favorites"]) if row["recent_favorites"] is not None else []

        summary = (
            f"User #{row['user_id']} ({row['total_interactions']} interactions, avg rating {row['mean_rating']:.2f}/5.0). "
            f"Core genres: {', '.join(top_g) if top_g else 'Eclectic'}. "
            f"Preferred eras: {', '.join(fav_dec) if fav_dec else 'All decades'}. "
        )
        if top_dir:
            summary += f"Auteur affinity: {', '.join(top_dir)}. "
        if fav_titles:
            summary += f"Anchor favorites: {', '.join(fav_titles[:4])}."
        summaries.append(summary)

    user_df["genre_affinities_json"] = affinities_json_list
    user_df["profile_summary_text"] = summaries
    user_df = user_df.drop(columns=["genre_affinities_raw"])
    user_df.to_parquet(user_features_pq, index=False)

    # -------------------------------------------------------------------------
    # 3. ITEM-ITEM CO-OCCURRENCE MATRIX (Top pairs among positive interactions)
    # -------------------------------------------------------------------------
    con.execute(
        f"""
        COPY (
            WITH pos_ratings AS (
                SELECT user_id, movie_id
                FROM '{train_pq}'
                WHERE rating >= 4.0
            ),
            popular_pos AS (
                SELECT movie_id, COUNT(*) AS item_pos_users
                FROM pos_ratings
                GROUP BY movie_id
                HAVING COUNT(*) >= 5
            ),
            filtered_pos AS (
                SELECT p.user_id, p.movie_id
                FROM pos_ratings p
                JOIN popular_pos pp ON p.movie_id = pp.movie_id
            ),
            pair_counts AS (
                SELECT
                    a.movie_id AS source_movie_id,
                    b.movie_id AS target_movie_id,
                    COUNT(*) AS cooccurrence_count
                FROM filtered_pos a
                JOIN filtered_pos b
                  ON a.user_id = b.user_id AND a.movie_id != b.movie_id
                GROUP BY a.movie_id, b.movie_id
                HAVING COUNT(*) >= 3
            ),
            ranked_pairs AS (
                SELECT
                    p.source_movie_id,
                    p.target_movie_id,
                    p.cooccurrence_count,
                    ROUND(
                        p.cooccurrence_count * 1.0 / SQRT(sa.item_pos_users * sb.item_pos_users),
                        4
                    ) AS cosine_cooccurrence,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.source_movie_id
                        ORDER BY p.cooccurrence_count DESC, p.target_movie_id ASC
                    ) AS pair_rn
                FROM pair_counts p
                JOIN popular_pos sa ON p.source_movie_id = sa.movie_id
                JOIN popular_pos sb ON p.target_movie_id = sb.movie_id
            )
            SELECT
                source_movie_id,
                target_movie_id,
                cooccurrence_count,
                cosine_cooccurrence,
                pair_rn
            FROM ranked_pairs
            WHERE pair_rn <= 25
            ORDER BY source_movie_id, pair_rn
        ) TO '{cooccurrence_pq}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    stats = {
        "item_features_count": con.execute(f"SELECT COUNT(*) FROM '{item_features_pq}'").fetchone()[0],
        "user_features_count": len(user_df),
        "cooccurrence_pairs_count": con.execute(f"SELECT COUNT(*) FROM '{cooccurrence_pq}'").fetchone()[0],
    }
    con.close()
    logger.info("Gold layer features built in DuckDB: %s", stats)
    return stats


def run_full_data_pipeline(dataset: str | None = None) -> dict[str, Any]:
    """Run Bronze ingestion -> Silver TMDB enrichment & Temporal split -> Gold DuckDB features."""
    bronze_stats = run_ingestion(dataset=dataset)
    enrich_catalog()
    split_stats = perform_temporal_split()
    gold_stats = build_gold_features_duckdb()
    return {
        "bronze": bronze_stats,
        "silver_split": split_stats,
        "gold": gold_stats,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    summary = run_full_data_pipeline()
    print(f"Medallion Pipeline Summary: {summary}")

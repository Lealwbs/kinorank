"""MovieLens official dataset ingestion into Bronze Parquet layer."""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

import duckdb
import httpx

from src.config import settings

logger = logging.getLogger(__name__)

MOVIELENS_URLS: dict[str, str] = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
    "ml-32m": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
}


def download_movielens(dataset: str = "ml-latest-small", force: bool = False) -> Path:
    """Download and extract official MovieLens zip archive into data/raw/."""
    settings.ensure_dirs()
    if dataset not in MOVIELENS_URLS:
        raise ValueError(f"Unsupported dataset '{dataset}'. Choose from {list(MOVIELENS_URLS)}")

    extract_dir = settings.raw_dir / dataset
    movies_csv = extract_dir / "movies.csv"
    if movies_csv.exists() and not force:
        logger.info("MovieLens dataset already extracted at %s", extract_dir)
        return extract_dir

    zip_path = settings.raw_dir / f"{dataset}.zip"
    url = MOVIELENS_URLS[dataset]
    logger.info("Downloading MovieLens (%s) from %s ...", dataset, url)

    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=1024 * 256):
                    f.write(chunk)

    logger.info("Extracting %s to %s ...", zip_path, settings.raw_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(settings.raw_dir)

    return extract_dir


def ingest_to_bronze(extract_dir: Path, bronze_dir: Path | None = None) -> dict[str, int]:
    """Normalize raw MovieLens CSVs into typed Bronze Parquet tables using DuckDB SQL."""
    target_bronze = bronze_dir or settings.bronze_dir
    target_bronze.mkdir(parents=True, exist_ok=True)

    movies_csv = (extract_dir / "movies.csv").as_posix()
    ratings_csv = (extract_dir / "ratings.csv").as_posix()
    tags_csv = (extract_dir / "tags.csv").as_posix()
    links_csv = (extract_dir / "links.csv").as_posix()

    movies_parquet = (target_bronze / "movies.parquet").as_posix()
    ratings_parquet = (target_bronze / "ratings.parquet").as_posix()
    tags_parquet = (target_bronze / "tags.parquet").as_posix()
    links_parquet = (target_bronze / "links.parquet").as_posix()

    con = duckdb.connect(database=":memory:")

    # 1. Movies Bronze Table: parse release_year, clean_title, and genre array
    con.execute(
        f"""
        COPY (
            WITH raw_movies AS (
                SELECT
                    CAST(movieId AS INTEGER) AS movie_id,
                    TRIM(title) AS raw_title,
                    CASE
                        WHEN genres = '(no genres listed)' OR genres IS NULL THEN []::VARCHAR[]
                        ELSE string_split(genres, '|')
                    END AS genres,
                    regexp_extract(TRIM(title), '\\((\\d{{4}})\\)\\s*$', 1) AS year_str
                FROM read_csv_auto('{movies_csv}', header=true)
            )
            SELECT
                movie_id,
                raw_title AS title,
                TRIM(
                    regexp_replace(
                        regexp_replace(raw_title, '\\s*\\(\\d{{4}}\\)\\s*$', ''),
                        '^(.*),\\s*(The|A|An)$',
                        '\\2 \\1'
                    )
                ) AS clean_title,
                CASE
                    WHEN year_str != '' THEN CAST(year_str AS INTEGER)
                    ELSE NULL
                END AS release_year,
                CASE
                    WHEN year_str != '' THEN (CAST(year_str AS INTEGER) // 10) * 10
                    ELSE NULL
                END AS decade,
                genres
            FROM raw_movies
            ORDER BY movie_id
        ) TO '{movies_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    # 2. Ratings Bronze Table
    con.execute(
        f"""
        COPY (
            SELECT
                CAST(userId AS INTEGER) AS user_id,
                CAST(movieId AS INTEGER) AS movie_id,
                CAST(rating AS DOUBLE) AS rating,
                CAST(timestamp AS BIGINT) AS timestamp,
                to_timestamp(CAST(timestamp AS BIGINT)) AS interaction_dt
            FROM read_csv_auto('{ratings_csv}', header=true)
            WHERE rating IS NOT NULL AND timestamp IS NOT NULL
            ORDER BY user_id, timestamp, movie_id
        ) TO '{ratings_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    # 3. Tags Bronze Table
    con.execute(
        f"""
        COPY (
            SELECT
                CAST(userId AS INTEGER) AS user_id,
                CAST(movieId AS INTEGER) AS movie_id,
                LOWER(TRIM(CAST(tag AS VARCHAR))) AS tag,
                CAST(timestamp AS BIGINT) AS timestamp
            FROM read_csv_auto('{tags_csv}', header=true)
            WHERE tag IS NOT NULL AND TRIM(CAST(tag AS VARCHAR)) != ''
        ) TO '{tags_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    # 4. Links Bronze Table (MovieLens <-> IMDb <-> TMDB ID mapping)
    con.execute(
        f"""
        COPY (
            SELECT
                CAST(movieId AS INTEGER) AS movie_id,
                 'tt' || LPAD(CAST(imdbId AS VARCHAR), 7, '0') AS imdb_id,
                TRY_CAST(tmdbId AS INTEGER) AS tmdb_id
            FROM read_csv_auto('{links_csv}', header=true)
        ) TO '{links_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    counts = {
        "movies": con.execute(f"SELECT COUNT(*) FROM '{movies_parquet}'").fetchone()[0],
        "ratings": con.execute(f"SELECT COUNT(*) FROM '{ratings_parquet}'").fetchone()[0],
        "tags": con.execute(f"SELECT COUNT(*) FROM '{tags_parquet}'").fetchone()[0],
        "links": con.execute(f"SELECT COUNT(*) FROM '{links_parquet}'").fetchone()[0],
    }
    con.close()
    logger.info("Bronze layer ingested successfully: %s", counts)
    return counts


def run_ingestion(dataset: str | None = None, force: bool = False) -> dict[str, int]:
    """End-to-end download and Bronze Parquet conversion."""
    ds = dataset or settings.movielens_variant
    extract_dir = download_movielens(dataset=ds, force=force)
    return ingest_to_bronze(extract_dir=extract_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Ingest MovieLens dataset to Bronze Parquet layer")
    parser.add_argument("--dataset", default=settings.movielens_variant, choices=list(MOVIELENS_URLS))
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()
    stats = run_ingestion(dataset=args.dataset, force=args.force)
    print(f"Bronze Ingestion Summary: {stats}")

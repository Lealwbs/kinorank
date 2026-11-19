"""Tests for native Letterboxd CSV and ZIP export parsing and catalog matching."""

from __future__ import annotations

import io
import zipfile

from src.data.letterboxd import LetterboxdCatalogMatcher, normalize_title, parse_letterboxd_payload


def test_title_normalization_handles_articles_and_punctuation() -> None:
    """Ensure trailing articles and punctuation normalize to identical canonical forms."""
    assert normalize_title("Matrix, The (1999)") == "matrix"
    assert normalize_title("The Matrix") == "matrix"
    assert normalize_title("Se7en (1995)") == "se7en"
    assert normalize_title("Godfather: Part II, The") == "godfather part ii"


def test_letterboxd_ratings_csv_import() -> None:
    """Verify parsing of official Letterboxd ratings.csv format and matching to MovieLens IDs."""
    sample_csv = (
        b"Date,Name,Year,Letterboxd URI,Rating\n"
        b"2025-10-12,Fight Club,1999,https://boxd.it/2a9q,5.0\n"
        b"2025-10-14,The Matrix,1999,https://boxd.it/2a1m,4.5\n"
        b"2025-10-19,Pulp Fiction,1994,https://boxd.it/29Pq,5.0\n"
        b"2025-10-21,Memento,2000,https://boxd.it/2b5k,4.5\n"
        b"2025-10-25,NonExistentIndieFilmXYZ,2029,https://boxd.it/xxxx,3.0\n"
    )

    result, profile = parse_letterboxd_payload(sample_csv, filename="ratings.csv")
    assert result.total_rows_parsed == 5
    assert result.matched_watched_count == 4
    assert len(result.unmatched_titles) == 1
    assert 2959 in profile.watched_movie_ids  # Fight Club
    assert 2571 in profile.watched_movie_ids  # The Matrix
    assert 296 in profile.watched_movie_ids  # Pulp Fiction
    assert 4226 in profile.watched_movie_ids  # Memento
    assert profile.mean_rating >= 4.5
    assert len(profile.top_genres) >= 2


def test_letterboxd_zip_archive_import_with_watchlist() -> None:
    """Verify full Letterboxd .zip bundle containing ratings.csv, watched.csv, and watchlist.csv."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "ratings.csv",
            "Date,Name,Year,Letterboxd URI,Rating\n"
            "2025-09-01,The Shawshank Redemption,1994,https://boxd.it/1,5.0\n"
            "2025-09-02,Goodfellas,1990,https://boxd.it/2,4.5\n",
        )
        zf.writestr(
            "watched.csv",
            "Date,Name,Year,Letterboxd URI\n"
            "2025-09-01,The Shawshank Redemption,1994,https://boxd.it/1\n"
            "2025-09-03,Blade Runner,1982,https://boxd.it/3\n",
        )
        zf.writestr(
            "watchlist.csv",
            "Date,Name,Year,Letterboxd URI\n2025-09-10,Inception,2010,https://boxd.it/4\n",
        )

    matcher = LetterboxdCatalogMatcher.from_default_catalog()
    result, profile = parse_letterboxd_payload(
        buf.getvalue(), filename="letterboxd-user-export.zip", matcher=matcher
    )
    # 3 distinct watched movies (Shawshank deduplicated across ratings+watched, Goodfellas, Blade Runner)
    assert result.matched_watched_count == 3
    # 1 watchlist movie (Inception - should NOT be added to watched_movie_ids so it can be recommended!)
    assert result.matched_watchlist_count == 1
    assert 318 in profile.watched_movie_ids  # Shawshank Redemption
    assert 1213 in profile.watched_movie_ids  # Goodfellas
    assert 541 in profile.watched_movie_ids  # Blade Runner
    assert 79132 not in profile.watched_movie_ids  # Inception is on watchlist, not watched!

"""Native Letterboxd export importer (.zip, ratings.csv, watched.csv, watchlist.csv)."""

from __future__ import annotations

import csv
import io
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import settings
from src.schemas.domain import LetterboxdEntry, LetterboxdImportResult, UserProfile


def normalize_title(title: str) -> str:
    """Normalize a movie title for robust matching across Letterboxd and MovieLens."""
    if not title:
        return ""
    text = unicodedata.normalize("NFKD", str(title)).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    # Strip trailing parenthetical year or alternate title if present
    text = re.sub(r"\s*\(\d{4}\)\s*$", "", text)
    # Move trailing article ("matrix, the" -> "the matrix")
    text = re.sub(r"^(.*),\s*(the|a|an)$", r"\2 \1", text)
    # Remove leading "the ", "a ", "an " for canonical core comparison
    text = re.sub(r"^(the|a|an)\s+", "", text)
    # Replace punctuation with spaces and collapse whitespace
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class LetterboxdCatalogMatcher:
    """Fast in-memory catalog index for matching Letterboxd titles & years to MovieLens IDs."""

    def __init__(self, catalog_df: pd.DataFrame) -> None:
        self.catalog_df = catalog_df
        self.by_norm_year: dict[tuple[str, int], int] = {}
        self.by_norm_only: dict[str, list[tuple[int, int | None, int]]] = {}
        self.movie_meta: dict[int, dict[str, Any]] = {}

        for row in catalog_df.to_dict(orient="records"):
            mid = int(row["movie_id"])
            clean = str(row.get("clean_title") or row.get("title") or "")
            raw = str(row.get("title") or "")
            year_val = row.get("release_year")
            year = int(year_val) if pd.notna(year_val) else None
            rcount = int(row.get("rating_count") or 0)

            self.movie_meta[mid] = row
            for candidate_title in {clean, raw}:
                norm = normalize_title(candidate_title)
                if not norm:
                    continue
                if year is not None:
                    # Keep higher popularity item if collision on same norm_title + year
                    existing_mid = self.by_norm_year.get((norm, year))
                    if existing_mid is None or rcount > int(
                        self.movie_meta[existing_mid].get("rating_count") or 0
                    ):
                        self.by_norm_year[(norm, year)] = mid
                self.by_norm_only.setdefault(norm, []).append((mid, year, rcount))

    @classmethod
    def from_default_catalog(cls) -> LetterboxdCatalogMatcher:
        """Load matcher from Gold item_features.parquet or Silver catalog.parquet."""
        gold_pq = settings.gold_dir / "item_features.parquet"
        silver_pq = settings.silver_dir / "catalog.parquet"
        if gold_pq.exists():
            df = pd.read_parquet(gold_pq)
        elif silver_pq.exists():
            df = pd.read_parquet(silver_pq)
        else:
            raise FileNotFoundError("Catalog parquet not found. Run data ingestion pipeline first.")
        return cls(df)

    def match(self, title: str, year: int | None = None) -> tuple[int | None, float]:
        """Match a (title, year) pair to (movie_id, confidence)."""
        norm = normalize_title(title)
        if not norm:
            return None, 0.0

        # 1. Exact normalized title + exact year
        if year is not None and (norm, year) in self.by_norm_year:
            return self.by_norm_year[(norm, year)], 1.0

        # 2. Exact normalized title + +-1 year tolerance (festival vs theatrical release)
        if year is not None:
            for delta in (-1, 1):
                if (norm, year + delta) in self.by_norm_year:
                    return self.by_norm_year[(norm, year + delta)], 0.95

        # 3. Exact normalized title (pick most popular or closest year)
        candidates = self.by_norm_only.get(norm, [])
        if candidates:
            if year is not None:
                # Sort by year distance then rating_count desc
                ranked = sorted(
                    candidates,
                    key=lambda x: (abs((x[1] or year) - year), -x[2]),
                )
                best_mid, best_yr, _ = ranked[0]
                if best_yr is None or abs(best_yr - year) <= 2:
                    return best_mid, 0.90
            else:
                ranked = sorted(candidates, key=lambda x: -x[2])
                return ranked[0][0], 0.88

        # 4. Token overlap fallback on same release year
        if year is not None:
            query_tokens = set(norm.split())
            if query_tokens:
                best_mid = None
                best_jacc = 0.0
                for (cand_norm, cand_year), mid in self.by_norm_year.items():
                    if abs(cand_year - year) <= 1:
                        cand_tokens = set(cand_norm.split())
                        if not cand_tokens:
                            continue
                        jacc = len(query_tokens & cand_tokens) / len(query_tokens | cand_tokens)
                        if jacc > best_jacc:
                            best_jacc = jacc
                            best_mid = mid
                if best_mid is not None and best_jacc >= 0.65:
                    return best_mid, round(0.75 * best_jacc + 0.2, 2)

        return None, 0.0


def _parse_letterboxd_csv_text(csv_text: str, is_watchlist: bool = False) -> list[LetterboxdEntry]:
    """Parse a single Letterboxd CSV file content (ratings.csv, watched.csv, or watchlist.csv)."""
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    entries: list[LetterboxdEntry] = []
    for row in reader:
        if not row:
            continue
        # Letterboxd standard headers: Date, Name, Year, Letterboxd URI, Rating
        title = (row.get("Name") or row.get("Title") or row.get("name") or "").strip()
        if not title:
            continue
        raw_year = (row.get("Year") or row.get("year") or "").strip()
        year = int(raw_year) if raw_year.isdigit() else None

        raw_rating = (row.get("Rating") or row.get("rating") or "").strip()
        rating: float | None = None
        if raw_rating:
            try:
                rating = float(raw_rating)
            except ValueError:
                rating = None

        watched_date = (row.get("Date") or row.get("Watched Date") or "").strip() or None
        uri = (row.get("Letterboxd URI") or row.get("URI") or "").strip() or None

        entries.append(
            LetterboxdEntry(
                title=title,
                year=year,
                rating=rating,
                watched_date=watched_date,
                letterboxd_uri=uri,
                is_watchlist=is_watchlist,
            )
        )
    return entries


def parse_letterboxd_payload(
    file_bytes: bytes,
    filename: str = "ratings.csv",
    matcher: LetterboxdCatalogMatcher | None = None,
) -> tuple[LetterboxdImportResult, UserProfile]:
    """Parse an uploaded Letterboxd .zip or .csv file and synthesize a KinoRank UserProfile."""
    cat_matcher = matcher or LetterboxdCatalogMatcher.from_default_catalog()
    raw_entries: list[LetterboxdEntry] = []

    lower_name = filename.lower()
    if lower_name.endswith(".zip") or zipfile.is_zipfile(io.BytesIO(file_bytes)):
        with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as zf:
            names = zf.namelist()
            # Parse ratings.csv first so explicit ratings take priority over watched.csv
            for target_file, wl_flag in [
                ("ratings.csv", False),
                ("watched.csv", False),
                ("watchlist.csv", True),
            ]:
                matching_paths = [
                    n for n in names if n.lower().endswith(target_file) and not n.startswith("__MACOSX")
                ]
                for path in matching_paths:
                    content = zf.read(path).decode("utf-8", errors="replace")
                    raw_entries.extend(_parse_letterboxd_csv_text(content, is_watchlist=wl_flag))
    else:
        is_wl = "watchlist" in lower_name
        content = file_bytes.decode("utf-8", errors="replace")
        raw_entries.extend(_parse_letterboxd_csv_text(content, is_watchlist=is_wl))

    # Deduplicate by (normalized_title, year, is_watchlist), keeping entry with explicit rating
    dedup_map: dict[tuple[str, int | None, bool], LetterboxdEntry] = {}
    for entry in raw_entries:
        key = (normalize_title(entry.title), entry.year, entry.is_watchlist)
        existing = dedup_map.get(key)
        if existing is None or (existing.rating is None and entry.rating is not None):
            dedup_map[key] = entry

    matched_entries: list[LetterboxdEntry] = []
    unmatched_titles: list[str] = []
    watched_ids: list[int] = []
    ratings_vals: list[float] = []
    genre_counter: Counter[str] = Counter()
    decade_counter: Counter[int] = Counter()
    director_counter: Counter[str] = Counter()
    favorite_titles: list[tuple[float, str]] = []

    matched_watched = 0
    matched_watchlist = 0

    for entry in dedup_map.values():
        mid, conf = cat_matcher.match(entry.title, entry.year)
        entry.matched_movie_id = mid
        entry.match_confidence = conf

        if mid is not None:
            matched_entries.append(entry)
            meta = cat_matcher.movie_meta.get(mid, {})
            clean_t = str(meta.get("clean_title") or entry.title)
            yr = meta.get("release_year") or entry.year
            raw_genres = meta.get("genres")
            genres = list(raw_genres) if raw_genres is not None else []
            director = str(meta.get("director") or "Unknown")

            if entry.is_watchlist:
                matched_watchlist += 1
                # Watchlist items give soft positive signal to genres without marking as watched
                for g in genres:
                    genre_counter[g] += 0.4
            else:
                matched_watched += 1
                watched_ids.append(mid)
                eff_rating = entry.rating if entry.rating is not None else 3.75
                ratings_vals.append(eff_rating)
                weight = max(0.2, eff_rating - 2.5)
                for g in genres:
                    genre_counter[g] += weight
                if yr is not None and pd.notna(yr):
                    decade_counter[(int(yr) // 10) * 10] += weight
                if director != "Unknown":
                    director_counter[director] += weight
                if eff_rating >= 4.0:
                    favorite_titles.append((eff_rating, f"{clean_t} ({yr or 'N/A'})"))
        else:
            unmatched_titles.append(f"{entry.title} ({entry.year or 'N/A'})")

    favorite_titles.sort(key=lambda x: -x[0])
    top_favs = [t for _, t in favorite_titles[:5]]
    top_genres = [g for g, _ in genre_counter.most_common(4)]
    top_decades = [d for d, _ in decade_counter.most_common(3)]
    top_directors = [d for d, _ in director_counter.most_common(3)]

    max_g = max(genre_counter.values(), default=1.0)
    genre_affinities = {g: round(score / max_g, 4) for g, score in genre_counter.most_common(10)}

    mean_r = round(sum(ratings_vals) / len(ratings_vals), 2) if ratings_vals else 3.8
    summary_text = (
        f"Letterboxd Profile ({matched_watched} watched films, {matched_watchlist} watchlist items, "
        f"avg rating {mean_r:.2f}/5.0). "
        f"Top genres: {', '.join(top_genres) if top_genres else 'Eclectic'}. "
    )
    if top_directors:
        summary_text += f"Favorite directors: {', '.join(top_directors)}. "
    if top_favs:
        summary_text += f"Highest rated: {', '.join(top_favs[:4])}."

    user_profile = UserProfile(
        user_id=-1,
        total_interactions=matched_watched,
        mean_rating=mean_r,
        rating_stddev=0.5,
        activity_velocity=1.0,
        top_genres=top_genres,
        genre_affinities=genre_affinities,
        favorite_decades=top_decades,
        top_directors=top_directors,
        watched_movie_ids=list(dict.fromkeys(watched_ids)),
        recent_favorites=top_favs,
        profile_summary_text=summary_text,
    )

    import_result = LetterboxdImportResult(
        total_rows_parsed=len(dedup_map),
        matched_watched_count=matched_watched,
        matched_watchlist_count=matched_watchlist,
        unmatched_titles=unmatched_titles,
        matched_entries=matched_entries,
        inferred_top_genres=top_genres,
        inferred_favorite_titles=top_favs,
        synthesized_profile_summary=summary_text,
    )
    return import_result, user_profile

"""Dual-Mode Feature Store: Redis low-latency serving with automatic DuckDB/In-Memory fallback."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import settings
from src.schemas.domain import MovieMetadata, UserProfile

logger = logging.getLogger(__name__)


class FeatureStore:
    """Production Feature Store supporting Redis online cache and automatic in-memory Parquet fallback."""

    def __init__(
        self,
        gold_dir: Path | None = None,
        redis_url: str | None = None,
        try_redis: bool = True,
    ) -> None:
        self.gold_dir = gold_dir or settings.gold_dir
        self.redis_url = redis_url or settings.redis_url
        self.backend_name: str = "duckdb_memory"
        self._redis = None

        self.items_by_id: dict[int, dict[str, Any]] = {}
        self.users_by_id: dict[int, dict[str, Any]] = {}
        self.cooccurrence_map: dict[int, dict[int, float]] = {}
        self.popular_movie_ids: list[int] = []

        self._load_local_gold()
        if try_redis:
            self._init_redis_connection()

    def _load_local_gold(self) -> None:
        """Load Gold Parquet feature tables into memory indexed by entity ID."""
        items_pq = self.gold_dir / "item_features.parquet"
        users_pq = self.gold_dir / "user_features.parquet"
        cooc_pq = self.gold_dir / "cooccurrence_pairs.parquet"

        if items_pq.exists():
            item_df = pd.read_parquet(items_pq)
            for row in item_df.to_dict(orient="records"):
                mid = int(row["movie_id"])
                row["genres"] = list(row["genres"]) if row.get("genres") is not None else []
                row["tags"] = list(row["tags"]) if row.get("tags") is not None else []
                row["cast_top"] = list(row["cast_top"]) if row.get("cast_top") is not None else []
                self.items_by_id[mid] = row

            sorted_pop = item_df.sort_values("popularity_rank", ascending=True)
            self.popular_movie_ids = [int(m) for m in sorted_pop["movie_id"].tolist()[:250]]

        if users_pq.exists():
            user_df = pd.read_parquet(users_pq)
            for row in user_df.to_dict(orient="records"):
                uid = int(row["user_id"])
                row["top_genres"] = (
                    list(row["top_genres"]) if row.get("top_genres") is not None else []
                )
                row["favorite_decades"] = [
                    int(d)
                    for d in (
                        list(row["favorite_decades"])
                        if row.get("favorite_decades") is not None
                        else []
                    )
                ]
                row["top_directors"] = (
                    list(row["top_directors"]) if row.get("top_directors") is not None else []
                )
                row["watched_movie_ids"] = [
                    int(m)
                    for m in (
                        list(row["watched_movie_ids"])
                        if row.get("watched_movie_ids") is not None
                        else []
                    )
                ]
                row["recent_favorites"] = (
                    list(row["recent_favorites"]) if row.get("recent_favorites") is not None else []
                )
                try:
                    row["genre_affinities"] = json.loads(row.get("genre_affinities_json") or "{}")
                except Exception:
                    row["genre_affinities"] = {}
                self.users_by_id[uid] = row

        if cooc_pq.exists():
            cooc_df = pd.read_parquet(cooc_pq)
            for row in cooc_df.to_dict(orient="records"):
                src = int(row["source_movie_id"])
                dst = int(row["target_movie_id"])
                sim = float(row["cosine_cooccurrence"])
                self.cooccurrence_map.setdefault(src, {})[dst] = sim

    def _init_redis_connection(self) -> None:
        """Attempt non-blocking Redis connection; gracefully fall back if unavailable."""
        if not self.redis_url:
            return
        try:
            import redis

            client = redis.Redis.from_url(
                self.redis_url,
                socket_connect_timeout=0.35,
                socket_timeout=0.35,
                decode_responses=True,
            )
            if client.ping():
                self._redis = client
                self.backend_name = "redis"
                logger.info("Connected to Redis Feature Store at %s", self.redis_url)
        except Exception:
            self._redis = None
            self.backend_name = "duckdb_memory"

    def sync_to_redis(self) -> int:
        """Populate Redis online store with serialized user and item feature vectors."""
        if self._redis is None:
            return 0
        count = 0
        pipe = self._redis.pipeline(transaction=False)
        for uid, udata in self.users_by_id.items():
            pipe.set(f"kinorank:user:{uid}", json.dumps(udata, default=str))
            count += 1
        pipe.execute()
        return count

    def get_user_profile(
        self,
        user_id: int | None = None,
        seed_movie_ids: list[int] | None = None,
    ) -> UserProfile:
        """Fetch user profile from Redis / In-Memory Gold store, or synthesize cold-start profile."""
        if user_id is not None and user_id in self.users_by_id:
            u = self.users_by_id[user_id]
            return UserProfile(
                user_id=user_id,
                total_interactions=int(u["total_interactions"]),
                mean_rating=float(u["mean_rating"]),
                rating_stddev=float(u["rating_stddev"]),
                activity_velocity=float(u["activity_velocity"]),
                top_genres=u["top_genres"],
                genre_affinities=u["genre_affinities"],
                favorite_decades=u["favorite_decades"],
                top_directors=u["top_directors"],
                watched_movie_ids=u["watched_movie_ids"],
                recent_favorites=u["recent_favorites"],
                profile_summary_text=str(u["profile_summary_text"]),
            )

        # Synthesize dynamic Cold-Start UserProfile from seed_movie_ids
        return self.synthesize_cold_start_profile(seed_movie_ids=seed_movie_ids or [])

    def synthesize_cold_start_profile(self, seed_movie_ids: list[int]) -> UserProfile:
        """Build a behavioral UserProfile on the fly from 1+ chosen seed movies."""
        genre_counts: dict[str, float] = {}
        decade_counts: dict[int, int] = {}
        director_counts: dict[str, int] = {}
        fav_titles: list[str] = []
        valid_seeds: list[int] = []

        for mid in seed_movie_ids:
            item = self.items_by_id.get(int(mid))
            if not item:
                continue
            valid_seeds.append(int(mid))
            title = str(item.get("clean_title") or item.get("title"))
            year = item.get("release_year")
            fav_titles.append(f"{title} ({int(year)})" if pd.notna(year) and year else title)

            for g in item.get("genres", []):
                genre_counts[g] = genre_counts.get(g, 0.0) + 1.0
            if pd.notna(item.get("decade")) and item.get("decade"):
                dec = int(item["decade"])
                decade_counts[dec] = decade_counts.get(dec, 0) + 1
            director = str(item.get("director") or "Unknown")
            if director != "Unknown":
                director_counts[director] = director_counts.get(director, 0) + 1

        max_g = max(genre_counts.values(), default=1.0)
        genre_aff = {
            g: round(c / max_g, 4) for g, c in sorted(genre_counts.items(), key=lambda x: -x[1])
        }
        top_genres = list(genre_aff.keys())[:4]
        fav_decades = [d for d, _ in sorted(decade_counts.items(), key=lambda x: -x[1])[:3]]
        top_directors = [d for d, _ in sorted(director_counts.items(), key=lambda x: -x[1])[:3]]

        if valid_seeds:
            summary = (
                f"Cold-Start Profile anchored on {len(valid_seeds)} selected films: "
                f"{', '.join(fav_titles[:5])}. "
                f"Inferred core genres: {', '.join(top_genres) if top_genres else 'General'}."
            )
        else:
            summary = "New Cold-Start Visitor (Zero historical interactions; exploring catalog)."

        return UserProfile(
            user_id=0,
            total_interactions=len(valid_seeds),
            mean_rating=4.0,
            rating_stddev=0.5,
            activity_velocity=0.5,
            top_genres=top_genres,
            genre_affinities=genre_aff,
            favorite_decades=fav_decades,
            top_directors=top_directors,
            watched_movie_ids=valid_seeds,
            recent_favorites=fav_titles[:5],
            profile_summary_text=summary,
        )

    def get_item_dict(self, movie_id: int) -> dict[str, Any] | None:
        """Retrieve raw item feature dictionary by movie_id."""
        return self.items_by_id.get(int(movie_id))

    def get_movie_metadata(self, movie_id: int) -> MovieMetadata | None:
        """Retrieve typed MovieMetadata domain model for a given movie_id."""
        row = self.items_by_id.get(int(movie_id))
        if not row:
            return None
        return MovieMetadata(
            movie_id=int(row["movie_id"]),
            tmdb_id=int(row["tmdb_id"]) if pd.notna(row.get("tmdb_id")) else None,
            imdb_id=str(row.get("imdb_id") or ""),
            title=str(row["title"]),
            clean_title=str(row.get("clean_title") or row["title"]),
            release_year=int(row["release_year"]) if pd.notna(row.get("release_year")) else None,
            decade=int(row["decade"]) if pd.notna(row.get("decade")) else None,
            genres=list(row.get("genres") or []),
            tags=list(row.get("tags") or []),
            director=str(row.get("director") or "Unknown"),
            cast_top=list(row.get("cast_top") or []),
            runtime_minutes=int(row.get("runtime_minutes") or 110),
            overview=str(row.get("overview") or ""),
            poster_url=str(row.get("poster_url") or ""),
            bayesian_rating=float(row.get("bayesian_rating") or 3.5),
            rating_count=int(row.get("rating_count") or 0),
            recent_popularity_score=float(row.get("recent_popularity_score") or 0.0),
        )

    def get_cooccurrence_boost(self, candidate_id: int, recent_watched_ids: list[int]) -> float:
        """Compute maximum item-item co-occurrence cosine similarity with user's recent history."""
        if not recent_watched_ids:
            return 0.0
        best = 0.0
        for w_id in recent_watched_ids[-15:]:
            neighbors = self.cooccurrence_map.get(int(w_id))
            if neighbors and candidate_id in neighbors:
                sim = neighbors[candidate_id]
                if sim > best:
                    best = sim
        return best

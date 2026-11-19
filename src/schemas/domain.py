"""Core domain models for movies, user profiles, candidates, and Letterboxd imports."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MovieMetadata(BaseModel):
    """Canonical enriched movie record stored in Silver/Gold layers."""

    movie_id: int
    tmdb_id: int | None = None
    imdb_id: str | None = None
    title: str
    clean_title: str
    release_year: int | None = None
    decade: int | None = None
    genres: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    director: str = "Unknown"
    cast_top: list[str] = Field(default_factory=list)
    runtime_minutes: int = 110
    overview: str = ""
    poster_url: str = ""
    bayesian_rating: float = 3.5
    rating_count: int = 0
    recent_popularity_score: float = 0.0


class UserProfile(BaseModel):
    """Behavioral user summary computed from historical interactions with temporal decay."""

    user_id: int
    total_interactions: int = 0
    mean_rating: float = 3.5
    rating_stddev: float = 0.5
    activity_velocity: float = 0.0
    top_genres: list[str] = Field(default_factory=list)
    genre_affinities: dict[str, float] = Field(default_factory=dict)
    favorite_decades: list[int] = Field(default_factory=list)
    top_directors: list[str] = Field(default_factory=list)
    watched_movie_ids: list[int] = Field(default_factory=list)
    recent_favorites: list[str] = Field(default_factory=list)
    profile_summary_text: str = ""


class CandidateItem(BaseModel):
    """Stage 1 retrieval candidate with hybrid scores."""

    movie_id: int
    collab_score: float = 0.0
    collab_rank: int | None = None
    semantic_score: float = 0.0
    semantic_rank: int | None = None
    rrf_score: float = 0.0
    retrieval_source: str = "hybrid"


class RankedRecommendation(BaseModel):
    """Stage 2/3 ranked movie recommendation with feature breakdown and explanation."""

    rank: int
    movie_id: int
    tmdb_id: int | None = None
    title: str
    release_year: int | None = None
    genres: list[str] = Field(default_factory=list)
    director: str = "Unknown"
    runtime_minutes: int = 110
    poster_url: str = ""
    overview: str = ""
    bayesian_rating: float = 3.5
    rrf_score: float = 0.0
    lgbm_score: float = 0.0
    mmr_score: float = 0.0
    final_score: float = 0.0
    feature_contributions: dict[str, float] = Field(default_factory=dict)
    explanation: str = ""


class LetterboxdEntry(BaseModel):
    """Parsed entry from an official Letterboxd export CSV/ZIP."""

    title: str
    year: int | None = None
    rating: float | None = None
    watched_date: str | None = None
    letterboxd_uri: str | None = None
    is_watchlist: bool = False
    matched_movie_id: int | None = None
    match_confidence: float = 0.0


class LetterboxdImportResult(BaseModel):
    """Summary of matching a Letterboxd export against the KinoRank catalog."""

    total_rows_parsed: int
    matched_watched_count: int
    matched_watchlist_count: int
    unmatched_titles: list[str] = Field(default_factory=list)
    matched_entries: list[LetterboxdEntry] = Field(default_factory=list)
    inferred_top_genres: list[str] = Field(default_factory=list)
    inferred_favorite_titles: list[str] = Field(default_factory=list)
    synthesized_profile_summary: str = ""

"""TMDB API v3/v4 Catalog Enricher with Parquet cache and deterministic offline fallback."""

from __future__ import annotations

import html
import logging
import urllib.parse
from pathlib import Path
from typing import Any

import duckdb
import httpx
import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# Canonical reference metadata for landmark MovieLens films (ensures rich director/runtime/synopsis
# anchors even when running 100% offline in CI, Kaggle, or before TMDB_API_KEY is configured).
CANONICAL_FILM_METADATA: dict[int, dict[str, Any]] = {
    1: {
        "director": "John Lasseter",
        "runtime_minutes": 81,
        "cast_top": ["Tom Hanks", "Tim Allen", "Don Rickles"],
        "overview": "Led by Woody, Andy's toys live happily in his room until Buzz Lightyear arrives and challenges Woody's place as top toy.",
        "poster_path": "/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg",
    },
    32: {
        "director": "Terry Gilliam",
        "runtime_minutes": 129,
        "cast_top": ["Bruce Willis", "Madeleine Stowe", "Brad Pitt"],
        "overview": "In a future world devastated by disease, a convict is sent back in time to gather information about the man-made virus that wiped out most of the human population.",
        "poster_path": "/gt3iyguaCIw8DpQZI1LIN5TohM2.jpg",
    },
    47: {
        "director": "David Fincher",
        "runtime_minutes": 127,
        "cast_top": ["Brad Pitt", "Morgan Freeman", "Gwyneth Paltrow"],
        "overview": "Two homicide detectives are on a desperate hunt for a serial killer whose crimes are based on the 'seven deadly sins' in this dark, rain-soaked psychological thriller.",
        "poster_path": "/6yoghtyTpznpBik8EngEmJskVUO.jpg",
    },
    50: {
        "director": "Bryan Singer",
        "runtime_minutes": 106,
        "cast_top": ["Kevin Spacey", "Gabriel Byrne", "Chazz Palminteri"],
        "overview": "Five criminals meet in a seemingly random police lineup and are coerced into a dangerous heist orchestrated by the mythical crime lord Keyser Soze.",
        "poster_path": "/bUPmtQzrRhzqYySeiMpv7GurAfm.jpg",
    },
    111: {
        "director": "Martin Scorsese",
        "runtime_minutes": 114,
        "cast_top": ["Robert De Niro", "Jodie Foster", "Cybill Shepherd"],
        "overview": "A mentally unstable Vietnam War veteran works as a night-time taxi driver in New York City where the perceived decadence and sleaze feed his urge for violent action.",
        "poster_path": "/ekstpH614fwDX8DUln1a2Opz0N8.jpg",
    },
    260: {
        "director": "George Lucas",
        "runtime_minutes": 121,
        "cast_top": ["Mark Hamill", "Harrison Ford", "Carrie Fisher"],
        "overview": "Luke Skywalker joins forces with a Jedi Knight, a cocky pilot, a Wookiee and two droids to save the galaxy from the Empire's world-destroying battle station.",
        "poster_path": "/6FfCtAuVAW8XJjZ7eWeLibRLWTw.jpg",
    },
    296: {
        "director": "Quentin Tarantino",
        "runtime_minutes": 154,
        "cast_top": ["John Travolta", "Samuel L. Jackson", "Uma Thurman"],
        "overview": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption in Los Angeles.",
        "poster_path": "/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg",
    },
    318: {
        "director": "Frank Darabont",
        "runtime_minutes": 142,
        "cast_top": ["Tim Robbins", "Morgan Freeman", "Bob Gunton"],
        "overview": "Imprisoned in the 1940s for the double murder of his wife and her lover, upstanding banker Andy Dufresne begins a new life at the Shawshank prison.",
        "poster_path": "/9cqNxx0GxF0bflZmeSMuL5tnGzr.jpg",
    },
    356: {
        "director": "Robert Zemeckis",
        "runtime_minutes": 142,
        "cast_top": ["Tom Hanks", "Robin Wright", "Gary Sinise"],
        "overview": "A man with a low IQ has accomplished great things in his life and been present during significant historic events, far exceeding what anyone imagined he could do.",
        "poster_path": "/arw2vcBveWOVZr6pxd9XTd1TdQa.jpg",
    },
    527: {
        "director": "Steven Spielberg",
        "runtime_minutes": 195,
        "cast_top": ["Liam Neeson", "Ben Kingsley", "Ralph Fiennes"],
        "overview": "The true story of how businessman Oskar Schindler saved over a thousand Jewish lives from the Nazis while they worked as slaves in his factory during World War II.",
        "poster_path": "/sF1U4EUQS8YHUYjNl3pMGNIQyr0.jpg",
    },
    541: {
        "director": "Ridley Scott",
        "runtime_minutes": 117,
        "cast_top": ["Harrison Ford", "Rutger Hauer", "Sean Young"],
        "overview": "In the smog-choked dystopian Los Angeles of 2019, blade runner Rick Deckard is called out of retirement to terminate a quartet of replicants who have escaped to Earth.",
        "poster_path": "/63N9uy8nd9j7Eog2axPQ8lbr3Wj.jpg",
    },
    589: {
        "director": "James Cameron",
        "runtime_minutes": 137,
        "cast_top": ["Arnold Schwarzenegger", "Linda Hamilton", "Edward Furlong"],
        "overview": "A cyborg, identical to the one who failed to kill Sarah Connor, must now protect her ten-year-old son John from a more advanced and powerful liquid-metal cyborg.",
        "poster_path": "/5M0j0B18abtBI5gi2RhfjjurTqb.jpg",
    },
    593: {
        "director": "Jonathan Demme",
        "runtime_minutes": 118,
        "cast_top": ["Jodie Foster", "Anthony Hopkins", "Scott Glenn"],
        "overview": "Young FBI cadet Clarice Starling must receive the help of incarcerated and manipulative cannibal killer Dr. Hannibal Lecter to help catch another serial killer, Buffalo Bill.",
        "poster_path": "/uS9m8OBk1A8eM9I042bx8XXpqAq.jpg",
    },
    608: {
        "director": "Joel Coen",
        "runtime_minutes": 98,
        "cast_top": ["Frances McDormand", "William H. Macy", "Steve Buscemi"],
        "overview": "Jerry Lundegaard's inept crime falls apart due to his and his henchmen's bungling and the persistent police work of the quite pregnant Marge Gunderson.",
        "poster_path": "/rt7cpEr1uP6RTZykBFhBTcRaKvG.jpg",
    },
    858: {
        "director": "Francis Ford Coppola",
        "runtime_minutes": 175,
        "cast_top": ["Marlon Brando", "Al Pacino", "James Caan"],
        "overview": "Spanning the years 1945 to 1955, a chronicle of the fictional Italian-American Corleone crime family as patriarch Vito Corleone transfers control to his reluctant son Michael.",
        "poster_path": "/3bhkrj58Vtu7enYsRolD1fZdja1.jpg",
    },
    1196: {
        "director": "Irvin Kershner",
        "runtime_minutes": 124,
        "cast_top": ["Mark Hamill", "Harrison Ford", "Carrie Fisher"],
        "overview": "The epic saga continues as Luke Skywalker, in hopes of defeating the evil Galactic Empire, learns the ways of the Jedi from aging master Yoda.",
        "poster_path": "/nNAeTmF4CtdSgMDplXTDPOpYzsX.jpg",
    },
    1198: {
        "director": "Steven Spielberg",
        "runtime_minutes": 115,
        "cast_top": ["Harrison Ford", "Karen Allen", "Paul Freeman"],
        "overview": "When Dr. Indiana Jones, the tweed-suited professor who happens to be a celebrated archaeologist, is hired by the government to locate the legendary Ark of the Covenant.",
        "poster_path": "/ceG9VzoRAVGwivFU403Wc3AHRys.jpg",
    },
    1200: {
        "director": "James Cameron",
        "runtime_minutes": 137,
        "cast_top": ["Sigourney Weaver", "Michael Biehn", "Carrie Henn"],
        "overview": "Ellen Ripley is rescued by a deep salvage team after being in hypersleep for 57 years and returns to LV-426 with colonial marines to confront the xenomorph hive.",
        "poster_path": "/r1x5JGpyqZU8PYhbs4UcrO1Xb6x.jpg",
    },
    1213: {
        "director": "Martin Scorsese",
        "runtime_minutes": 145,
        "cast_top": ["Ray Liotta", "Robert De Niro", "Joe Pesci"],
        "overview": "The true story of Henry Hill, a half-Irish, half-Sicilian Brooklyn kid who is adopted by neighbourhood gangsters at an early age and climbs the ranks of a Mafia family.",
        "poster_path": "/aKuFiU82s5ISJpGZp7YkIr3kCUd.jpg",
    },
    1258: {
        "director": "Stanley Kubrick",
        "runtime_minutes": 146,
        "cast_top": ["Jack Nicholson", "Shelley Duvall", "Danny Lloyd"],
        "overview": "Jack Torrance accepts a caretaker job at the Overlook Hotel, where he, along with his wife Wendy and their psychic son Danny, must live isolated through the winter.",
        "poster_path": "/xazWoLealQwEgqZ89MLZklLZD3k.jpg",
    },
    2571: {
        "director": "Lana Wachowski",
        "runtime_minutes": 136,
        "cast_top": ["Keanu Reeves", "Laurence Fishburne", "Carrie-Anne Moss"],
        "overview": "Set in the 22nd century, The Matrix tells the story of a computer hacker who joins a group of underground insurgents fighting the vast and powerful computers who now rule the earth.",
        "poster_path": "/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg",
    },
    2959: {
        "director": "David Fincher",
        "runtime_minutes": 139,
        "cast_top": ["Edward Norton", "Brad Pitt", "Helena Bonham Carter"],
        "overview": "A ticking-time-bomb insomniac and a slippery soap salesman channel primal male aggression into a shocking new form of therapy that spirals into Project Mayhem.",
        "poster_path": "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
    },
    4226: {
        "director": "Christopher Nolan",
        "runtime_minutes": 113,
        "cast_top": ["Guy Pearce", "Carrie-Anne Moss", "Joe Pantoliano"],
        "overview": "Leonard Shelby is tracking down the man who raped and murdered his wife, while suffering from a rare, untreatable form of short-term memory loss told in reverse chronology.",
        "poster_path": "/yuNs09hvpHVU1cBTCAk9zxsL2oW.jpg",
    },
    4993: {
        "director": "Peter Jackson",
        "runtime_minutes": 178,
        "cast_top": ["Elijah Wood", "Ian McKellen", "Viggo Mortensen"],
        "overview": "Young hobbit Frodo Baggins, after inheriting a mysterious ring from his uncle Bilbo, must leave his home in order to keep it from falling into the hands of its evil creator.",
        "poster_path": "/6oom5QYQ2yQTMJIbnvbkBL9cHo6.jpg",
    },
    5618: {
        "director": "Hayao Miyazaki",
        "runtime_minutes": 125,
        "cast_top": ["Rumi Hiiragi", "Miyu Irino", "Mari Natsuki"],
        "overview": "A young girl, Chihiro, becomes trapped in a strange new world of spirits. When her parents undergo a mysterious transformation, she must call upon the courage she never knew she had.",
        "poster_path": "/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg",
    },
    7361: {
        "director": "Michel Gondry",
        "runtime_minutes": 108,
        "cast_top": ["Jim Carrey", "Kate Winslet", "Kirsten Dunst"],
        "overview": "Joel Barish, heartbroken that his girlfriend underwent a procedure to erase him from her memory, decides to do the same, only to rediscover his love as his memories fade.",
        "poster_path": "/5MwkWH9tYHv3mV9OdYTMR5qreIz.jpg",
    },
    58559: {
        "director": "Christopher Nolan",
        "runtime_minutes": 152,
        "cast_top": ["Christian Bale", "Heath Ledger", "Aaron Eckhart"],
        "overview": "Batman raises the stakes in his war on crime. With the help of Lt. Jim Gordon and District Attorney Harvey Dent, Batman sets out to dismantle the criminal anarchy of the Joker.",
        "poster_path": "/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
    },
    74458: {
        "director": "Martin Scorsese",
        "runtime_minutes": 138,
        "cast_top": ["Leonardo DiCaprio", "Mark Ruffalo", "Ben Kingsley"],
        "overview": "World War II soldier-turned-U.S. Marshal Teddy Daniels investigates the disappearance of a patient from a hospital for the criminally insane on remote Shutter Island.",
        "poster_path": "/4GDy0PHH3VRXUtwK5ysFbg3kEx.jpg",
    },
    79132: {
        "director": "Christopher Nolan",
        "runtime_minutes": 148,
        "cast_top": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Elliot Page"],
        "overview": "Cobb, a skilled thief who commits corporate espionage by infiltrating the subconscious of his targets is offered a chance to regain his old life through inception.",
        "poster_path": "/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg",
    },
    80463: {
        "director": "David Fincher",
        "runtime_minutes": 120,
        "cast_top": ["Jesse Eisenberg", "Andrew Garfield", "Justin Timberlake"],
        "overview": "On a fall night in 2003, Harvard undergrad and computer programming genius Mark Zuckerberg sits down at his computer and heatedly begins working on a new idea that becomes Facebook.",
        "poster_path": "/n0ybibhJtQ5icDqTp8eRytcIHJx.jpg",
    },
    109487: {
        "director": "Christopher Nolan",
        "runtime_minutes": 169,
        "cast_top": ["Matthew McConaughey", "Anne Hathaway", "Jessica Chastain"],
        "overview": "The adventures of a group of explorers who make use of a newly discovered wormhole to surpass the limitations on human space travel and conquer the vast distances of an interstellar voyage.",
        "poster_path": "/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
    },
    112552: {
        "director": "Damien Chazelle",
        "runtime_minutes": 107,
        "cast_top": ["Miles Teller", "J.K. Simmons", "Paul Reiser"],
        "overview": "Under the direction of a ruthless instructor, a talented young drummer begins to pursue perfection at any cost, even his humanity.",
        "poster_path": "/7fn624j5lj3xTme2SgiLCeuedmO.jpg",
    },
    112852: {
        "director": "David Fincher",
        "runtime_minutes": 149,
        "cast_top": ["Ben Affleck", "Rosamund Pike", "Neil Patrick Harris"],
        "overview": "With his wife's disappearance having become the focus of an intense media circus, a man sees the spotlight turned on him when it's suspected that he may not be innocent.",
        "poster_path": "/lv5xShBIDPe7m4ufdlV0IaVLbwO.jpg",
    },
    164179: {
        "director": "Denis Villeneuve",
        "runtime_minutes": 116,
        "cast_top": ["Amy Adams", "Jeremy Renner", "Forest Whitaker"],
        "overview": "Taking place after alien crafts land around the world, an expert linguist is recruited by the military to determine whether they come in peace or are a threat.",
        "poster_path": "/x2FJsf1ElAgr63Y3PNPtJrcmpoe.jpg",
    },
    176371: {
        "director": "Denis Villeneuve",
        "runtime_minutes": 164,
        "cast_top": ["Ryan Gosling", "Harrison Ford", "Ana de Armas"],
        "overview": "Thirty years after the events of the first film, a new blade runner, LAPD Officer K, unearths a long-buried secret that has the potential to plunge what's left of society into chaos.",
        "poster_path": "/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg",
    },
}


def build_editorial_svg_poster(title: str, year: int | None, genres: list[str]) -> str:
    """Create a minimalist editorial SVG poster data-URI for items without TMDB poster image."""
    safe_title = html.escape(title[:36])
    safe_year = str(year) if year else "CLASSIC"
    safe_genre = html.escape(" · ".join(genres[:2]) if genres else "CINEMA")
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 600' width='400' height='600'>"
        "<defs><linearGradient id='g' x1='0' y1='0' x2='0' y2='1'>"
        "<stop offset='0%' stop-color='#18181b'/><stop offset='100%' stop-color='#09090b'/>"
        "</linearGradient></defs>"
        "<rect width='400' height='600' fill='url(#g)'/>"
        "<rect x='24' y='24' width='352' height='552' fill='none' stroke='#27272a' stroke-width='1.5'/>"
        f"<text x='44' y='72' fill='#a1a1aa' font-family='monospace' font-size='13' letter-spacing='2'>{safe_year}</text>"
        f"<text x='44' y='300' fill='#f4f4f5' font-family='sans-serif' font-weight='700' font-size='22'>{safe_title}</text>"
        f"<text x='44' y='540' fill='#71717a' font-family='sans-serif' font-size='13' letter-spacing='1.5'>{safe_genre}</text>"
        "</svg>"
    )
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)


def fetch_tmdb_movie(tmdb_id: int, api_key: str, client: httpx.Client) -> dict[str, Any] | None:
    """Query TMDB API v3/v4 for a single tmdb_id including credits and keywords."""
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    params: dict[str, str] = {"append_to_response": "credits,keywords", "language": "en-US"}
    headers: dict[str, str] = {"Accept": "application/json"}

    if len(api_key) > 40:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        params["api_key"] = api_key

    try:
        resp = client.get(url, params=params, headers=headers)
        if resp.status_code != 200:
            return None
        payload = resp.json()

        crew = payload.get("credits", {}).get("crew", [])
        directors = [c["name"] for c in crew if c.get("job") == "Director"]
        director = directors[0] if directors else "Unknown"

        cast_raw = payload.get("credits", {}).get("cast", [])[:4]
        cast_top = [c["name"] for c in cast_raw if "name" in c]

        kw_list = payload.get("keywords", {}).get("keywords", [])[:8]
        keywords = [k["name"].lower() for k in kw_list if "name" in k]

        poster_path = payload.get("poster_path")
        poster_url = f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else ""

        return {
            "tmdb_id": int(tmdb_id),
            "director": director,
            "cast_top": cast_top,
            "runtime_minutes": int(payload.get("runtime") or 110),
            "overview": str(payload.get("overview") or ""),
            "poster_url": poster_url,
            "tmdb_keywords": keywords,
        }
    except Exception as exc:
        logger.debug("TMDB fetch error for tmdb_id=%s: %s", tmdb_id, exc)
        return None


def enrich_catalog(
    bronze_dir: Path | None = None,
    silver_dir: Path | None = None,
    api_key: str | None = None,
    max_api_fetch: int | None = None,
) -> Path:
    """Build Silver catalog.parquet combining MovieLens Bronze, user tags, and TMDB enrichment."""
    b_dir = bronze_dir or settings.bronze_dir
    s_dir = silver_dir or settings.silver_dir
    s_dir.mkdir(parents=True, exist_ok=True)

    tmdb_key = (api_key if api_key is not None else settings.tmdb_api_key).strip()
    limit_fetch = max_api_fetch if max_api_fetch is not None else settings.tmdb_max_enrich_items

    movies_pq = (b_dir / "movies.parquet").as_posix()
    links_pq = (b_dir / "links.parquet").as_posix()
    tags_pq = (b_dir / "tags.parquet").as_posix()
    ratings_pq = (b_dir / "ratings.parquet").as_posix()
    cache_pq = s_dir / "tmdb_cache.parquet"
    catalog_pq = s_dir / "catalog.parquet"

    con = duckdb.connect(database=":memory:")

    # Aggregate user tags and interaction counts per movie
    base_df = con.execute(
        f"""
        WITH tag_agg AS (
            SELECT
                movie_id,
                list_slice(LIST(DISTINCT tag ORDER BY tag), 1, 15) AS user_tags
            FROM '{tags_pq}'
            GROUP BY movie_id
        ),
        pop_agg AS (
            SELECT
                movie_id,
                COUNT(*) AS total_ratings
            FROM '{ratings_pq}'
            GROUP BY movie_id
        )
        SELECT
            m.movie_id,
            l.tmdb_id,
            l.imdb_id,
            m.title,
            m.clean_title,
            m.release_year,
            m.decade,
            m.genres,
            COALESCE(t.user_tags, []::VARCHAR[]) AS user_tags,
            COALESCE(p.total_ratings, 0) AS total_ratings
        FROM '{movies_pq}' m
        LEFT JOIN '{links_pq}' l ON m.movie_id = l.movie_id
        LEFT JOIN tag_agg t ON m.movie_id = t.movie_id
        LEFT JOIN pop_agg p ON m.movie_id = p.movie_id
        ORDER BY total_ratings DESC, m.movie_id ASC
        """
    ).df()

    # Load existing TMDB cache if present
    cached_records: dict[int, dict[str, Any]] = {}
    if cache_pq.exists():
        cache_df = pd.read_parquet(cache_pq)
        for row in cache_df.to_dict(orient="records"):
            if row.get("tmdb_id") is not None:
                cached_records[int(row["tmdb_id"])] = row

    # Optionally fetch from live TMDB API for top popular movies not in cache
    if tmdb_key and limit_fetch > 0:
        to_fetch: list[int] = []
        for _, row in base_df.iterrows():
            tmdb_id_val = row["tmdb_id"]
            if pd.notna(tmdb_id_val):
                tid = int(tmdb_id_val)
                if tid not in cached_records:
                    to_fetch.append(tid)
                if len(to_fetch) >= limit_fetch:
                    break

        if to_fetch:
            logger.info("Fetching %d movies from live TMDB API...", len(to_fetch))
            new_fetched = 0
            with httpx.Client(timeout=6.0) as client:
                for tid in to_fetch:
                    res = fetch_tmdb_movie(tid, tmdb_key, client)
                    if res:
                        cached_records[tid] = res
                        new_fetched += 1
            if new_fetched > 0 and cached_records:
                pd.DataFrame(list(cached_records.values())).to_parquet(cache_pq, index=False)
                logger.info("Saved %d records to TMDB cache at %s", len(cached_records), cache_pq)

    # Merge canonical, cached TMDB, and deterministic fallback fields
    directors: list[str] = []
    casts: list[list[str]] = []
    runtimes: list[int] = []
    overviews: list[str] = []
    poster_urls: list[str] = []
    combined_tags: list[list[str]] = []
    semantic_texts: list[str] = []

    for row in base_df.to_dict(orient="records"):
        mid = int(row["movie_id"])
        tid = int(row["tmdb_id"]) if pd.notna(row["tmdb_id"]) else None
        clean_title = str(row["clean_title"] or row["title"])
        year = int(row["release_year"]) if pd.notna(row["release_year"]) else None
        genres = list(row["genres"]) if row["genres"] is not None else []
        utags = list(row["user_tags"]) if row["user_tags"] is not None else []

        tmdb_rec = cached_records.get(tid) if tid is not None else None
        canon_rec = CANONICAL_FILM_METADATA.get(mid)

        if tmdb_rec:
            director = str(tmdb_rec.get("director") or "Unknown")
            cast_top = list(tmdb_rec.get("cast_top") or [])
            runtime = int(tmdb_rec.get("runtime_minutes") or 110)
            overview = str(tmdb_rec.get("overview") or "")
            poster_url = str(tmdb_rec.get("poster_url") or "")
            extra_kw = list(tmdb_rec.get("tmdb_keywords") or [])
        elif canon_rec:
            director = canon_rec["director"]
            cast_top = canon_rec["cast_top"]
            runtime = canon_rec["runtime_minutes"]
            overview = canon_rec["overview"]
            poster_url = f"{TMDB_IMAGE_BASE}{canon_rec['poster_path']}"
            extra_kw = []
        else:
            director = "Unknown"
            cast_top = []
            # Estimate realistic runtime heuristic by genre when offline
            if "Animation" in genres or "Children" in genres:
                runtime = 92
            elif "Documentary" in genres:
                runtime = 96
            elif "War" in genres or "Drama" in genres:
                runtime = 124
            else:
                runtime = 108
            overview = ""
            poster_url = ""
            extra_kw = []

        merged_tags = list(dict.fromkeys([*utags, *extra_kw]))[:15]

        if not overview:
            genre_phrase = ", ".join(genres) if genres else "Feature film"
            year_phrase = f" ({year})" if year else ""
            tag_phrase = f" Key themes: {', '.join(merged_tags[:6])}." if merged_tags else ""
            dir_phrase = f" Directed by {director}." if director != "Unknown" else ""
            overview = f"{clean_title}{year_phrase}: {genre_phrase}.{dir_phrase}{tag_phrase}"

        if not poster_url:
            poster_url = build_editorial_svg_poster(clean_title, year, genres)

        # Build dense semantic document for sentence-transformers embedding
        sem_parts = [
            f"Title: {clean_title} ({year or 'Unknown'})",
            f"Genres: {', '.join(genres) if genres else 'General'}",
        ]
        if director != "Unknown":
            sem_parts.append(f"Director: {director}")
        if cast_top:
            sem_parts.append(f"Cast: {', '.join(cast_top)}")
        if merged_tags:
            sem_parts.append(f"Tags: {', '.join(merged_tags)}")
        sem_parts.append(f"Synopsis: {overview}")

        directors.append(director)
        casts.append(cast_top)
        runtimes.append(runtime)
        overviews.append(overview)
        poster_urls.append(poster_url)
        combined_tags.append(merged_tags)
        semantic_texts.append(" | ".join(sem_parts))

    base_df["director"] = directors
    base_df["cast_top"] = casts
    base_df["runtime_minutes"] = runtimes
    base_df["overview"] = overviews
    base_df["poster_url"] = poster_urls
    base_df["tags"] = combined_tags
    base_df["semantic_document"] = semantic_texts
    base_df = base_df.drop(columns=["user_tags", "total_ratings"]).sort_values("movie_id")

    base_df.to_parquet(catalog_pq, index=False)
    con.close()
    logger.info("Silver catalog enriched and saved to %s (%d movies)", catalog_pq, len(base_df))
    return catalog_pq

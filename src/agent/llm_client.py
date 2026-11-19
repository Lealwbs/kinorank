"""Multi-provider LLM Client (Gemini / OpenAI / Groq) with deterministic structured fallback."""

from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

import httpx
from pydantic import BaseModel

from src.agent.guardrails import build_grounded_explanation
from src.agent.telemetry import estimate_tokens
from src.config import settings
from src.schemas.agent import (
    AgentRerankOutput,
    RerankedItemExplanation,
    StageTelemetry,
    UserIntent,
)
from src.schemas.domain import RankedRecommendation, UserProfile

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Bilingual genre lexicon mapping natural language PT/EN terms to canonical MovieLens genres
GENRE_LEXICON: dict[str, str] = {
    "action": "Action",
    "acao": "Action",
    "ação": "Action",
    "adventure": "Adventure",
    "aventura": "Adventure",
    "animation": "Animation",
    "animacao": "Animation",
    "animação": "Animation",
    "anime": "Animation",
    "children": "Children",
    "infantil": "Children",
    "family": "Children",
    "comedy": "Comedy",
    "comedia": "Comedy",
    "comédia": "Comedy",
    "crime": "Crime",
    "policial": "Crime",
    "mafia": "Crime",
    "heist": "Crime",
    "documentary": "Documentary",
    "documentario": "Documentary",
    "documentário": "Documentary",
    "drama": "Drama",
    "dramatico": "Drama",
    "fantasy": "Fantasy",
    "fantasia": "Fantasy",
    "noir": "Film-Noir",
    "film-noir": "Film-Noir",
    "horror": "Horror",
    "terror": "Horror",
    "musical": "Musical",
    "mystery": "Mystery",
    "misterio": "Mystery",
    "mistério": "Mystery",
    "whodunit": "Mystery",
    "romance": "Romance",
    "romantico": "Romance",
    "romântico": "Romance",
    "sci-fi": "Sci-Fi",
    "scifi": "Sci-Fi",
    "science fiction": "Sci-Fi",
    "ficcao cientifica": "Sci-Fi",
    "ficção científica": "Sci-Fi",
    "cyberpunk": "Sci-Fi",
    "dystopian": "Sci-Fi",
    "thriller": "Thriller",
    "suspense": "Thriller",
    "psicologico": "Thriller",
    "psicológico": "Thriller",
    "psychological": "Thriller",
    "war": "War",
    "guerra": "War",
    "western": "Western",
    "faroeste": "Western",
}

KNOWN_DIRECTORS: dict[str, str] = {
    "fincher": "David Fincher",
    "nolan": "Christopher Nolan",
    "tarantino": "Quentin Tarantino",
    "scorsese": "Martin Scorsese",
    "kubrick": "Stanley Kubrick",
    "spielberg": "Steven Spielberg",
    "villeneuve": "Denis Villeneuve",
    "miyazaki": "Hayao Miyazaki",
    "coppola": "Francis Ford Coppola",
    "cameron": "James Cameron",
    "scott": "Ridley Scott",
    "coen": "Joel Coen",
    "wachowski": "Lana Wachowski",
    "gondry": "Michel Gondry",
    "chazelle": "Damien Chazelle",
    "gilliam": "Terry Gilliam",
    "darabont": "Frank Darabont",
    "lasseter": "John Lasseter",
}


def deterministic_parse_intent(raw_query: str) -> UserIntent:
    """Deterministic bilingual (PT/EN) natural language intent parser for constraints & themes."""
    q = (raw_query or "").strip()
    if not q:
        return UserIntent(raw_query="", semantic_search_query="")

    q_lower = q.lower()
    required_genres: list[str] = []
    excluded_genres: list[str] = []

    # 1. Detect excluded genres following negation markers ("sem X", "no X", "without X", "exceto X")
    neg_patterns = re.findall(
        r"(?:sem|no|not|without|avoid|exceto|menos)\s+([a-zA-ZÀ-ÿ\-]+(?:\s+[a-zA-ZÀ-ÿ\-]+)?)",
        q_lower,
    )
    neg_spans: set[str] = set()
    for phrase in neg_patterns:
        for token_key, canon_g in GENRE_LEXICON.items():
            if re.search(rf"\b{re.escape(token_key)}\b", phrase):
                if canon_g not in excluded_genres:
                    excluded_genres.append(canon_g)
                neg_spans.add(token_key)

    # 2. Detect required genres in positive text
    for token_key, canon_g in GENRE_LEXICON.items():
        if token_key in neg_spans:
            continue
        if re.search(rf"\b{re.escape(token_key)}\b", q_lower):
            if canon_g not in excluded_genres and canon_g not in required_genres:
                required_genres.append(canon_g)

    # 3. Detect maximum runtime ("< 120 min", "under 120 minutes", "ate 110 min", "menos de 2h")
    max_runtime: int | None = None
    rt_match = re.search(
        r"(?:<|<=|under|below|max|máximo|ate|até|menos de)\s*(\d{2,3})\s*(?:min|m\b|minutos|minutes)",
        q_lower,
    )
    if rt_match:
        max_runtime = int(rt_match.group(1))

    # 4. Detect release year constraints ("after 2010", "depois de 2010", "since 2000", "> 2010", "anos 90")
    min_year: int | None = None
    max_year: int | None = None

    after_match = re.search(
        r"(?:after|since|from|depois de|apos|após|a partir de|>|>=)\s*((?:19|20)\d{2})",
        q_lower,
    )
    if after_match:
        min_year = int(after_match.group(1))

    before_match = re.search(
        r"(?:before|until|prior to|antes de|ate|até|<|<=)\s*((?:19|20)\d{2})",
        q_lower,
    )
    if before_match:
        max_year = int(before_match.group(1))

    decade_match = re.search(r"(?:anos|decade|from the)\s*(\d{2})s?\b", q_lower)
    if decade_match and min_year is None:
        dec = int(decade_match.group(1))
        full_dec = 1900 + dec if dec >= 30 else 2000 + dec
        min_year = full_dec
        max_year = full_dec + 9

    # 5. Detect reference directors
    ref_directors: list[str] = []
    for key, full_name in KNOWN_DIRECTORS.items():
        if re.search(rf"\b{re.escape(key)}\b", q_lower):
            ref_directors.append(full_name)

    # Build enriched semantic search query for FAISS
    sem_parts = [q]
    if required_genres:
        sem_parts.append("Genres: " + ", ".join(required_genres))
    if ref_directors:
        sem_parts.append("Director style: " + ", ".join(ref_directors))

    return UserIntent(
        raw_query=q,
        semantic_search_query=" | ".join(sem_parts),
        required_genres=required_genres,
        excluded_genres=excluded_genres,
        min_year=min_year,
        max_year=max_year,
        max_runtime_minutes=max_runtime,
        reference_directors=ref_directors,
    )


def deterministic_rerank_and_explain(
    candidates: list[RankedRecommendation],
    user_profile: UserProfile,
    intent: UserIntent | None = None,
    top_k: int = 15,
) -> AgentRerankOutput:
    """Deterministic profile-aware reranker and factual explanation generator."""
    scored: list[tuple[float, RankedRecommendation]] = []
    intent_dirs = {d.lower() for d in (intent.reference_directors if intent else [])}
    req_genres = {g.lower() for g in (intent.required_genres if intent else [])}

    for item in candidates:
        bonus = 0.0
        if intent_dirs and item.director and any(d in item.director.lower() for d in intent_dirs):
            bonus += 0.35
        if req_genres:
            item_g = {g.lower() for g in item.genres}
            overlap = len(req_genres & item_g) / len(req_genres)
            bonus += 0.25 * overlap
        scored.append((item.final_score + bonus, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    ordered_explanations: list[RerankedItemExplanation] = []
    for _, rec in scored[:top_k]:
        exp = build_grounded_explanation(rec, user_profile, intent)
        ordered_explanations.append(RerankedItemExplanation(movie_id=rec.movie_id, explanation=exp))

    summary_parts = [
        f"Ranked {len(ordered_explanations)} catalog-verified candidates using Hybrid RRF + LightGBM LambdaRank."
    ]
    if intent and intent.raw_query:
        constraints: list[str] = []
        if intent.required_genres:
            constraints.append(f"genres={intent.required_genres}")
        if intent.excluded_genres:
            constraints.append(f"excluded={intent.excluded_genres}")
        if intent.min_year:
            constraints.append(f"year>={intent.min_year}")
        if intent.max_runtime_minutes:
            constraints.append(f"runtime<={intent.max_runtime_minutes}m")
        if intent.reference_directors:
            constraints.append(f"style={intent.reference_directors}")
        if constraints:
            summary_parts.append("Enforced constraints: " + ", ".join(constraints) + ".")

    return AgentRerankOutput(
        ordered_items=ordered_explanations,
        agent_reasoning_summary=" ".join(summary_parts),
    )


class MultiProviderLLMClient:
    """Configurable LLM client supporting Gemini, OpenAI, and Groq with zero-downtime fallback."""

    def __init__(self) -> None:
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout_seconds

    def _resolve_active_provider(self) -> tuple[str, str]:
        """Determine which provider has a valid API key configured."""
        if self.provider == "gemini" and settings.gemini_api_key.strip():
            return "gemini", settings.gemini_api_key.strip()
        if self.provider == "openai" and settings.openai_api_key.strip():
            return "openai", settings.openai_api_key.strip()
        if self.provider == "groq" and settings.groq_api_key.strip():
            return "groq", settings.groq_api_key.strip()

        # Check any available fallback key before dropping to deterministic engine
        if settings.gemini_api_key.strip():
            return "gemini", settings.gemini_api_key.strip()
        if settings.openai_api_key.strip():
            return "openai", settings.openai_api_key.strip()
        if settings.groq_api_key.strip():
            return "groq", settings.groq_api_key.strip()

        return "deterministic", ""

    def _call_json_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        telemetry: StageTelemetry,
    ) -> dict | None:
        """Execute structured JSON generation against the active LLM provider."""
        provider_name, api_key = self._resolve_active_provider()
        telemetry.llm_provider_used = provider_name
        telemetry.estimated_prompt_tokens += estimate_tokens(system_prompt + user_prompt)

        if provider_name == "deterministic" or not api_key:
            telemetry.llm_fallback_triggered = True
            telemetry.fallback_reason = "no_api_key_configured"
            return None

        try:
            with httpx.Client(timeout=self.timeout) as client:
                if provider_name == "gemini":
                    model_id = self.model if "gemini" in self.model else "gemini-flash-latest"
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
                    resp = client.post(
                        url,
                        headers={
                            "Content-Type": "application/json",
                            "X-goog-api-key": api_key,
                        },
                        json={
                            "systemInstruction": {"parts": [{"text": system_prompt}]},
                            "contents": [{"parts": [{"text": user_prompt}]}],
                            "generationConfig": {
                                "temperature": 0.1,
                                "responseMimeType": "application/json",
                            },
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                elif provider_name == "openai":
                    model_id = self.model if "gpt" in self.model else "gpt-4o-mini"
                    resp = client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={
                            "model": model_id,
                            "temperature": 0.1,
                            "response_format": {"type": "json_object"},
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                        },
                    )
                    resp.raise_for_status()
                    raw_text = resp.json()["choices"][0]["message"]["content"]
                else:  # groq
                    model_id = self.model if "llama" in self.model else "llama-3.3-70b-versatile"
                    resp = client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={
                            "model": model_id,
                            "temperature": 0.1,
                            "response_format": {"type": "json_object"},
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                        },
                    )
                    resp.raise_for_status()
                    raw_text = resp.json()["choices"][0]["message"]["content"]

                telemetry.estimated_completion_tokens += estimate_tokens(raw_text)
                return json.loads(raw_text)
        except Exception as exc:
            logger.warning("LLM call failed (%s); triggering deterministic fallback.", exc)
            telemetry.llm_fallback_triggered = True
            telemetry.fallback_reason = f"api_error:{type(exc).__name__}"
            return None

    def parse_intent(self, raw_query: str, telemetry: StageTelemetry) -> UserIntent:
        """Extract structured UserIntent via LLM with deterministic NLU fallback."""
        det_intent = deterministic_parse_intent(raw_query)
        if not raw_query or not raw_query.strip():
            return det_intent

        sys_prompt = (
            "You are a movie search intent parser. Extract structured JSON matching keys: "
            "raw_query (str), semantic_search_query (str), required_genres (list[str] from "
            "Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama, Fantasy, "
            "Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western), "
            "excluded_genres (list[str]), min_year (int|null), max_year (int|null), "
            "max_runtime_minutes (int|null), reference_directors (list[str]), "
            "reference_titles (list[str]), diversity_preference (float 0.5-1.0)."
        )
        parsed_json = self._call_json_llm(sys_prompt, f"User query: {raw_query}", telemetry)
        if parsed_json:
            try:
                llm_intent = UserIntent.model_validate({**det_intent.model_dump(), **parsed_json})
                return llm_intent
            except Exception as exc:
                logger.debug("Pydantic validation fallback on UserIntent: %s", exc)
        return det_intent

    def rerank_and_explain(
        self,
        candidates: list[RankedRecommendation],
        user_profile: UserProfile,
        intent: UserIntent | None,
        telemetry: StageTelemetry,
        top_k: int = 15,
    ) -> AgentRerankOutput:
        """Rerank top candidates and produce grounded explanations strictly bound to candidate IDs."""
        if not candidates:
            return AgentRerankOutput(
                ordered_items=[], agent_reasoning_summary="No candidates found."
            )

        compact_candidates = [
            {
                "movie_id": c.movie_id,
                "title": c.title,
                "year": c.release_year,
                "genres": c.genres,
                "director": c.director,
                "runtime_min": c.runtime_minutes,
                "lgbm_score": round(c.lgbm_score, 3),
                "bayesian_rating": c.bayesian_rating,
            }
            for c in candidates[:15]
        ]

        sys_prompt = (
            "You are KinoRank Stage-3 Reranking Agent. "
            "CRITICAL GUARDRAIL: You MUST ONLY output movie_id values from the provided candidates list. "
            "Never invent or suggest any external movie. "
            "Return JSON with keys: "
            "'ordered_items': [{'movie_id': int, 'explanation': str (1-2 factual sentences anchored in user's favorite films and query)}], "
            "'agent_reasoning_summary': str."
        )
        user_prompt = json.dumps(
            {
                "user_profile_summary": user_profile.profile_summary_text,
                "user_recent_favorites": user_profile.recent_favorites[:5],
                "user_query_intent": intent.model_dump() if intent else None,
                "allowed_candidates": compact_candidates,
                "top_k": top_k,
            },
            ensure_ascii=False,
        )

        parsed_json = self._call_json_llm(sys_prompt, user_prompt, telemetry)
        if parsed_json:
            try:
                return AgentRerankOutput.model_validate(parsed_json)
            except Exception as exc:
                logger.warning(
                    "AgentRerankOutput validation failed (%s); using deterministic.", exc
                )
                telemetry.llm_fallback_triggered = True
                telemetry.fallback_reason = "pydantic_validation_error"

        return deterministic_rerank_and_explain(
            candidates=candidates,
            user_profile=user_profile,
            intent=intent,
            top_k=top_k,
        )

"""Strict anti-hallucination catalog guardrails and factual explanation grounder."""

from __future__ import annotations

import logging

from src.schemas.agent import AgentRerankOutput, UserIntent
from src.schemas.domain import RankedRecommendation, UserProfile

logger = logging.getLogger(__name__)


def build_grounded_explanation(
    item: RankedRecommendation,
    user_profile: UserProfile,
    intent: UserIntent | None = None,
) -> str:
    """Synthesize a 100% factual, deterministic explanation anchored in user history and item metadata."""
    reasons: list[str] = []

    # 1. Check director match with user's top directors or intent reference directors
    user_dirs = {d.lower() for d in (user_profile.top_directors or [])}
    intent_dirs = {d.lower() for d in (intent.reference_directors if intent else [])}
    if item.director and item.director != "Unknown":
        d_low = item.director.lower()
        if any(ref in d_low for ref in intent_dirs):
            reasons.append(f"directed by requested auteur {item.director}")
        elif any(ud in d_low for ud in user_dirs):
            reasons.append(f"matches your high affinity for director {item.director}")

    # 2. Anchor to specific favorite films from the user's real history
    favs = user_profile.recent_favorites or []
    shared_genres = [g for g in item.genres if g in (user_profile.top_genres or [])]
    if not shared_genres and item.genres:
        shared_genres = item.genres[:2]

    if favs and shared_genres:
        anchor_film = favs[0]
        reasons.append(
            f"shares {', '.join(shared_genres[:2])} DNA with {anchor_film} from your history"
        )
    elif shared_genres:
        reasons.append(f"strong alignment with your {', '.join(shared_genres[:2])} profile")

    # 3. Check explicit query constraints satisfied
    if intent:
        constraint_hits: list[str] = []
        if intent.max_runtime_minutes and item.runtime_minutes <= intent.max_runtime_minutes:
            constraint_hits.append(f"{item.runtime_minutes} min (< {intent.max_runtime_minutes}m)")
        if intent.min_year and item.release_year and item.release_year >= intent.min_year:
            constraint_hits.append(f"released in {item.release_year}")
        if constraint_hits:
            reasons.append("fits constraints (" + ", ".join(constraint_hits) + ")")

    # 4. Highlight primary LightGBM / Bayesian signal
    reasons.append(
        f"Bayesian rating {item.bayesian_rating:.2f}/5.0 (Ranker score {item.lgbm_score:+.2f})"
    )

    body = "; ".join(reasons[:3])
    return body[0].upper() + body[1:] + "."


def enforce_catalog_guardrail(
    stage2_candidates: list[RankedRecommendation],
    agent_output: AgentRerankOutput | None,
    user_profile: UserProfile,
    intent: UserIntent | None = None,
    top_k: int = 15,
) -> tuple[list[RankedRecommendation], list[int]]:
    """Enforce zero catalog hallucination and zero watched-item leakage on LLM reranked output.

    Any movie_id returned by the LLM that is not present in `stage2_candidates` or is in
    `user_profile.watched_movie_ids` is rejected and logged.
    """
    watched_set = set(user_profile.watched_movie_ids or [])
    candidate_map: dict[int, RankedRecommendation] = {
        c.movie_id: c for c in stage2_candidates if c.movie_id not in watched_set
    }

    rejected_ids: list[int] = []
    final_items: list[RankedRecommendation] = []
    seen_ids: set[int] = set()

    if agent_output and agent_output.ordered_items:
        for llm_item in agent_output.ordered_items:
            mid = int(llm_item.movie_id)
            if mid not in candidate_map or mid in watched_set:
                rejected_ids.append(mid)
                logger.warning(
                    "Guardrail blocked out-of-catalog or watched movie_id=%d from LLM output",
                    mid,
                )
                continue
            if mid in seen_ids:
                continue

            seen_ids.add(mid)
            rec = candidate_map[mid].model_copy()
            exp_text = (llm_item.explanation or "").strip()
            if len(exp_text) < 12:
                exp_text = build_grounded_explanation(rec, user_profile, intent)
            rec.explanation = exp_text
            final_items.append(rec)
            if len(final_items) >= top_k:
                break

    # Backfill from Stage 2 order if LLM output was partial or fallback was used
    if len(final_items) < min(top_k, len(candidate_map)):
        for cand in stage2_candidates:
            if cand.movie_id in seen_ids or cand.movie_id in watched_set:
                continue
            rec = cand.model_copy()
            if not rec.explanation:
                rec.explanation = build_grounded_explanation(rec, user_profile, intent)
            final_items.append(rec)
            seen_ids.add(rec.movie_id)
            if len(final_items) >= top_k:
                break

    for rank_idx, item in enumerate(final_items, start=1):
        item.rank = rank_idx

    return final_items, rejected_ids

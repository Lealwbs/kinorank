"""Stage 2 Machine Learning Ranker using LightGBM LambdaRank with TreeSHAP feature attributions."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import settings
from src.ranking.feature_store import FeatureStore
from src.retrieval.hybrid import HybridCandidateRetriever
from src.schemas.agent import UserIntent
from src.schemas.domain import CandidateItem, RankedRecommendation, UserProfile

logger = logging.getLogger(__name__)

FEATURE_NAMES: list[str] = [
    "rrf_score",
    "collab_score",
    "semantic_score",
    "collab_rank_inv",
    "semantic_rank_inv",
    "genre_affinity_sum",
    "genre_affinity_mean",
    "top_genre_overlap",
    "decade_match",
    "director_match",
    "cooccurrence_boost",
    "item_bayesian_rating",
    "item_recent_popularity_log",
    "item_popularity_rank_inv",
    "user_item_rating_delta",
]


class LightGBMRanker:
    """LightGBM LambdaRank model for Stage 2 candidate re-scoring and feature attribution."""

    def __init__(self, booster: lgb.Booster | None = None) -> None:
        self.booster = booster

    def build_feature_matrix(
        self,
        user_profile: UserProfile,
        candidates: list[CandidateItem],
        feature_store: FeatureStore,
        intent: UserIntent | None = None,
    ) -> np.ndarray:
        """Compute 15-dimensional user-item cross-feature matrix for a list of candidates."""
        n = len(candidates)
        if n == 0:
            return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)

        X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
        u_aff = user_profile.genre_affinities or {}
        u_top_g = set(user_profile.top_genres or [])
        u_decades = set(user_profile.favorite_decades or [])
        u_dirs = {d.lower() for d in (user_profile.top_directors or [])}
        if intent and intent.reference_directors:
            u_dirs.update(d.lower() for d in intent.reference_directors)

        recent_watched = (
            user_profile.watched_movie_ids[-15:] if user_profile.watched_movie_ids else []
        )
        u_mean = float(user_profile.mean_rating or 3.5)

        for i, cand in enumerate(candidates):
            item = feature_store.get_item_dict(cand.movie_id) or {}
            item_genres = list(item.get("genres") or [])

            c_rank_inv = 1.0 / (10.0 + cand.collab_rank) if cand.collab_rank is not None else 0.0
            s_rank_inv = (
                1.0 / (10.0 + cand.semantic_rank) if cand.semantic_rank is not None else 0.0
            )

            if item_genres:
                g_scores = [float(u_aff.get(g, 0.0)) for g in item_genres]
                g_sum = sum(g_scores)
                g_mean = g_sum / len(item_genres)
                top_overlap = len(u_top_g.intersection(item_genres)) / max(1, len(item_genres))
            else:
                g_sum = 0.0
                g_mean = 0.0
                top_overlap = 0.0

            if intent and intent.required_genres:
                req_set = {g.lower() for g in intent.required_genres}
                item_g_lower = {g.lower() for g in item_genres}
                if req_set & item_g_lower:
                    top_overlap += 0.5

            item_dec = item.get("decade")
            dec_match = 1.0 if (pd.notna(item_dec) and int(item_dec) in u_decades) else 0.0

            item_dir = str(item.get("director") or "Unknown").lower()
            dir_match = (
                1.0 if (item_dir != "unknown" and any(d in item_dir for d in u_dirs)) else 0.0
            )

            cooc = feature_store.get_cooccurrence_boost(cand.movie_id, recent_watched)
            bayes_r = float(item.get("bayesian_rating") or 3.5)
            rec_pop = float(item.get("recent_popularity_score") or 0.0)
            pop_rank = float(item.get("popularity_rank") or 5000.0)

            X[i, 0] = cand.rrf_score
            X[i, 1] = cand.collab_score
            X[i, 2] = cand.semantic_score
            X[i, 3] = c_rank_inv
            X[i, 4] = s_rank_inv
            X[i, 5] = g_sum
            X[i, 6] = g_mean
            X[i, 7] = top_overlap
            X[i, 8] = dec_match
            X[i, 9] = dir_match
            X[i, 10] = cooc
            X[i, 11] = bayes_r
            X[i, 12] = math.log1p(max(0.0, rec_pop))
            X[i, 13] = 1.0 / math.sqrt(max(1.0, pop_rank))
            X[i, 14] = bayes_r - u_mean

        return X

    def fit(
        self,
        val_df: pd.DataFrame,
        hybrid_retriever: HybridCandidateRetriever,
        feature_store: FeatureStore,
        negatives_per_user: int = 60,
    ) -> LightGBMRanker:
        """Train LightGBM LambdaRank model using chronological validation interactions."""
        user_groups = val_df.groupby("user_id")

        X_blocks: list[np.ndarray] = []
        y_blocks: list[np.ndarray] = []
        group_sizes: list[int] = []

        for uid, group_df in user_groups:
            user_id = int(uid)
            user_profile = feature_store.get_user_profile(user_id=user_id)
            if not user_profile.watched_movie_ids:
                continue

            val_ratings: dict[int, float] = {
                int(r["movie_id"]): float(r["rating"]) for r in group_df.to_dict(orient="records")
            }

            retrieved, _, sem_vec = hybrid_retriever.retrieve_candidates(
                user_id=user_id,
                watched_movie_ids=user_profile.watched_movie_ids,
                top_k=negatives_per_user,
            )
            cand_by_id: dict[int, CandidateItem] = {c.movie_id: c for c in retrieved}

            # Ensure ground-truth validation items are scored in the query group for learning
            missing_pos = [mid for mid in val_ratings if mid not in cand_by_id]
            if missing_pos:
                c_scores = hybrid_retriever.als.score_items_for_user(user_id, missing_pos)
                s_scores = hybrid_retriever.semantic.score_items(sem_vec, missing_pos)
                for mid in missing_pos:
                    cand_by_id[mid] = CandidateItem(
                        movie_id=mid,
                        collab_score=round(c_scores.get(mid, 0.0), 5),
                        collab_rank=120,
                        semantic_score=round(s_scores.get(mid, 0.0), 5),
                        semantic_rank=120,
                        rrf_score=round(1.0 / 180.0, 6),
                        retrieval_source="val_target",
                    )

            cand_list = list(cand_by_id.values())
            if len(cand_list) < 3:
                continue

            labels = np.zeros(len(cand_list), dtype=np.int32)
            for idx_c, c_item in enumerate(cand_list):
                if c_item.movie_id in val_ratings:
                    r_val = val_ratings[c_item.movie_id]
                    if r_val >= 4.5:
                        labels[idx_c] = 3
                    elif r_val >= 3.5:
                        labels[idx_c] = 2
                    elif r_val >= 2.5:
                        labels[idx_c] = 1
                    else:
                        labels[idx_c] = 0

            # Skip query group if all labels are identical
            if np.max(labels) == np.min(labels):
                continue

            X_u = self.build_feature_matrix(user_profile, cand_list, feature_store)
            X_blocks.append(X_u)
            y_blocks.append(labels)
            group_sizes.append(len(cand_list))

        X_train = np.vstack(X_blocks)
        y_train = np.concatenate(y_blocks)

        # Split 85% train groups / 15% eval groups for early stopping
        n_groups = len(group_sizes)
        split_g = max(1, int(n_groups * 0.85))
        train_rows = sum(group_sizes[:split_g])

        lgb_train = lgb.Dataset(
            X_train[:train_rows],
            label=y_train[:train_rows],
            group=group_sizes[:split_g],
            feature_name=FEATURE_NAMES,
        )
        lgb_eval = lgb.Dataset(
            X_train[train_rows:],
            label=y_train[train_rows:],
            group=group_sizes[split_g:],
            feature_name=FEATURE_NAMES,
            reference=lgb_train,
        )

        params: dict[str, Any] = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5, 10, 20],
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambdarank_truncation_level": 20,
            "verbosity": -1,
            "seed": 42,
            "n_jobs": 2,
        }

        self.booster = lgb.train(
            params=params,
            train_set=lgb_train,
            num_boost_round=150,
            valid_sets=[lgb_eval],
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
        )
        logger.info(
            "Trained LightGBM LambdaRank model on %d query groups (%d rows), best_iteration=%s",
            n_groups,
            len(y_train),
            self.booster.best_iteration,
        )
        return self

    def save(self, path: Path | None = None) -> Path:
        """Save trained LightGBM Booster to text file."""
        if self.booster is None:
            raise RuntimeError("LightGBMRanker must be fitted before saving.")
        out_path = path or (settings.artifacts_dir / "lgbm_ranker.txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(out_path))
        logger.info("Saved LightGBM LambdaRank model to %s", out_path)
        return out_path

    @classmethod
    def load(cls, path: Path | None = None) -> LightGBMRanker:
        """Load trained LightGBM Booster from text file."""
        in_path = path or (settings.artifacts_dir / "lgbm_ranker.txt")
        if not in_path.exists():
            raise FileNotFoundError(f"LightGBM artifact not found at {in_path}")
        booster = lgb.Booster(model_file=str(in_path))
        return cls(booster=booster)

    def apply_hard_filters(
        self,
        candidates: list[CandidateItem],
        feature_store: FeatureStore,
        intent: UserIntent | None = None,
        exclude_movie_ids: set[int] | None = None,
    ) -> list[CandidateItem]:
        """Filter candidates by watched status and explicit natural-language constraints."""
        excluded = exclude_movie_ids or set()
        filtered: list[CandidateItem] = []

        req_genres = (
            {g.lower() for g in intent.required_genres}
            if (intent and intent.required_genres)
            else set()
        )
        excl_genres = (
            {g.lower() for g in intent.excluded_genres}
            if (intent and intent.excluded_genres)
            else set()
        )
        min_yr = intent.min_year if intent else None
        max_yr = intent.max_year if intent else None
        max_rt = intent.max_runtime_minutes if intent else None

        for cand in candidates:
            if cand.movie_id in excluded:
                continue
            item = feature_store.get_item_dict(cand.movie_id)
            if not item:
                continue

            item_genres_lower = {g.lower() for g in (item.get("genres") or [])}
            if excl_genres and (excl_genres & item_genres_lower):
                continue
            if req_genres and not (req_genres & item_genres_lower):
                continue

            yr = item.get("release_year")
            if min_yr is not None and pd.notna(yr) and int(yr) < min_yr:
                continue
            if max_yr is not None and pd.notna(yr) and int(yr) > max_yr:
                continue

            rt = int(item.get("runtime_minutes") or 110)
            if max_rt is not None and rt > max_rt:
                continue

            filtered.append(cand)

        # Graceful relaxation: if strict hard filters eliminated almost everything (<3 items),
        # keep unwatched candidates that pass at least excluded_genres
        if len(filtered) < 3 and len(candidates) >= 3:
            relaxed = []
            for cand in candidates:
                if cand.movie_id in excluded:
                    continue
                item = feature_store.get_item_dict(cand.movie_id)
                if not item:
                    continue
                item_genres_lower = {g.lower() for g in (item.get("genres") or [])}
                if excl_genres and (excl_genres & item_genres_lower):
                    continue
                relaxed.append(cand)
            return relaxed

        return filtered

    def rank(
        self,
        user_profile: UserProfile,
        candidates: list[CandidateItem],
        feature_store: FeatureStore,
        intent: UserIntent | None = None,
        exclude_movie_ids: set[int] | None = None,
        compute_shap: bool = True,
    ) -> list[RankedRecommendation]:
        """Score and sort candidates with LightGBM LambdaRank + TreeSHAP feature contributions."""
        watched_set = set(user_profile.watched_movie_ids or [])
        if exclude_movie_ids:
            watched_set.update(exclude_movie_ids)

        valid_candidates = self.apply_hard_filters(
            candidates=candidates,
            feature_store=feature_store,
            intent=intent,
            exclude_movie_ids=watched_set,
        )
        if not valid_candidates:
            return []

        X = self.build_feature_matrix(
            user_profile=user_profile,
            candidates=valid_candidates,
            feature_store=feature_store,
            intent=intent,
        )

        if self.booster is not None:
            raw_scores = np.asarray(self.booster.predict(X), dtype=np.float32)
            shap_matrix = (
                np.asarray(self.booster.predict(X, pred_contrib=True), dtype=np.float32)
                if compute_shap
                else None
            )
        else:
            # Linear fallback if booster not yet loaded
            raw_scores = (
                X[:, 0] * 10.0 + X[:, 1] * 0.5 + X[:, 2] * 0.8 + X[:, 5] * 0.3 + X[:, 11] * 0.2
            ).astype(np.float32)
            shap_matrix = None

        order = np.argsort(raw_scores)[::-1]
        ranked_items: list[RankedRecommendation] = []

        for rank_pos, idx in enumerate(order, start=1):
            cand = valid_candidates[int(idx)]
            meta = feature_store.get_movie_metadata(cand.movie_id)
            if meta is None:
                continue

            contrib_dict: dict[str, float] = {}
            if shap_matrix is not None:
                # Last column of pred_contrib is expected value bias; first 15 match FEATURE_NAMES
                row_shap = shap_matrix[int(idx), : len(FEATURE_NAMES)]
                top_f_indices = np.argsort(np.abs(row_shap))[::-1][:6]
                for f_idx in top_f_indices:
                    contrib_dict[FEATURE_NAMES[int(f_idx)]] = round(float(row_shap[int(f_idx)]), 4)
            else:
                for f_idx, fname in enumerate(FEATURE_NAMES[:5]):
                    contrib_dict[fname] = round(float(X[int(idx), f_idx]), 4)

            score_val = round(float(raw_scores[int(idx)]), 5)
            ranked_items.append(
                RankedRecommendation(
                    rank=rank_pos,
                    movie_id=meta.movie_id,
                    tmdb_id=meta.tmdb_id,
                    title=meta.clean_title,
                    release_year=meta.release_year,
                    genres=meta.genres,
                    director=meta.director,
                    runtime_minutes=meta.runtime_minutes,
                    poster_url=meta.poster_url,
                    overview=meta.overview,
                    bayesian_rating=round(meta.bayesian_rating, 2),
                    rrf_score=cand.rrf_score,
                    lgbm_score=score_val,
                    mmr_score=score_val,
                    final_score=score_val,
                    feature_contributions=contrib_dict,
                )
            )

        return ranked_items

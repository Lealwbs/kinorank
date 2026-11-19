"""Formal Offline Scientific Benchmark comparing 5 recommendation architectures on held-out future data."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import settings
from src.evaluation.metrics import (
    average_precision_at_k,
    catalog_coverage_pct,
    hit_rate_at_k,
    intra_list_diversity,
    ndcg_at_k,
    recall_at_k,
)
from src.ranking.diversity import apply_mmr_diversity
from src.ranking.feature_store import FeatureStore
from src.ranking.lgbm_ranker import LightGBMRanker
from src.retrieval.collaborative import CollaborativeALSRetriever
from src.retrieval.hybrid import HybridCandidateRetriever
from src.retrieval.semantic import SemanticFAISSRetriever

logger = logging.getLogger(__name__)


def _plot_benchmark_results(results: list[dict[str, Any]], out_path: Path) -> None:
    """Generate publication-grade editorial benchmark chart comparing the 5 architectures."""
    plt.rcParams.update(
        {
            "figure.facecolor": "#09090b",
            "axes.facecolor": "#18181b",
            "axes.edgecolor": "#27272a",
            "axes.labelcolor": "#e4e4e7",
            "text.color": "#f4f4f5",
            "xtick.color": "#a1a1aa",
            "ytick.color": "#a1a1aa",
            "grid.color": "#27272a",
            "font.family": "sans-serif",
            "font.size": 10,
        }
    )

    models = [r["model"] for r in results]
    short_names = [
        "Popularity\nBaseline",
        "Collaborative\n(ALS)",
        "Semantic\n(FAISS)",
        "Hybrid 2-Stage\n(RRF+LightGBM)",
        "Hybrid + MMR\n(Full KinoRank)",
    ][: len(models)]

    ndcg = [r["NDCG@10"] for r in results]
    recall = [r["Recall@20"] for r in results]
    map10 = [r["MAP@10"] for r in results]
    cold_hr = [r["ColdStart_HitRate@10"] for r in results]
    cov = [r["Catalog_Coverage_%"] for r in results]
    ild = [r["ILD@10"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=160)
    fig.suptitle(
        "KinoRank: Offline Benchmark (Temporal Split)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Accuracy Metrics (NDCG@10, MAP@10, Recall@20)
    x = np.arange(len(short_names))
    w = 0.26
    ax1 = axes[0]
    ax1.bar(x - w, ndcg, width=w, label="NDCG@10", color="#f59e0b")
    ax1.bar(x, map10, width=w, label="MAP@10", color="#38bdf8")
    ax1.bar(x + w, recall, width=w, label="Recall@20", color="#10b981")
    ax1.set_title("Ranking Accuracy (Held-out Test Split)", fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, fontsize=8.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="#09090b", edgecolor="#27272a")

    # Panel 2: Cold-Start HitRate@10 (3 Seed Movies)
    ax2 = axes[1]
    colors_p2 = ["#52525b", "#38bdf8", "#a855f7", "#10b981", "#f59e0b"][: len(short_names)]
    bars = ax2.bar(x, cold_hr, color=colors_p2, width=0.52)
    ax2.set_title("Cold-Start HitRate@10 (3 Seed Films)", fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, fontsize=8.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars:
        h = b.get_height()
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            h + 0.008,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#e4e4e7",
        )

    # Panel 3: Precision (NDCG@10) vs Diversity & Catalog Coverage Frontier
    ax3 = axes[2]
    scatter_sizes = [max(80.0, c * 18.0) for c in cov]
    for idx, name in enumerate(models):
        ax3.scatter(
            ild[idx],
            ndcg[idx],
            s=scatter_sizes[idx],
            color=colors_p2[idx],
            alpha=0.9,
            edgecolors="#f4f4f5",
            linewidth=1.2,
            label=f"{name} (Cov: {cov[idx]:.1f}%)",
        )
    ax3.set_title("Precision (NDCG@10) vs. Diversity (ILD@10)", fontweight="bold", pad=10)
    ax3.set_xlabel("Intra-List Diversity (ILD@10)")
    ax3.set_ylabel("NDCG@10")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(
        loc="lower left", fontsize=7.8, frameon=True, facecolor="#09090b", edgecolor="#27272a"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_offline_benchmark() -> dict[str, Any]:
    """Evaluate all 5 architectures on held-out temporal test set and export tables + charts."""
    settings.ensure_dirs()
    train_df = pd.read_parquet(settings.silver_dir / "interactions_train.parquet")
    val_df = pd.read_parquet(settings.silver_dir / "interactions_val.parquet")
    test_df = pd.read_parquet(settings.silver_dir / "interactions_test.parquet")

    fs = FeatureStore(try_redis=False)
    als = CollaborativeALSRetriever.load()
    sem = SemanticFAISSRetriever.load()
    ranker = LightGBMRanker.load()
    hybrid = HybridCandidateRetriever(als, sem, fs.popular_movie_ids)

    # Active catalog = items with at least 5 interactions in train (fair denominator for coverage)
    active_catalog_size = sum(
        1 for item in fs.items_by_id.values() if int(item.get("rating_count") or 0) >= 5
    )
    active_catalog_size = max(active_catalog_size, 1)

    # Build per-user ground truth on test split (items rated >= 3.5 or top test items if strict)
    # Also include val watched IDs in exclusion set when predicting test split
    val_watched_by_user: dict[int, set[int]] = {}
    for uid, gdf in val_df.groupby("user_id"):
        val_watched_by_user[int(uid)] = set(gdf["movie_id"].astype(int).tolist())

    # First 3 high-rated training movies per user for Cold-Start simulation
    cold_seeds_by_user: dict[int, list[int]] = {}
    for uid, gdf in train_df[train_df["rating"] >= 4.0].groupby("user_id"):
        cold_seeds_by_user[int(uid)] = (
            gdf.sort_values("timestamp")["movie_id"].astype(int).tolist()[:3]
        )

    test_ground_truth: dict[int, tuple[set[int], dict[int, float]]] = {}
    for uid, gdf in test_df.groupby("user_id"):
        pos_df = gdf[gdf["rating"] >= settings.positive_rating_threshold]
        if len(pos_df) == 0:
            pos_df = gdf[gdf["rating"] >= 3.0]
        if len(pos_df) == 0:
            continue
        rel_set = set(pos_df["movie_id"].astype(int).tolist())
        graded = {
            int(r["movie_id"]): float(max(1.0, r["rating"] - 2.5))
            for r in pos_df.to_dict(orient="records")
        }
        test_ground_truth[int(uid)] = (rel_set, graded)

    eval_users = sorted(test_ground_truth.keys())
    logger.info("Evaluating 5 architectures across %d held-out test users...", len(eval_users))

    model_names = [
        "Popularity Baseline",
        "Collaborative-Only (ALS)",
        "Semantic-Only (FAISS)",
        "Hybrid Two-Stage (RRF + LightGBM)",
        "Hybrid + MMR (Full KinoRank)",
    ]

    recs_by_model: dict[str, list[list[int]]] = {m: [] for m in model_names}
    cold_hits_by_model: dict[str, list[float]] = {m: [] for m in model_names}
    ndcg_by_model: dict[str, list[float]] = {m: [] for m in model_names}
    recall_by_model: dict[str, list[float]] = {m: [] for m in model_names}
    map_by_model: dict[str, list[float]] = {m: [] for m in model_names}
    ild_by_model: dict[str, list[float]] = {m: [] for m in model_names}
    latencies_by_model: dict[str, list[float]] = {m: [] for m in model_names}

    global_pop_list = fs.popular_movie_ids

    for uid in eval_users:
        rel_set, graded_rel = test_ground_truth[uid]
        u_prof = fs.get_user_profile(user_id=uid)
        exclude_set = set(u_prof.watched_movie_ids) | val_watched_by_user.get(uid, set())
        cold_seeds = cold_seeds_by_user.get(uid, u_prof.watched_movie_ids[:3])

        # ---------------------------------------------------------------------
        # 1. Popularity Baseline
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        pop_recs = [mid for mid in global_pop_list if mid not in exclude_set][:20]
        lat_pop = (time.perf_counter() - t0) * 1000.0
        latencies_by_model["Popularity Baseline"].append(lat_pop)
        pop_cold_recs = [mid for mid in global_pop_list if mid not in set(cold_seeds)][:10]

        # ---------------------------------------------------------------------
        # 2. Collaborative-Only (ALS)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        als_hits = als.retrieve(user_id=uid, exclude_movie_ids=exclude_set, top_k=20)
        als_recs = [mid for mid, _ in als_hits]
        lat_als = (time.perf_counter() - t0) * 1000.0
        latencies_by_model["Collaborative-Only (ALS)"].append(lat_als)

        # Cold-start ALS (fold-in from 3 seeds)
        als_cold_hits = als.retrieve(
            user_id=None,
            seed_movie_ids=cold_seeds,
            exclude_movie_ids=set(cold_seeds),
            top_k=10,
        )
        als_cold_recs = [mid for mid, _ in als_cold_hits]

        # ---------------------------------------------------------------------
        # 3. Semantic-Only (FAISS)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        sem_hits = sem.retrieve(
            seed_or_favorite_movie_ids=u_prof.watched_movie_ids[-12:],
            exclude_movie_ids=exclude_set,
            top_k=20,
        )
        sem_recs = [mid for mid, _ in sem_hits]
        lat_sem = (time.perf_counter() - t0) * 1000.0
        latencies_by_model["Semantic-Only (FAISS)"].append(lat_sem)

        sem_cold_hits = sem.retrieve(
            seed_or_favorite_movie_ids=cold_seeds,
            exclude_movie_ids=set(cold_seeds),
            top_k=10,
        )
        sem_cold_recs = [mid for mid, _ in sem_cold_hits]

        # ---------------------------------------------------------------------
        # 4. Hybrid Two-Stage (RRF + LightGBM LambdaRank)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        cands, _, _ = hybrid.retrieve_candidates(
            user_id=uid,
            watched_movie_ids=list(exclude_set),
            top_k=100,
        )
        ranked_lgbm = ranker.rank(
            user_profile=u_prof,
            candidates=cands,
            feature_store=fs,
            exclude_movie_ids=exclude_set,
            compute_shap=False,
        )
        hyb_recs = [r.movie_id for r in ranked_lgbm[:20]]
        lat_hyb = (time.perf_counter() - t0) * 1000.0
        latencies_by_model["Hybrid Two-Stage (RRF + LightGBM)"].append(lat_hyb)

        # Cold-start Hybrid (3 seeds)
        cold_prof = fs.synthesize_cold_start_profile(cold_seeds)
        cold_cands, _, _ = hybrid.retrieve_candidates(
            user_id=None,
            watched_movie_ids=cold_seeds,
            favorite_movie_ids=cold_seeds,
            top_k=60,
        )
        cold_ranked = ranker.rank(
            user_profile=cold_prof,
            candidates=cold_cands,
            feature_store=fs,
            exclude_movie_ids=set(cold_seeds),
            compute_shap=False,
        )
        hyb_cold_recs = [r.movie_id for r in cold_ranked[:10]]

        # ---------------------------------------------------------------------
        # 5. Hybrid + MMR (Full KinoRank)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        mmr_items = apply_mmr_diversity(
            ranked_items=ranked_lgbm[:40],
            semantic_retriever=sem,
            mmr_lambda=settings.mmr_lambda,
            top_k=20,
        )
        mmr_recs = [r.movie_id for r in mmr_items]
        lat_mmr = lat_hyb + (time.perf_counter() - t0) * 1000.0
        latencies_by_model["Hybrid + MMR (Full KinoRank)"].append(lat_mmr)

        cold_mmr_items = apply_mmr_diversity(
            ranked_items=cold_ranked[:30],
            semantic_retriever=sem,
            mmr_lambda=settings.mmr_lambda,
            top_k=10,
        )
        mmr_cold_recs = [r.movie_id for r in cold_mmr_items]

        # Record metrics for each model
        per_model_outputs = [
            ("Popularity Baseline", pop_recs, pop_cold_recs),
            ("Collaborative-Only (ALS)", als_recs, als_cold_recs),
            ("Semantic-Only (FAISS)", sem_recs, sem_cold_recs),
            ("Hybrid Two-Stage (RRF + LightGBM)", hyb_recs, hyb_cold_recs),
            ("Hybrid + MMR (Full KinoRank)", mmr_recs, mmr_cold_recs),
        ]

        for m_name, rec_ids, cold_ids in per_model_outputs:
            recs_by_model[m_name].append(rec_ids)
            ndcg_by_model[m_name].append(ndcg_at_k(rec_ids, rel_set, graded_rel, k=10))
            recall_by_model[m_name].append(recall_at_k(rec_ids, rel_set, k=20))
            map_by_model[m_name].append(average_precision_at_k(rec_ids, rel_set, k=10))
            cold_hits_by_model[m_name].append(hit_rate_at_k(cold_ids, rel_set, k=10))
            if sem.item_embeddings is not None:
                ild_by_model[m_name].append(
                    intra_list_diversity(rec_ids, sem.item_embeddings, sem.movie_id_to_idx, k=10)
                )

    summary_rows: list[dict[str, Any]] = []
    for m_name in model_names:
        lats = np.array(latencies_by_model[m_name], dtype=np.float64)
        row = {
            "model": m_name,
            "NDCG@10": round(float(np.mean(ndcg_by_model[m_name])), 4),
            "Recall@20": round(float(np.mean(recall_by_model[m_name])), 4),
            "MAP@10": round(float(np.mean(map_by_model[m_name])), 4),
            "Catalog_Coverage_%": round(
                catalog_coverage_pct(recs_by_model[m_name], active_catalog_size, k=10), 2
            ),
            "ColdStart_HitRate@10": round(float(np.mean(cold_hits_by_model[m_name])), 4),
            "ILD@10": round(float(np.mean(ild_by_model[m_name])), 4),
            "Latency_p50_ms": round(float(np.percentile(lats, 50)), 2),
            "Latency_p95_ms": round(float(np.percentile(lats, 95)), 2),
        }
        summary_rows.append(row)

    # Save JSON, Markdown, and Chart artifacts
    json_path = settings.metrics_dir / "benchmark_results.json"
    md_path = settings.metrics_dir / "benchmark_results.md"
    png_path = settings.metrics_dir / "benchmark_comparison.png"

    payload = {
        "dataset": settings.movielens_variant,
        "evaluated_users": len(eval_users),
        "active_catalog_denominator": active_catalog_size,
        "split_method": "strict_per_user_chronological_70_15_15",
        "results": summary_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Build Markdown table
    md_lines = [
        "### Resultados",
        "",
        "| Modelo | NDCG@10 | Recall@20 | MAP@10 | Cobertura (%) | Cold-Start HR@10 | Diversidade (ILD@10) | p50 (ms) | p95 (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in summary_rows:
        md_lines.append(
            f"| **{r['model']}** | `{r['NDCG@10']:.4f}` | `{r['Recall@20']:.4f}` | "
            f"`{r['MAP@10']:.4f}` | `{r['Catalog_Coverage_%']:.2f}%` | "
            f"`{r['ColdStart_HitRate@10']:.4f}` | `{r['ILD@10']:.4f}` | "
            f"`{r['Latency_p50_ms']:.2f}` | `{r['Latency_p95_ms']:.2f}` |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    _plot_benchmark_results(summary_rows, png_path)
    logger.info("Benchmark complete. Saved JSON, Markdown, and Plot to %s", settings.metrics_dir)
    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = run_offline_benchmark()
    print((settings.metrics_dir / "benchmark_results.md").read_text(encoding="utf-8"))

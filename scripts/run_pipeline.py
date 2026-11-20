"""End-to-end KinoRank MLOps Training & Artifact Generation Pipeline."""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from typing import Any

import pandas as pd

from src.config import settings
from src.data.features_duckdb import run_full_data_pipeline
from src.ranking.feature_store import FeatureStore
from src.ranking.lgbm_ranker import LightGBMRanker
from src.retrieval.collaborative import CollaborativeALSRetriever
from src.retrieval.hybrid import HybridCandidateRetriever
from src.retrieval.semantic import SemanticFAISSRetriever

logger = logging.getLogger(__name__)


def train_and_save_models(use_neural_encoder: bool = True) -> dict[str, Any]:
    """Train ALS Collaborative, FAISS Semantic, and LightGBM LambdaRank models and save artifacts."""
    settings.ensure_dirs()
    t_start = time.perf_counter()

    # Ensure Medallion data exists
    if not (settings.gold_dir / "item_features.parquet").exists():
        run_full_data_pipeline()

    train_df = pd.read_parquet(settings.silver_dir / "interactions_train.parquet")
    val_df = pd.read_parquet(settings.silver_dir / "interactions_val.parquet")
    items_df = pd.read_parquet(settings.gold_dir / "item_features.parquet")

    # 1. Train Collaborative ALS
    t0 = time.perf_counter()
    als_retriever = CollaborativeALSRetriever()
    als_retriever.fit(train_df=train_df, all_movie_ids=items_df["movie_id"].to_numpy())
    als_path = als_retriever.save()
    als_sec = round(time.perf_counter() - t0, 2)

    # 2. Train Semantic Encoder & FAISS Index
    t0 = time.perf_counter()
    sem_retriever = SemanticFAISSRetriever()
    sem_retriever.fit(catalog_df=items_df, use_neural_encoder=use_neural_encoder)
    faiss_path, emb_path = sem_retriever.save()
    sem_sec = round(time.perf_counter() - t0, 2)

    # 3. Initialize Feature Store & Hybrid Retriever
    feature_store = FeatureStore(try_redis=True)
    hybrid_retriever = HybridCandidateRetriever(
        als_retriever=als_retriever,
        semantic_retriever=sem_retriever,
        popular_fallback_ids=feature_store.popular_movie_ids,
    )

    # 4. Train LightGBM LambdaRank Ranker
    t0 = time.perf_counter()
    ranker = LightGBMRanker()
    ranker.fit(
        val_df=val_df,
        hybrid_retriever=hybrid_retriever,
        feature_store=feature_store,
    )
    lgbm_path = ranker.save()
    lgbm_sec = round(time.perf_counter() - t0, 2)

    # 5. Copy Gold feature summary artifact for Hugging Face packaging
    feature_stats_path = settings.artifacts_dir / "feature_stats.parquet"
    shutil.copy2(settings.gold_dir / "item_features.parquet", feature_stats_path)

    total_sec = round(time.perf_counter() - t_start, 2)
    report = {
        "als_artifact": str(als_path),
        "faiss_artifact": str(faiss_path),
        "embeddings_artifact": str(emb_path),
        "lgbm_artifact": str(lgbm_path),
        "feature_stats_artifact": str(feature_stats_path),
        "timing_seconds": {
            "als_train_s": als_sec,
            "semantic_faiss_s": sem_sec,
            "lgbm_train_s": lgbm_sec,
            "total_s": total_sec,
        },
    }
    logger.info("Model training pipeline complete: %s", report)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="KinoRank End-to-End Pipeline Orchestrator")
    parser.add_argument(
        "--stage",
        choices=["data", "train", "all"],
        default="all",
        help="Pipeline stage to execute",
    )
    parser.add_argument(
        "--dataset",
        default=settings.movielens_variant,
        help="MovieLens dataset variant",
    )
    args = parser.parse_args()

    if args.stage in ("data", "all"):
        data_summary = run_full_data_pipeline(dataset=args.dataset)
        print(f"Data Pipeline Summary: {data_summary}")

    if args.stage in ("train", "all"):
        train_summary = train_and_save_models()
        print(f"Training Pipeline Summary: {train_summary}")


if __name__ == "__main__":
    main()

"""Root entrypoint for Hugging Face Spaces (auto-bootstraps data & artifacts if missing)."""

from __future__ import annotations

import logging

from scripts.run_pipeline import train_and_save_models
from src.config import settings
from src.data.features_duckdb import run_full_data_pipeline
from ui.gradio_app import create_gradio_interface

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def ensure_ready() -> None:
    """Ensure Medallion data and trained artifacts exist when booting on Hugging Face Spaces."""
    if not (settings.gold_dir / "item_features.parquet").exists():
        run_full_data_pipeline()
    if not (settings.artifacts_dir / "lgbm_ranker.txt").exists():
        train_and_save_models()


ensure_ready()
demo = create_gradio_interface()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

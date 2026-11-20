"""Automated Publisher for Hugging Face Model Hub and Hugging Face Spaces (huggingface.co/Lealwbs)."""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from src.config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)

REQUIRED_MODEL_ARTIFACTS: list[str] = [
    "als_model.npz",
    "faiss_semantic.index",
    "item_embeddings.npy",
    "item_embeddings.ids.npy",
    "lgbm_ranker.txt",
    "feature_stats.parquet",
    "metrics/benchmark_results.json",
    "metrics/benchmark_results.md",
    "metrics/benchmark_comparison.png",
]

SPACE_REQUIREMENTS_TXT = """duckdb>=1.0.0
pyarrow>=15.0.0
pandas>=2.1.0
numpy>=1.26.0,<2.1.0
scipy>=1.11.0
implicit>=0.7.2
faiss-cpu>=1.8.0
lightgbm>=4.3.0
scikit-learn>=1.4.0
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.2.0
sentence-transformers>=3.0.0
fastapi>=0.111.0
pydantic>=2.7.0
pydantic-settings>=2.2.0
httpx>=0.27.0
gradio>=4.44.0
huggingface-hub>=0.24.0
matplotlib>=3.8.0
seaborn>=0.13.0
"""


def verify_local_artifacts() -> dict[str, Any]:
    """Check that all trained model artifacts and benchmark metrics exist locally."""
    missing: list[str] = []
    present: dict[str, int] = {}

    for rel_path in REQUIRED_MODEL_ARTIFACTS:
        full_p = settings.artifacts_dir / rel_path
        if not full_p.exists():
            missing.append(rel_path)
        else:
            present[rel_path] = full_p.stat().st_size

    model_card = PROJECT_ROOT / "MODEL_CARD.md"
    if not model_card.exists():
        missing.append("MODEL_CARD.md")
    else:
        present["MODEL_CARD.md"] = model_card.stat().st_size

    if missing:
        raise FileNotFoundError(
            f"Missing artifacts required for Hugging Face upload: {missing}. "
            "Run `uv run python -m scripts.run_pipeline --stage all` and `uv run python -m src.evaluation.benchmark` first."
        )

    return {
        "verified_files_count": len(present),
        "total_bytes": sum(present.values()),
        "files": present,
    }


def publish_to_huggingface(
    dry_run: bool = False,
    publish_space: bool = True,
    token: str | None = None,
) -> dict[str, Any]:
    """Upload trained artifacts + MODEL_CARD.md to HF Model Repo and optionally deploy HF Space."""
    verification = verify_local_artifacts()
    hf_token = (token or settings.hf_token).strip()

    if dry_run or not hf_token:
        mode_reason = "explicit --dry-run flag" if dry_run else "HF_TOKEN not set in environment"
        logger.info(
            "Hugging Face dry-run package verified (%s): %d files (%d bytes) ready for '%s' and Space '%s'.",
            mode_reason,
            verification["verified_files_count"],
            verification["total_bytes"],
            settings.hf_model_repo,
            settings.hf_space_repo,
        )
        return {
            "status": "dry_run_verified",
            "reason": mode_reason,
            "target_model_repo": settings.hf_model_repo,
            "target_space_repo": settings.hf_space_repo,
            "verification": verification,
        }

    api = HfApi(token=hf_token)

    # 1. Publish Model Repository
    logger.info("Creating/updating Hugging Face Model Repo: %s", settings.hf_model_repo)
    api.create_repo(repo_id=settings.hf_model_repo, repo_type="model", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        stage_dir = Path(tmp_dir)
        shutil.copy2(PROJECT_ROOT / "MODEL_CARD.md", stage_dir / "README.md")
        for rel_path in REQUIRED_MODEL_ARTIFACTS:
            src_file = settings.artifacts_dir / rel_path
            dst_file = stage_dir / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)

        api.upload_folder(
            folder_path=str(stage_dir),
            repo_id=settings.hf_model_repo,
            repo_type="model",
            commit_message="Upload KinoRank hybrid RecSys artifacts and benchmark results",
        )

    # 1b. Publish Gold Feature Tables Dataset Repository
    dataset_repo = f"{settings.hf_username}/kinorank-movielens-gold"
    logger.info("Creating/updating Hugging Face Dataset Repo: %s", dataset_repo)
    api.create_repo(repo_id=dataset_repo, repo_type="dataset", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_ds:
        ds_dir = Path(tmp_ds)
        shutil.copy2(PROJECT_ROOT / "MODEL_CARD.md", ds_dir / "README.md")
        for pq_name in (
            "item_features.parquet",
            "user_features.parquet",
            "cooccurrence_pairs.parquet",
        ):
            src_pq = settings.gold_dir / pq_name
            if src_pq.exists():
                shutil.copy2(src_pq, ds_dir / pq_name)
        api.upload_folder(
            folder_path=str(ds_dir),
            repo_id=dataset_repo,
            repo_type="dataset",
            commit_message="Upload KinoRank Gold feature tables",
        )

    # 2. Publish Hugging Face Space (Interactive Static Space + full Python/Gradio source)
    space_sdk_used = "static"
    if publish_space:
        logger.info("Creating/updating Hugging Face Space: %s", settings.hf_space_repo)
        api.create_repo(
            repo_id=settings.hf_space_repo,
            repo_type="space",
            space_sdk="static",
            exist_ok=True,
        )

        with tempfile.TemporaryDirectory() as tmp_space:
            sp_dir = Path(tmp_space)
            from ui.static_space import generate_static_space_bundle

            generate_static_space_bundle(sp_dir)
            space_readme = (
                "---\n"
                "title: KinoRank\n"
                "emoji: 🎬\n"
                "colorFrom: yellow\n"
                "colorTo: gray\n"
                "sdk: static\n"
                "pinned: true\n"
                "license: mit\n"
                "---\n\n" + (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
            )

            (sp_dir / "README.md").write_text(space_readme, encoding="utf-8")
            (sp_dir / "requirements.txt").write_text(SPACE_REQUIREMENTS_TXT, encoding="utf-8")
            shutil.copy2(PROJECT_ROOT / "app.py", sp_dir / "app.py")
            shutil.copytree(PROJECT_ROOT / "src", sp_dir / "src")
            shutil.copytree(PROJECT_ROOT / "ui", sp_dir / "ui")
            shutil.copytree(PROJECT_ROOT / "scripts", sp_dir / "scripts")

            api.upload_folder(
                folder_path=str(sp_dir),
                repo_id=settings.hf_space_repo,
                repo_type="space",
                commit_message="Update KinoRank interactive Space and documentation",
            )

    return {
        "status": "published",
        "space_sdk": space_sdk_used,
        "model_url": f"https://huggingface.co/{settings.hf_model_repo}",
        "space_url": f"https://huggingface.co/spaces/{settings.hf_space_repo}",
        "verification": verification,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Publish KinoRank artifacts & Space to Hugging Face"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Verify artifacts without network upload"
    )
    parser.add_argument("--no-space", action="store_true", help="Skip Hugging Face Space upload")
    args = parser.parse_args()
    res = publish_to_huggingface(dry_run=args.dry_run, publish_space=not args.no_space)
    print(res)

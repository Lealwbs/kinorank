"""Automated Kaggle Notebook / Kernel publisher using KAGGLE_USERNAME and KAGGLE_KEY."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


def publish_to_kaggle(dry_run: bool = False) -> dict[str, Any]:
    """Verify executed notebook and push to Kaggle Kernels if credentials are configured."""
    nb_dir = PROJECT_ROOT / "notebooks"
    nb_path = nb_dir / "kaggle_recsys_benchmark.ipynb"
    meta_path = nb_dir / "kernel-metadata.json"

    if not nb_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Missing kaggle_recsys_benchmark.ipynb or kernel-metadata.json")

    nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb_data.get("cells", []) if c.get("cell_type") == "code"]
    executed_cells = [c for c in code_cells if c.get("execution_count") is not None]

    kaggle_user = os.environ.get("KAGGLE_USERNAME", "").strip()
    kaggle_key = os.environ.get("KAGGLE_KEY", "").strip()
    kaggle_token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    kaggle_access_token = Path.home() / ".kaggle" / "access_token"

    has_creds = bool(
        (kaggle_user and kaggle_key)
        or kaggle_token
        or kaggle_json.exists()
        or kaggle_access_token.exists()
    )

    if dry_run or not has_creds:
        reason = (
            "explicit --dry-run"
            if dry_run
            else "KAGGLE_API_TOKEN / access_token not configured yet"
        )
        return {
            "status": "dry_run_verified",
            "reason": reason,
            "notebook_path": str(nb_path),
            "code_cells": len(code_cells),
            "executed_cells": len(executed_cells),
        }

    if kaggle_user:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        slug = meta["id"].split("/")[-1]
        meta["id"] = f"{kaggle_user}/{slug}"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    res = subprocess.run(
        [
            "uv",
            "run",
            "--with",
            "kaggle",
            "--with",
            "six",
            "kaggle",
            "kernels",
            "push",
            "-p",
            str(nb_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "status": "published",
        "stdout": res.stdout.strip(),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Publish KinoRank Notebook to Kaggle")
    parser.add_argument("--dry-run", action="store_true", help="Verify notebook without pushing")
    args = parser.parse_args()
    print(publish_to_kaggle(dry_run=args.dry_run))

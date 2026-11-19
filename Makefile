.PHONY: setup data train benchmark test lint format api ui notebook hf-dry-run all

setup:
	uv sync

data:
	uv run python -m src.data.features_duckdb

train:
	uv run python -m scripts.run_pipeline --stage train

benchmark:
	uv run python -m src.evaluation.benchmark

pipeline:
	uv run python -m scripts.run_pipeline --stage all
	uv run python -m src.evaluation.benchmark

test:
	uv run pytest -v

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

api:
	uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	uv run python app.py

hf-publish:
	uv run python -m scripts.publish_hf

hf-dry-run:
	uv run python -m scripts.publish_hf --dry-run

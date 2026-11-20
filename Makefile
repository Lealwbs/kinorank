.PHONY: help setup data train benchmark pipeline test lint format precommit clean api ui docker-build docker-up docker-down hf-publish hf-dry-run

help: ## Exibe a lista de comandos disponiveis
	@echo "Comandos disponiveis:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Instala as dependencias do projeto com uv
	uv sync

data: ## Executa a ingestao MovieLens e gera as camadas Bronze, Silver e Gold
	uv run python -m src.data.features_duckdb

train: ## Treina os modelos ALS, FAISS e LightGBM LambdaRank
	uv run python -m scripts.run_pipeline --stage train

benchmark: ## Roda a avaliacao comparativa nos 604 usuarios de teste
	uv run python -m src.evaluation.benchmark

pipeline: ## Executa dados, treino e benchmark completo de ponta a ponta
	uv run python -m scripts.run_pipeline --stage all
	uv run python -m src.evaluation.benchmark

test: ## Executa os 22 testes automatizados com pytest
	uv run pytest -v

lint: ## Checa a qualidade e formatacao do codigo com Ruff
	uv run ruff check .
	uv run ruff format --check .

format: ## Aplica correcoes de lint e formatacao automatica
	uv run ruff check --fix .
	uv run ruff format .

precommit: ## Instala os hooks do pre-commit no repositorio local
	uv run pre-commit install

clean: ## Remove caches temporarios do Python, Pytest e Ruff
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .coverage

api: ## Sobe a API FastAPI localmente na porta 8000
	uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

ui: ## Sobe a interface web localmente na porta 7860
	uv run python app.py

docker-build: ## Constroi as imagens Docker
	docker compose build

docker-up: ## Sobe API, Redis e interface web em segundo plano
	docker compose up -d

docker-down: ## Para e remove os containers Docker
	docker compose down --remove-orphans

hf-publish: ## Publica modelo, dataset e Space no Hugging Face
	uv run python -m scripts.publish_hf

hf-dry-run: ## Valida o pacote do Hugging Face localmente sem enviar
	uv run python -m scripts.publish_hf --dry-run

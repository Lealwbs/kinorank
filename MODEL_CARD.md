---
license: mit
library_name: lightgbm
pipeline_tag: text-ranking
tags:
  - recommender-systems
  - hybrid-retrieval
  - learning-to-rank
  - lightgbm
  - lambdarank
  - faiss
  - collaborative-filtering
  - sentence-transformers
  - agentic-ai
  - letterboxd
  - movielens
datasets:
  - grouplens/movielens-latest-small
metrics:
  - ndcg
  - recall
  - map
---

# KinoRank

Sistema de recomendação híbrido em três estágios (`Lealwbs/kinorank-hybrid-recsys`) combinando fatoração de matrizes (ALS), busca vetorial semântica (`BAAI/bge-small-en-v1.5` + FAISS), ranqueamento supervisionado (`LightGBM LambdaRank`), diversidade com Maximal Marginal Relevance (MMR) e camada conversacional com trava de catálogo.

## Arquitetura

1. **Estágio 1 (Busca Híbrida < 15 ms):**
   - **Via Colaborativa:** Alternating Least Squares (`factors=64`) com projeção em tempo real para Cold-Start e arquivos do Letterboxd.
   - **Via Semântica:** vetores densos de 384 dimensões (`BAAI/bge-small-en-v1.5`) indexados no FAISS (`IndexFlatIP`).
   - **Fusão RRF:** combina os dois rankings para recuperar 100 candidatos inéditos por usuário.
2. **Estágio 2 (Ranqueamento e Diversidade):**
   - **LightGBM (`objective="lambdarank"`):** ordena os candidatos usando 15 variáveis de usuário, item e cruzamento, extraindo contribuições **TreeSHAP** por filme.
   - **MMR ($\lambda = 0{,}75$):** equilibra relevância e diversidade semântica/gênero na lista final.
3. **Estágio 3 (Agente e Governança):**
   - Tools Pydantic (`parse_user_intent`, `retrieve_and_rank`, `rerank_and_explain`) com bloqueio de IDs externos ao catálogo e fallback determinístico.

## Resultados

Avaliação nos 604 usuários do conjunto de teste temporal (treino nos primeiros 70% cronológicos de cada usuário, validação em 15% e teste nos 15% finais):

| Modelo | NDCG@10 | Recall@20 | MAP@10 | Cobertura (%) | Cold-Start HR@10 | Diversidade (ILD@10) | p50 (ms) | p95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | `0.0515` | `0.0620` | `0.0280` | `2.50%` | `0.1738` | `0.3599` | `0.01` | `0.01` |
| **Collaborative-Only (ALS)** | `0.0574` | `0.1181` | `0.0277` | `31.48%` | `0.1474` | `0.3328` | `0.16` | `0.47` |
| **Semantic-Only (FAISS)** | `0.0082` | `0.0173` | `0.0038` | `32.17%` | `0.0712` | `0.1967` | `0.72` | `0.87` |
| **Hybrid Two-Stage (RRF + LightGBM)** | `0.0642` | `0.1002` | `0.0323` | `28.94%` | `0.1556` | `0.3379` | `5.93` | `7.76` |
| **Hybrid + MMR (Full KinoRank)** | **`0.0648`** | `0.1012` | **`0.0325`** | `28.84%` | `0.1573` | **`0.3421`** | `11.26` | `14.21` |

## Artefatos

- `als_model.npz`: fatores latentes de usuário e item (64-D) e mapeamentos de IDs.
- `faiss_semantic.index`: índice FAISS Inner-Product (384-D).
- `item_embeddings.npy` e `item_embeddings.ids.npy`: matriz de embeddings normalizados por filme.
- `lgbm_ranker.txt`: modelo LightGBM LambdaRank treinado.
- `feature_stats.parquet`: tabela Gold com estatísticas bayesianas e metadados enriquecidos.
- `metrics/benchmark_results.json` e `metrics/benchmark_comparison.png`: métricas e gráfico do benchmark.

## Uso

```python
from src.agent.tools import KinoRankAgentOrchestrator

orchestrator = KinoRankAgentOrchestrator()
response = orchestrator.recommend(
    user_id=1,
    query="thriller psicologico estilo Fincher, < 135 min, depois de 1994, sem terror",
    top_k=10,
)
for rec in response.recommendations:
    print(f"#{rec.rank} {rec.title} ({rec.release_year}) -> {rec.explanation}")
```

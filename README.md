# KinoRank

[![CI](https://github.com/Lealwbs/kinorank/actions/workflows/ci.yml/badge.svg)](https://github.com/Lealwbs/kinorank/actions)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-SQL-FFF000?logo=duckdb&logoColor=black)](https://duckdb.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-LambdaRank-10B981)](https://lightgbm.readthedocs.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Lealwbs%2Fkinorank-F59E0B)](https://huggingface.co/spaces/Lealwbs/kinorank)

Sistema de recomendação de filmes em três estágios, combinando filtragem colaborativa (ALS), busca vetorial semântica (FAISS), ranqueamento supervisionado (LightGBM) e camada conversacional com LLM. Inclui suporte direto a arquivos exportados do Letterboxd (`.csv` e `.zip`).

## Problema

Recomendar itens em catálogos grandes exige equilibrar três pontos que costumam conflitar:
1. **Filtro-bolha da popularidade:** recomendar apenas os itens mais famosos concentra cliques em menos de 3% do catálogo.
2. **Latência vs. precisão:** rodar um modelo pesado ou um LLM sobre milhares de itens inviabiliza o tempo de resposta em produção.
3. **Alucinação em buscas conversacionais:** pedir recomendações diretamente a um LLM gera filmes fora do catálogo ou títulos que o usuário já assistiu.

Este projeto resolve isso dividindo a decisão em um funil de três estágios: recupera 100 candidatos em menos de 15 ms, ordena os melhores com LightGBM e diversidade (MMR), e usa o LLM apenas nos 15 itens finais com validação rígida de IDs.

## Resultados

| 9.742 | 100.836 | 610 | 11,26 ms |
|:---:|:---:|:---:|:---:|
| filmes no catálogo | avaliações (MovieLens) | usuários avaliados | latência mediana (p50) |

| +25,8% | 28,84% vs 2,50% | 0% |
|:---:|:---:|:---:|
| ganho de NDCG@10 sobre popularidade | cobertura ativa do catálogo (11,5x maior) | filmes alucinados ou já assistidos |

Comparativo offline nos **604 usuários do conjunto de teste temporal** (treino nos primeiros 70% cronológicos de cada usuário, validação em 15% e teste nos 15% finais):

| Modelo | NDCG@10 | Recall@20 | MAP@10 | Cobertura (%) | Cold-Start HR@10 | Diversidade (ILD@10) | p50 (ms) | p95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | `0.0515` | `0.0620` | `0.0280` | `2.50%` | `0.1738` | `0.3599` | `0.01` | `0.01` |
| **Collaborative-Only (ALS)** | `0.0574` | `0.1181` | `0.0277` | `31.48%` | `0.1474` | `0.3328` | `0.16` | `0.47` |
| **Semantic-Only (FAISS)** | `0.0082` | `0.0173` | `0.0038` | `32.17%` | `0.0712` | `0.1967` | `0.72` | `0.87` |
| **Hybrid Two-Stage (RRF + LightGBM)** | `0.0642` | `0.1002` | `0.0323` | `28.94%` | `0.1556` | `0.3379` | `5.93` | `7.76` |
| **Hybrid + MMR (Full KinoRank)** | **`0.0648`** | `0.1012` | **`0.0325`** | `28.84%` | `0.1573` | **`0.3421`** | `11.26` | `14.21` |

![Comparativo de Modelos](artifacts/metrics/benchmark_comparison.png)

- **Ganho do funil híbrido:** combinar ALS e FAISS via RRF e reordenar com LightGBM (`lambdarank`) supera o modelo colaborativo isolado em **+12,9% no NDCG@10** e **+17,3% no MAP@10**.
- **Diversidade sem perda de acerto:** o pós-processamento com MMR ($\lambda = 0{,}75$) reduz itens redundantes na mesma lista e eleva tanto a diversidade (`ILD@10 = 0.3421`) quanto a precisão no topo (`NDCG@10 = 0.0648`).
- **Baixa latência:** os estágios 1 e 2 combinados rodam em **11,26 ms (p50)** e **14,21 ms (p95)** em CPU.

## Pipeline

```text
MovieLens + TMDB + Letterboxd -> Parquet (Bronze / Silver / Gold)
                              -> Estagio 1: ALS + FAISS -> Fusao RRF (Top 100)
                              -> Estagio 2: LightGBM LambdaRank + MMR (Top 15)
                              -> Estagio 3: Agente LLM + Trava de Catalogo (Top 10)
```

```mermaid
flowchart LR
    subgraph Dados["Camada de Dados"]
        RAW["MovieLens + TMDB\n+ Letterboxd CSV/ZIP"] --> BRONZE["Bronze (Parquet)\nTabelas brutas"]
        BRONZE --> SILVER["Silver (Parquet)\nCatálogo + Split Temporal\n(70% / 15% / 15%)"]
        SILVER --> GOLD["Gold (DuckDB / PySpark)\nWindow SQL + Time Decay\n+ Média Bayesiana"]
    end

    subgraph Estagio1["Estágio 1: Busca (< 15ms)"]
        GOLD --> ALS["Colaborativo\nImplicit ALS (64-D)"]
        GOLD --> FAISS["Semântico\nbge-small-en (384-D) + FAISS"]
        ALS --> RRF["Fusão RRF + Filtro de Assistidos\n(9.742 -> 100 candidatos)"]
        FAISS --> RRF
    end

    subgraph Estagio2["Estágio 2: Ranqueamento"]
        GOLD --> FS["Feature Store\nRedis ou DuckDB em memória"]
        RRF --> LGBM["LightGBM LambdaRank\n15 features + TreeSHAP"]
        FS --> LGBM
        LGBM --> MMR["Diversidade MMR\n(100 -> 15 candidatos)"]
    end

    subgraph Estagio3["Estágio 3: Agente"]
        QUERY["Busca em texto livre"] --> T1["Tool 1: parse_user_intent\nFiltros Pydantic"]
        T1 --> FAISS
        T1 --> LGBM
        MMR --> T3["Tool 3: rerank_and_explain\nGemini / OpenAI / Groq\n+ Fallback Determinístico"]
        T3 --> GUARD["Trava de Catálogo\nBloqueia IDs externos"]
    end
```

## Estágios

### Dados
Modelagem em três camadas Parquet (`bronze`, `silver`, `gold`) com DuckDB ([`src/data/features_duckdb.py`](src/data/features_duckdb.py)) e módulo equivalente em PySpark ([`src/data/features_pyspark.py`](src/data/features_pyspark.py)):
- **Split temporal por usuário:** ordena as interações de cada usuário por `timestamp` antes de separar treino, validação e teste. Isso garante que nenhuma avaliação futura vaze para o cálculo de features do passado.
- **Média bayesiana:** suaviza a nota de filmes com poucas avaliações usando a média global ($m = 15$ votos de peso prévio).
- **Decaimento no tempo:** aplica peso exponencial $\exp(-\lambda \Delta t)$ para dar mais importância ao gosto recente do usuário e à tendência atual de cada filme.

### Busca
Recupera rapidamente 100 candidatos a partir dos 9.742 filmes ([`src/retrieval/hybrid.py`](src/retrieval/hybrid.py)):
- **Via colaborativa ([`src/retrieval/collaborative.py`](src/retrieval/collaborative.py)):** fatoração de matrizes com ALS (`implicit`, 64 dimensões). Para novos usuários (Cold-Start ou importação do Letterboxd), projeta o vetor do usuário em tempo real pela média dos vetores dos filmes escolhidos.
- **Via semântica ([`src/retrieval/semantic.py`](src/retrieval/semantic.py)):** gera vetores densos de 384 dimensões (`BAAI/bge-small-en-v1.5`) a partir de título, ano, gêneros, diretor, tags e sinopse, indexados no FAISS (`IndexFlatIP`).
- **Reciprocal Rank Fusion (RRF):** combina as posições das duas listas sem depender de calibração de escala entre produto interno e similaridade cosseno:

$$\text{RRF}(i) = \frac{w_{\text{als}}}{60 + \text{pos}_{\text{als}}(i)} + \frac{w_{\text{faiss}}}{60 + \text{pos}_{\text{faiss}}(i)}$$

### Ranqueamento
Ordena os 100 candidatos com `LightGBM` (`objective="lambdarank"`, otimizando NDCG) sobre 15 variáveis de usuário, item e cruzamento ([`src/ranking/lgbm_ranker.py`](src/ranking/lgbm_ranker.py)):
- Sinais do Estágio 1 (`rrf_score`, `collab_score`, `semantic_score` e inverso das posições).
- Afinidade histórica do usuário com os gêneros, década e diretor do filme candidato.
- Coocorrência item-item com os últimos filmes favoritos do usuário.
- Nota bayesiana, popularidade recente e diferença para a nota média do usuário.
- Ao final, calcula a contribuição exata de cada variável via **TreeSHAP** (`pred_contrib=True`) e aplica **MMR** ([`src/ranking/diversity.py`](src/ranking/diversity.py)) para penalizar filmes muito parecidos entre si na mesma resposta.

### Agente
Camada conversacional estruturada com Pydantic ([`src/agent/tools.py`](src/agent/tools.py)):
1. `parse_user_intent`: extrai restrições do texto livre (ex.: *"thriller estilo Fincher, < 135 min, depois de 1994, sem terror"* vira `required_genres=["Thriller"]`, `excluded_genres=["Horror"]`, `max_runtime_minutes=135`, `min_year=1994`, `reference_directors=["David Fincher"]`).
2. `retrieve_and_rank`: aciona os Estágios 1 e 2 aplicando os filtros rígidos.
3. `rerank_and_explain`: gera a justificativa curta ancorada nos filmes reais do histórico do usuário.
- **Trava contra alucinação ([`src/agent/guardrails.py`](src/agent/guardrails.py)):** qualquer `movie_id` devolvido pelo LLM que não pertença à lista do Estágio 2 ou que já tenha sido assistido é descartado automaticamente. Se a API externa falhar ou estiver sem chave, um motor determinístico assume em menos de 1 ms.

## Execução

### Local

```bash
# 1. Instalar dependencias, processar dados, treinar modelos e rodar benchmark
make setup && make pipeline

# 2. Rodar os 22 testes automatizados
make test

# 3. Subir a interface web (http://localhost:7860) ou a API FastAPI (http://localhost:8000/docs)
make ui
# ou: make api
```

### Docker

```bash
cp .env.example .env
docker compose up --build
```

## Decisões

Como cada escolha técnica deste projeto se conecta a cenários reais de marketplaces de alta escala (como recomendação de restaurantes e pratos no iFood):

| Decisão | Por que foi usada aqui | Paralelo em Marketplace (ex.: iFood) |
| :--- | :--- | :--- |
| **Funil em 3 estágios (`9.742 -> 100 -> 15 -> 10`)** | Calcular 15 features de cruzamento, TreeSHAP e LLM para todo o catálogo derrubaria a latência. O RRF corta o espaço em `< 2 ms` e o LightGBM avalia só 100 itens em `< 5 ms`. | Em delivery, o Estágio 1 filtra lojas abertas no polígono de entrega via ANN/Two-Tower, o Estágio 2 ranqueia com GBDT usando tempo de entrega e conversão, e o Estágio 3 atua apenas no topo da vitrine. |
| **Busca Híbrida (ALS + FAISS com RRF)** | O modelo colaborativo acerta o gosto implícito mas falha em buscas específicas ou itens novos; a busca semântica entende texto livre mas tem baixo NDCG isolada (`0.0082` vs `0.0648` no híbrido). | Atende tanto o pedido recorrente por hábito (sinal colaborativo) quanto a busca por intenção (*"almoço saudável até 35 reais entregue em 30 min"*). |
| **Feature Store com fallback (Redis + Memória)** | Em produção com Docker o Redis serve features em baixa latência; em ambientes sem Redis externo (Hugging Face Spaces, CI, testes locais) o sistema lê direto da camada Gold em memória. | Separa o cálculo pesado offline (Spark/DuckDB) da leitura online rápida sem mudar o código do ranker. |
| **Diversidade com MMR antes do estágio final** | Sem controle de diversidade, o ranker tende a empilhar itens quase idênticos (ex.: várias sequências da mesma franquia). O MMR penaliza similaridade com os itens já escolhidos na lista. | Evita preencher a home do usuário com 10 combos idênticos da mesma categoria, melhorando a descoberta de novos restaurantes. |
| **Trava de catálogo + fallback determinístico** | O LLM nunca escolhe itens do zero: ele apenas reordena e explica IDs já validados pelo Estágio 2, e cai para um explicador determinístico se a API atrasar ou retornar erro `429`. | Garante que nenhum item indisponível ou fechado seja exibido ao cliente, mantendo o SLA mesmo sob falha do provedor de LLM. |

## Roteiro

Resumo direto caso precise apresentar a arquitetura em 3 minutos:

1. **Dados sem vazamento do futuro (`src/data/`):** usamos MovieLens enriquecido com TMDB. Em vez de sorteio aleatório, dividimos a linha do tempo de cada usuário (`70% passado / 15% meio / 15% futuro`). As features SQL no DuckDB (e espelho em PySpark) usam apenas os 70% de treino, aplicando média bayesiana e decaimento temporal.
2. **Recuperação híbrida rápida (`src/retrieval/`):** combinamos quem tem gosto parecido (**ALS**, 64 dimensões) com o conteúdo do filme (**FAISS** com `bge-small-en`, 384 dimensões) usando **Reciprocal Rank Fusion (RRF)** para selecionar 100 candidatos inéditos em menos de 15 ms.
3. **Aprendizado de ranqueamento (`src/ranking/`):** um modelo **LightGBM (`lambdarank`)** cruza 15 variáveis de afinidade entre o usuário e os 100 candidatos, extrai os pesos **TreeSHAP** para transparência do score e aplica **MMR** para diversificar a lista.
4. **Camada agêntica segura (`src/agent/`):** converte pedidos em texto livre para filtros tipados em Pydantic, gera explicações baseadas nos filmes que a pessoa realmente curtiu e bloqueia qualquer ID fora da lista validada.

## Estrutura

```text
kinorank/
├── .github/workflows/ci.yml                # Pipeline CI (Ruff + Pytest)
├── artifacts/metrics/                      # Tabelas JSON/MD e grafico do benchmark
├── notebooks/kaggle_recsys_benchmark.ipynb # Notebook executavel com EDA, SQL e benchmark
├── scripts/
│   ├── run_pipeline.py                     # Orquestrador de dados e treino
│   ├── publish_hf.py                       # Publicacao no Hugging Face (Model e Space)
│   └── publish_kaggle.py                   # Publicacao no Kaggle
├── src/
│   ├── config.py                           # Configuracoes centralizadas (.env)
│   ├── schemas/                            # Contratos Pydantic (Dominio, Agente, API)
│   ├── data/                               # Ingestao, TMDB, Split, DuckDB, PySpark, Letterboxd
│   ├── retrieval/                          # Estagio 1: ALS, FAISS e RRF
│   ├── ranking/                            # Estagio 2: Feature Store, LightGBM e MMR
│   ├── agent/                              # Estagio 3: Cliente LLM, Tools, Guardrails, Telemetria
│   ├── evaluation/                         # Metricas (NDCG, Recall, MAP, Cobertura, ILD) e Benchmark
│   └── api/main.py                         # API FastAPI
├── tests/                                  # 22 testes automatizados (dados reais, zero mocks)
├── ui/
│   ├── gradio_app.py                       # Interface Gradio
│   └── static_space.py                     # Gerador do Space estatico interativo
├── app.py                                  # Entrypoint raiz
├── Dockerfile                              # Build multi-stage
├── docker-compose.yml                      # FastAPI + Redis + UI
└── Makefile                                # Atalhos de execucao
```

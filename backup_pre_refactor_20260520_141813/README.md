# RAG-a-muffin

**Production-grade RAG pipeline** for the DataTalks.Club FAQ knowledge base (4 courses). Focuses on **clean data → rich metadata (topics + NER) → multi-vector ingestion → benchmarking + LLM-as-judge evaluation**.

## Pipeline Overview

See [pipeline_diagram.md](pipeline_diagram.md) for the full dataflow.

### High-level Steps

1. **p01_data_cleaning** — Raw → Clean structured data
2. **p02_eda** — Exploratory analysis, topic modeling, entity patterns, model comparison
3. **p03_generation** — Synthetic test queries (in-domain + OOD)
4. **p04_ingestion** — Vector stores (Qdrant + Elasticsearch), multi-model embeddings, benchmarking
5. **p05_evaluation** — Diverse test sets + LLM-as-judge scoring
6. **p06_answer_generation** — End-to-end RAG response generation

**Main orchestrator**: `production_pipeline/run_clean_pipeline.py`

## Detailed Steps & Outputs

### p01_data_cleaning
- **p00_load_llm_queries.py** — Loads any pre-generated LLM queries if needed.
- **p01_download.py** — Downloads `faq.zip` from DataTalksClub/faq repo and extracts MD files.
- **p02_parse.py** — Parses markdown into structured JSONL (`id`, `question`, `answer`, `course`, `section`). Cleans images, HTML, headers, Jinja macros.
- **p03_dedup.py** — Deduplicates at ~95% similarity threshold.
- **p04_stratified_test_split.py** — Creates `train.jsonl` + `test.jsonl` (stratified by course/section).

**Key outputs**:
- `production_pipeline/p01_data_cleaning/data/processed/clean.jsonl` (~1140 docs)
- `test.jsonl` (used downstream)

### p02_eda
- Topic modeling (BERTopic) across multiple embedding models.
- Topic validation + subtopics.
- NER/entity pattern learning + merging.
- Embedding model comparison + TF-IDF analysis.

**Key outputs**:
- Topic assignments per model.
- `eda_summary.json`
- Entity patterns for boosting.

### p03_generation
- Samples documents.
- Generates synthetic queries with LLMs.
- Adds out-of-distribution (OOD / "chaos monkey") queries.

**Outputs**: Augmented test query sets.

### p04_ingestion
- **p00_ingest_es.py** / **p00_ingest_qdrant.py** — Ingestion into ES (BM25 + dense) and Qdrant (per-model collections).
- **p02_ingest_models.py** — Multi-embedding support.
- Benchmarking with/without rerankers, payload filtering (topic/NER boosting).

**Outputs**:
- Qdrant collections (e.g. `faqs_bge_base_en_v1.5`)
- Elasticsearch `faqs_complete` index
- Benchmark results (`experiments/benchmark_results.json`)

### p05_evaluation
- Generates diverse test queries.
- **p05_llm_judge.py** — Uses LLM (e.g. via Grok or local) to score faithfulness + factual correctness.

**Outputs**: Judge scores, failure analysis.

### p06_answer_generation
End-to-end RAG inference with chosen retriever/reranker/config.

## Quick Start

```bash
# Install
uv sync

# Run full clean pipeline (recommended)
uv run python -m production_pipeline.run_clean_pipeline

# Or individual stages via just (see justfile)
just clean-pipeline
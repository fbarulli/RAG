# RAG-a-muffin Handoff — 2026-05-30d

## Baseline (end of session)
| Config | H@1 | H@5 | MRR |
|---|---|---|---|
| entity_boosted (faqs_bge_base_en_v1_5_qa) | 89.8% | 98.5% | 0.9377 |

Query type breakdown:
| Type | n | H@1 |
|---|---|---|
| chaos_monkey | 147 | 83.0% |
| creative_student | 120 | 90.8% |
| grounded_analyst | 147 | 93.2% |
| original | 49 | 98.0% |

DB verified metrics:
- `ndcg_1` = 0.8985 (equals H@1 by definition — correct)
- `ndcg_5` = 0.9488
- `map_score` = 0.9377
- `latency_p99` = 19.7ms

---

## What was done this session

### 1. MLflow infrastructure wired end-to-end
- `src/rag_pipeline/mlflow/` created as dedicated module
- `mlflow_logger.py` moved from `ingestion/` → `mlflow/logger.py`
- `mlflow/ablation_logger.py` created — logs per-config metrics + JSONL artifact per ablation run
- Both loggers now use SQLite backend (`sqlite:///{Paths.mlflow_db()}`) — fixes file store deprecation warning
- `benchmark.py` wired: calls `log_benchmark_run` after each `_run_config`
- `experiment.py` wired: calls `log_ablation_run` before `return result`
- MLflow experiment names: `"rag-retrieval"` (benchmark), `"rag-ablation"` (ablation)

### 2. SQLite results DB (`experiments/rag_results.db`)
- `src/rag_pipeline/db/` created with `models.py`, `engine.py`, `store.py`
- Schema: `corpus`, `runs`, `run_metrics`, `query_results`
- No ORM relationships — explicit SQL joins in `store.py` to avoid SQLModel forward-ref issues
- `save_experiment_result(result, ablation_results_dir)` wired into `experiment.py`
- DB config in `configs/db.json`, accessed via `Paths.results_db()` and `Paths.mlflow_db()`

### 3. `MetricsSnapshot` → `MetricSummary` consolidation
- `MetricsSnapshot` deleted from `core/models/ablation.py`
- `MetricSummary` (in `benchmark_types.py`) is now the single metrics class
- `MetricSummary.from_benchmark_row(row)` classmethod added — handles both short (`h1`) and full (`hit_rate_1`) field names
- `ExperimentResult.metrics` is now `dict[str, MetricSummary]`
- `RunMetrics` in `db/models.py` mirrors `MetricSummary` fields exactly
- All callers updated: `experiment.py`, `store.py`, `ablation_logger.py`, `cli.py`, `report.py`, `__init__.py`

### 4. New metrics added throughout
Added to `core.py`, `aggregation.py`, `MetricSummary`, `RunMetrics`:
- `ndcg_1`, `ndcg_5` — via existing `compute_ndcg_at_k`
- `map_score` — new `compute_map` function in `core.py`
- `latency_p99` — was computed but not propagated; now fully wired

### 5. Ablation report extended
`report.py` `_fmt_row` and header now show: `H@1, H@5, MRR, NDCG@10, p50ms, fail%` + query type columns

### 6. `load_all()` validates through Pydantic
`report.py` now calls `ExperimentResult.model_validate()` on each `_meta.json` — catches schema drift on load

### 7. Ingest always uses `--no-skip-existing` in ablation
Both `_run_payload_only` and `_run_with_rerun` in `experiment.py` now pass `--no-skip-existing` to `ingest_models`

### 8. `configs/db.json` added
```json
{
  "_comment": "Database configuration.",
  "results_db": "experiments/rag_results.db",
  "mlflow_db": "experiments/mlflow/mlflow.db"
}
```
`Paths` now has `results_db()`, `mlflow_db()`, `mlflow_dir()` reading from this file.

---

## Architecture & dataflow

### Metrics flow (single source of truth)
```
aggregation.py::aggregate_metrics()
  └─ returns MetricSummary
       ├─ benchmark.py → save_benchmark_results (JSON backup)
       ├─ benchmark.py → log_benchmark_run (MLflow: rag-retrieval)
       └─ experiment.py::_collect_results()
            └─ MetricSummary.from_benchmark_row(row)
                 └─ ExperimentResult.metrics[cfg]
                      ├─ ablation_logger.py → log_ablation_run (MLflow: rag-ablation)
                      └─ store.py → save_experiment_result → RunMetrics (SQLite)
```

### DB schema
```
corpus        (es_id PK, question, answer, course, section, ner_*, topic, subtopic)
runs          (run_id PK, experiment, patch, config, model, git_commit, timestamp, corpus_size)
run_metrics   (run_id FK, num_queries, hit_rate_1..10, mrr, ndcg_1/5/10, map_score,
               latency_p50/95/99, avg_code_integrity_*, cross_course_contamination,
               rank_std, failure_count, avg_failure_similarity)
query_results (id PK, run_id FK, query_id, query_text, expected_id FK→corpus,
               course, topic, subtopic, query_type, hit_ids, hit_scores,
               latency_ms, hit_at_1, hit_at_5, rank)
```

### Key paths
```python
Paths.results_db()     # experiments/rag_results.db
Paths.mlflow_db()      # experiments/mlflow/mlflow.db
Paths.mlflow_dir()     # experiments/mlflow/
```

### Module layout
```
src/rag_pipeline/
  db/
    __init__.py
    models.py       — SQLModel table definitions (no ORM relationships)
    engine.py       — get_engine(), init_db(), get_session()
    store.py        — ExperimentResult → DB (save_experiment_result)
  mlflow/
    __init__.py
    logger.py       — log_benchmark_run (rag-retrieval experiment)
    ablation_logger.py — log_ablation_run (rag-ablation experiment)
```

---

## Known issues & open items

### 1. 89.8% vs 92.2% baseline gap (immediate — carried from previous session)
Still unresolved. `_expand_urls` shifted 64 docs' `ner_primary_entity`. MLflow now tracking — run the planned experiments:

| Experiment | Description |
|---|---|
| `baseline_spacy` | no URL expansion — reproduce 92.2% |
| `url_expand` | with `_expand_urls` — measure impact |
| `no_rules_primary` | rules reclassify without setting primary entity |

```bash
uv run python -m rag_pipeline.ablation run --name baseline_spacy
uv run python -m rag_pipeline.ablation run --name url_expand
```

### 2. All old ablation results deleted — need full rerun
All JSONLs and `_meta.json` files were cleared for clean slate. Re-run all ablations:
```bash
uv run python -m rag_pipeline.ablation flow --configs entity_boosted --rerun
```

### 3. DB → report pipeline not complete
`_collect_results` in `experiment.py` still reads from `benchmark_results.json`, not the DB. The report therefore shows old metrics (`ndcg_1/5`, `map_score`, `latency_p99` are None in report). Next step: rewrite `_collect_results` to query `run_metrics` from SQLite directly.

### 4. `corpus` table not yet populated
`CorpusDoc` table exists but is empty. Populate from `clean.jsonl`:
```bash
uv run python -c "
from rag_pipeline.db.store import populate_corpus
populate_corpus()
"
```
`populate_corpus()` needs to be written in `store.py`.

### 5. Architecture auto-generation (planned)
Decision made: generate `ARCHITECTURE.json` by walking imports and class usages, for LLM context loading. Not yet implemented.

### 6. MLflow UI traces 500 errors (low priority)
UI `/traces/metrics` and `/datasets/search` endpoints 500 with SQLite backend on this MLflow version. Run data is accessible via `mlflow.search_runs()`. Not worth fixing now.

### 7. Ablation `report.py` reads old `_meta.json` format
`load_all()` calls `ExperimentResult.model_validate()` which coerces old dicts. After full rerun this is a non-issue.

### 8. Noisy `ner_primary_entity` from `extract_entity` (carried)
257 reclassified docs have category keywords as `ner_primary_entity`. Fix: don't set when `classification_source == 'rules'`.

### 9. Cross-encoder reranking (deferred)
Never benchmarked against QA collection. Infrastructure exists.

---

## How to re-run everything cleanly
```bash
# 1. Clean DB
rm -f experiments/rag_results.db

# 2. Run all ablations
uv run python -m rag_pipeline.ablation flow --configs entity_boosted --rerun

# 3. Check report
uv run python -m rag_pipeline.ablation report

# 4. Check DB
python -c "
import sqlite3
con = sqlite3.connect('experiments/rag_results.db')
cols = [d[1] for d in con.execute('PRAGMA table_info(run_metrics)').fetchall()]
for row in con.execute('SELECT * FROM run_metrics').fetchall():
    print(dict(zip(cols, row)))
"
```
## Ablation summary (pre-session baseline, QA collection)

| Experiment | Delta H@1 | Meaning |
|---|---|---|
| no_entity | -11.9% | entity signal is load-bearing |
| no_cluster | -3.5% | cluster fallback matters |
| empty_patterns | -3.5% | hand-crafted patterns matter |
| no_rules | -2.8% | rules matter |
| no_generic_entity | -3.2% | generic entities still useful |

Redundant signals (no delta): `ner_category`, `topic` assignments, low-confidence topic nulling.
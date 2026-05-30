# RAG-a-muffin Results Database

## Overview

`experiments/rag_results.db` is a SQLite database that stores all experiment runs, aggregate metrics, and per-query results. It is the single source of truth for analysis and reporting.

MLflow uses a separate `experiments/mlflow/mlflow.db` for its own run tracking UI.

Both paths are configured in `configs/db.json` and accessed exclusively via `Paths.results_db()` and `Paths.mlflow_db()`.

---

## Schema

### `corpus`
The FAQ corpus — questions, answers, and NER/topic metadata. Populated once from `data/processed/clean.jsonl` and updated when topic modeling reruns.

| Column | Type | Notes |
|---|---|---|
| `es_id` | TEXT PK | Stable document ID |
| `question` | TEXT | |
| `answer` | TEXT | |
| `course` | TEXT | |
| `section` | TEXT | |
| `ner_category` | TEXT | |
| `ner_primary_entity` | TEXT | |
| `topic` | INT | -1 = unassigned |
| `subtopic` | INT | |

### `runs`
One row per experiment+config execution.

| Column | Type | Notes |
|---|---|---|
| `run_id` | TEXT PK | `{experiment}__{config}` |
| `experiment` | TEXT | e.g. `baseline`, `no_entity` |
| `patch` | TEXT | `Patch.label()` output |
| `config` | TEXT | e.g. `entity_boosted` |
| `model` | TEXT | e.g. `BAAI/bge-base-en-v1.5` |
| `git_commit` | TEXT | |
| `timestamp` | DATETIME | UTC |
| `corpus_size` | INT | |

### `run_metrics`
Aggregate metrics for a run. 1:1 with `runs`.

| Column | Type | Notes |
|---|---|---|
| `run_id` | TEXT PK FK | |
| `h1` | FLOAT | Hit@1 |
| `h3` | FLOAT | Hit@3 |
| `h5` | FLOAT | Hit@5 |
| `h10` | FLOAT | Hit@10 |
| `mrr` | FLOAT | Mean Reciprocal Rank |
| `ndcg_10` | FLOAT | NDCG@10 |
| `latency_p50` | FLOAT | ms |
| `latency_p95` | FLOAT | ms |
| `rank_std` | FLOAT | |
| `cross_course` | FLOAT | Cross-course contamination rate |
| `failure_rate` | FLOAT | |

### `query_results`
One row per query per run. Foreign-keyed to both `runs` and `corpus`.

| Column | Type | Notes |
|---|---|---|
| `id` | INT PK | Auto |
| `run_id` | TEXT FK | |
| `query_id` | TEXT | |
| `query_text` | TEXT | |
| `expected_id` | TEXT FK → corpus | |
| `course` | TEXT | |
| `topic` | INT | |
| `subtopic` | INT | |
| `query_type` | TEXT | `chaos_monkey`, `creative_student`, `grounded_analyst`, `original` |
| `hit_ids` | TEXT | JSON-encoded list |
| `hit_scores` | TEXT | JSON-encoded list |
| `latency_ms` | FLOAT | |
| `hit_at_1` | BOOL | |
| `hit_at_5` | BOOL | |
| `rank` | INT | Exact rank of correct answer, NULL if not retrieved |

---

## Code layout

```
src/rag_pipeline/db/
  __init__.py
  models.py      — SQLModel table definitions (single source of truth)
  engine.py      — get_engine(), init_db(), get_session()
  store.py       — write ExperimentResult → DB (coming)
  backfill.py    — migrate existing JSONL/meta files into DB (coming)
```

---

## Config

```json
// configs/db.json
{
  "results_db": "experiments/rag_results.db",
  "mlflow_db":  "experiments/mlflow/mlflow.db"
}
```

Accessed via:
```python
from rag_pipeline.core.paths import Paths
Paths.results_db()   # Path to rag_results.db
Paths.mlflow_db()    # Path to mlflow.db
```

---

## Pydantic ↔ DB layer

`core/models/` and `db/models.py` are kept separate intentionally:

- `core/models/` — validation, business logic, API contracts
- `db/models.py` — persistence, relationships, querying

Conversion happens in `db/store.py` only. No other layer should import from `db/models.py` directly.

---

## Adding a new field

1. Add to `MetricsSnapshot` in `core/models/ablation.py`
2. Add column to `RunMetrics` in `db/models.py`
3. Add migration in `experiments/migrations/NNN_description.sql`
4. Update `store.py` conversion

---

## Useful queries

```sql
-- Best H@1 per experiment
SELECT r.experiment, r.patch, r.config, m.h1, m.mrr
FROM runs r JOIN run_metrics m ON r.run_id = m.run_id
ORDER BY m.h1 DESC;

-- Failure analysis by query type
SELECT query_type, COUNT(*) as n, AVG(hit_at_1) as h1
FROM query_results WHERE run_id = 'baseline__entity_boosted'
GROUP BY query_type;

-- Which corpus docs are hardest to retrieve
SELECT q.expected_id, c.question, COUNT(*) as misses
FROM query_results q JOIN corpus c ON q.expected_id = c.es_id
WHERE q.hit_at_1 = 0
GROUP BY q.expected_id ORDER BY misses DESC LIMIT 20;
```
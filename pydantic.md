# RAG-a-muffin Handoff — 2026-05-30c

## Baseline (pre-session, from ablation at commit `68fd559`)
| Config | H@1 | H@5 | MRR |
|--------|-----|-----|-----|
| entity_boosted (QA collection) | 92.2% | 98.7% | 0.9529 |

## Current state (end of session)
| Config | Collection | H@1 | H@5 | MRR |
|--------|-----------|-----|-----|-----|
| entity_boosted | faqs_bge_base_en_v1_5_qa | 89.6% | 98.5% | 0.9366 |

The 2.6% H@1 gap vs baseline is under investigation. Root cause is likely `_expand_urls` shifting some `ner_primary_entity` values (64 docs changed). MLflow was added this session to track experiments going forward.

---

## What was done this session

### 1. Multi-entity fix: `_tag_ner` emits full entity list (carried from previous session)
Already in place at session start. Confirmed working.

### 2. Bug fix: reclassified docs now append to `ner_entities`
`src/rag_pipeline/eda/topics/core/topic_modeling.py` — after `extract_entity` sets `ner_primary_entity` on reclassified docs, the entity is now also appended to `ner_entities` if not already present:
```python
a["ner_primary_entity"] = entity
if entity and entity not in a["ner_entities"]:
    a["ner_entities"].append(entity)
```
**Caveat**: `extract_entity` returns classification signals (e.g. `ask good questions`, `could not`), not clean named entities. These are useful for category inference but noisy for retrieval. This is a known limitation — see open issues below.

### 3. URL entity pre-processing: `_expand_urls`
Added to `_tag_ner` to help spaCy tag entities buried in URLs like `public.ecr.aws/lambda/python:3.8`.
```python
def _expand_urls(text: str) -> str:
    return re.sub(r'(?:https?://|(?:[a-z0-9_-]+\.){2,}[a-z]{2,}(?:[/:.@][\w.-]*)*)',
        lambda m: m.group(0).replace('/', ' ').replace(':', ' ').replace('.', ' ').replace('@', ' '),
        text)
```
Pipe expanded text through spaCy but key results by original text so the assignment dict stays correct. The regex only fires on actual URLs/paths — not general punctuation — to avoid scrambling normal text tokenization.

### 4. `ner_entities` wired end-to-end through ingestion
**`ingest_models.py` fixes:**
- Added `DocNERInfo` import
- Both `load_ner_map` and `_load_ner_map` now return `dict[str, DocNERInfo]` (was `dict[str, dict]`)
- Both now read `ner_entities` from assignments JSON via `a.get('ner_entities', [])`
- `_prepare_payload` type hint changed from `ner_info: dict` to `ner_info: DocNERInfo`
- Hardcoded module path `rag_pipeline.eda.p02_topic_modeling` fixed to `rag_pipeline.eda.topics.core.topic_modeling`

### 5. `QueryResult` and `MetricSummary` migrated to Pydantic
`src/rag_pipeline/ingestion/benchmark_types.py` — both classes converted from `@dataclass(frozen=True)` to `BaseModel, frozen=True`. `SearchResult` also converted.
- `to_dict()` now calls `self.model_dump()` instead of `asdict(self)`
- `benchmark_persistence.py` updated: removed `from dataclasses import asdict`, all `asdict(s)` → `s.model_dump()`
- All 6 `QueryResult(...)` call sites and 3 `MetricSummary(...)` call sites are compatible — Pydantic v2 keyword construction is identical to dataclass

New fields added to `QueryResult`:
```python
ner_primary_entity: Optional[str] = None
ner_entities: tuple[str, ...] = ()
rank: Optional[int] = None        # exact rank of correct answer, None if not retrieved
hit_at_1: bool = False
hit_at_3: bool = False
hit_at_5: bool = False
```
These are populated in `_build_query_result` in `evaluation.py` from `ctx['ner_primary_entity']`, `ctx['ner_entities']`, and computed rank.

### 6. MLflow infrastructure added
```bash
uv add mlflow
```
- `src/rag_pipeline/ingestion/mlflow_logger.py` created — `log_benchmark_run(cfg_name, summary, results, model_entry, tags)` logs aggregate metrics + per-query JSONL artifact
- `Paths.mlflow_dir()` added to `src/rag_pipeline/core/paths.py` — auto-creates directory
- `configs/paths.json` — added `"mlflow_dir": "experiments/mlflow"`
- Experiment name: `"rag-retrieval"`, tracking URI: `file://{Paths.mlflow_dir()}`
- **Not yet wired into `benchmark.py`** — next step is calling `log_benchmark_run` inside `main()` after `_run_config`

Metrics logged per run:
`h1, h3, h5, h10, mrr, ndcg_10, latency_p50, latency_p95, failure_rate, cross_course, rank_std`

Per-query artifact fields:
`query_id, query_text, expected_id, course, topic, subtopic, query_type, ner_primary_entity, ner_entities, rank, hit_at_1, hit_at_3, hit_at_5, hit_ids, hit_scores, latency_ms`

---

## Architecture & dataflow

### Full pipeline (topic modeling → Qdrant → retrieval)

```
topic_modeling.py
  └─ _tag_ner(questions)          # spaCy NER, emits full entity list
       └─ _expand_urls(text)      # pre-process URL-buried tokens
  └─ _build_assignments(...)      # writes ner_entities, ner_categories per doc
  └─ process_model(...)           # writes per-model JSON to experiments/
       └─ reclassify loop         # may update ner_category + ner_primary_entity
  └─ writes: experiments/topic_assignments_{slug}.json

topic_merge.py (TopicMerger.merge())
  └─ reads: experiments/topic_assignments_*.json  (all models)
  └─ wraps in: {"metadata": {...}, "results": {model_name: file_contents}}
  └─ writes: output/topic_assignments_all.json    ← Paths.topic_assignments()

ingest_models.py
  └─ _load_ner_map(path, model_name)   # reads results[model_name]['assignments']
       └─ returns dict[str, DocNERInfo]
  └─ _prepare_payload(doc, ner_info)   # writes all DocNERInfo fields to Qdrant payload
  └─ _create_points_batch(...)         # assembles PointStructs
  └─ client.upsert(...)                # pushes to Qdrant

evaluation.py (_run_retrieval)
  └─ reads ner_entities from topic_map (QueryContext)
  └─ filters by entity_freq_threshold (from defaults.json, currently 999 = effectively off)
  └─ passes filtered ner_entities list to retriever

composite_retrievers.py (run_entity_category_boosted_retrieval)
  └─ iterates ner_entities, one should clause per entity
  └─ matches against ner_primary_entity field in Qdrant
  └─ fallback: uses ner_primary_entity if ner_entities is empty
```

### Qdrant payload fields (per document)
```
es_id, question, answer, course, section,
ner_category, ner_primary_entity,
ner_entities (list[str]),   ← NEW this session
topic, subtopic
```

### Qdrant collections (current)
| Collection | Encode mode | Notes |
|---|---|---|
| faqs_bge_base_en_v1_5 | question | Old baseline collection |
| faqs_bge_base_en_v1_5_qa | qa | Current production collection |
| faqs_bge_base_en_v1_5_answer | answer | Exists, not used in benchmarks |

---

## Pydantic implementation

### Single source of truth: `DocNERInfo`
`src/rag_pipeline/core/models/topics.py`:
```python
class DocNERInfo(BaseModel, frozen=True):
    ner_category: str = "OTHER"
    ner_primary_entity: Optional[str] = None
    ner_entities: list[str] = []
    topic: int = -1
    subtopic: Optional[int] = None
```

**Rule**: add new NER fields here only. They propagate automatically to `_load_ner_map` (reads from JSON), `_prepare_payload` (writes to Qdrant), and `_build_query_context` (reads into evaluation). No other files need touching unless the retriever uses the field.

**Do not** use `.get('ner_entities', [])` anywhere — use attribute access on `DocNERInfo` instances.

### Topic assignments JSON schemas
Two formats exist:

**Per-model file** (written by `topic_modeling.py --run-all`):
```json
{"metadata": {"model": "...", ...}, "assignments": [...]}
```

**Merged file** (written by `topic_merge.py`, read by `ingest_models.py`):
```json
{"metadata": {"models_merged": [...]}, "results": {"model_name": <per-model file contents>}}
```

The ablation runner reads from `git show HEAD:...` — always commit before running ablation.

---

## Path resolution — always use `Paths`
```python
from rag_pipeline.core.paths import Paths
Paths.clean_jsonl()                              # corpus
Paths.topic_assignments()                        # output/topic_assignments_all.json
Paths.topics_default_output()                    # experiments/topic_assignments.json (flat, no --run-all)
Paths.topics_experiments_dir()                   # experiments/ dir
Paths.ablation_results_dir()
Paths.reranker_results_dir()                     # misnamed — actually benchmark output
Paths.collection_for_model(model, encode_mode)   # Qdrant collection name
Paths.defaults()                                 # configs/defaults.json
Paths.mlflow_dir()                               # experiments/mlflow (auto-created)
Paths.defaults()["entity_freq_threshold"]        # never hardcode
Paths.defaults()["production_model"]
Paths.defaults()["static_llm_model"]
```

---

## How to make changes correctly

### Editing source files
```bash
uv run python - << 'EOF'
from pathlib import Path
p = Path("src/rag_pipeline/...")
t = p.read_text()
old = "exact string"
new = "replacement"
assert t.count(old) == 1, f"matched {t.count(old)} times"
t = t.replace(old, new, 1)
p.write_text(t)
print("OK")
EOF
```
If `OK` is not printed, the string didn't match. Use `cat -A file | sed -n 'N,Mp'` to get exact whitespace before retrying.

### Changing production defaults
Edit `configs/defaults.json`. Never hardcode. Current values:
```json
{
    "production_model":       "BAAI/bge-base-en-v1.5",
    "production_config":      "entity_boosted",
    "production_encode_mode": "qa",
    "entity_freq_threshold":  999,
    "static_llm_model":       "nvidia_nim/meta/llama-3.1-8b-instruct",
    "topic_modeling": {
        "min_topic_size":     5,
        "min_samples":        1,
        "subtopic_threshold": 40,
        "subtopic_min_size":  5,
        "ner_batch_size":     5
    }
}
```

### Full pipeline re-run sequence
```bash
# 1. Topic modeling (per-model file for merger)
rm src/rag_pipeline/eda/topics/experiments/topic_assignments_BAAI_bge_base_en_v1.5.json
uv run python -m rag_pipeline.eda.topics.core.topic_modeling \
    --embedding-model "BAAI/bge-base-en-v1.5" --run-all

# 2. Merge all models into production file
python -c "from rag_pipeline.eda.topics.core.topic_merge import TopicMerger; TopicMerger().merge(); print('OK')"

# 3. Validate JSON
python -c "import json; json.load(open('src/rag_pipeline/eda/topics/output/topic_assignments_all.json')); print('valid')"

# 4. Re-ingest (always use --no-skip-existing when payload schema changed)
uv run python -m rag_pipeline.ingestion.ingest_models \
    --model "BAAI/bge-base-en-v1.5" \
    --encode-mode qa \
    --no-skip-existing

# 5. Benchmark
uv run python -m rag_pipeline.ingestion.benchmark \
    --model "BAAI/bge-base-en-v1.5" \
    --collection faqs_bge_base_en_v1_5_qa \
    --configs entity_boosted

# 6. Ablation (commit first — reads from git HEAD)
git add -A && git commit -m "..."
uv run python -m rag_pipeline.ablation flow --configs entity_boosted --rerun
```

---

## Known issues & open items

### 1. Resolve 89.6% vs 92.2% gap (immediate)
64 docs have changed `ner_primary_entity` vs the 92.2% baseline commit (`68fd559`). Suspect: `_expand_urls` regex shifting entity boundaries for some edge cases. Approach with MLflow:
- Run baseline (no URL expansion) vs current as tracked experiments
- Identify which of the 64 changed docs are in the failing queries

### 2. Noisy `ner_primary_entity` from `extract_entity` (known, not fixed)
When `reclassify` fires (`classification_source == 'rules'`), `extract_entity` returns category keyword signals (e.g. `ask good questions`, `could not`, `out of space`) as `ner_primary_entity`. These pollute Qdrant and the retriever's should clauses. 257 docs are reclassified.
- Potential fix: don't set `ner_primary_entity` when `classification_source == 'rules'`; category alone is sufficient for boosting
- Needs experiment to verify impact

### 3. MLflow integration (in progress)
Infrastructure done. One step remaining to be functional:

**Wire into `benchmark.py` `main()`** — add after `_run_config` call:
```python
from .mlflow_logger import log_benchmark_run
# inside the for loop, after: name, summary, results = _run_config(...)
log_benchmark_run(name, summary, results, model_entry, tags={"collection": model_entry["collection"]})
```

Then wire into ablation:
- `ablation/experiment.py` — call `log_benchmark_run` per patch run, add `"patch"` tag

Planned experiments once wired:
| Experiment | NER system | URL expansion | Goal |
|---|---|---|---|
| baseline_spacy | build_base_nlp | off | reproduce 92.2% |
| url_expand | build_base_nlp | on | measure _expand_urls impact |
| config_ner | build_ner_from_config | off | compare 632-term dictionary vs spaCy |
| config_ner_url | build_ner_from_config | on | combined |
| no_rules_primary | rules reclassify, no primary | — | fix noisy extract_entity primaries |

### 4. Cross-encoder reranking (deferred)
`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~10-20ms/query on CPU. Infrastructure exists. Never benchmarked against QA collection.
```bash
uv run python -m rag_pipeline.ingestion.benchmark_with_reranker \
    --model "BAAI/bge-base-en-v1.5" \
    --collection faqs_bge_base_en_v1_5_qa \
    --configs entity_boosted
```

### 5. Housekeeping (deferred)
- Rename `reranker_results_dir` → `benchmark_results_dir` (3 callers)
- Model registry Pydantic model (`core/models/registry.py` skeleton exists)
- `GENERIC_ENTITIES` centralization
- Remaining dataclass → Pydantic migrations (`QueryResult`, `MetricSummary`, `SearchResult` done this session)
- Cluster entity voting (unblocked by multi-entity schema)
- `llm_ner.py` — keep as future option

---

## Ablation summary (pre-session baseline, QA collection)

| Experiment | Delta H@1 | Meaning |
|---|---|---|
| no_entity | -11.9% | entity signal is load-bearing |
| no_cluster | -3.5% | cluster fallback matters |
| empty_patterns | -3.5% | hand-crafted patterns matter |
| no_rules | -2.8% | rules matter |
| no_generic_entity | -3.2% | generic entities still useful |

Redundant signals (no delta): `ner_category`, `topic` assignments, low-confidence topic nulling.
# RAG-a-muffin Handoff — 2026-05-31 (Session 2)

## Branch
`mlflow-tracking` (continued from Session 1)

## What was done this session

### 1. encode_mode bug fixed
`BenchmarkConfig.from_defaults()` was not reading `production_encode_mode` from `defaults.json` — defaulting to `EncodeMode.question` for all models. Collections are all `_qa`, so query vectors were mismatched.

Fix: `from_defaults()` now reads `defaults["production_encode_mode"]` → `EncodeMode.qa`.

**Root cause of snowflake/mpnet poor results:** after investigation, encode_mode was NOT the cause — `arctic_embed_l_qa.npy` and `mpnet_base_qa.npy` already existed and were being used. Those models genuinely underperform on this dataset.

### 2. Startup validator added to `BenchmarkConfig`
`check_encode_mode_cache_consistency` (model_validator, mode='after') — raises immediately with a clear error if any configured model is missing a cache file for the active `encode_mode`, with a hint if a different suffix exists. Fails at config build time, not mid-run.

### 3. MLflow unified into single module
Deleted `src/rag_pipeline/mlflow/logger.py` and `src/rag_pipeline/mlflow/ablation_logger.py`.

New: `src/rag_pipeline/mlflow/tracking.py` — single source of truth.

```python
from rag_pipeline.mlflow.tracking import log_benchmark_run, log_ablation_run
```

Changes vs old loggers:
- `set_tracking_uri` at module level — no more split between file store and sqlite
- `_dedup()` helper — re-runs cleanly replace old runs in both benchmark and ablation paths
- `log_benchmark_run` now accepts `cfg_dict` and `encode_mode` — full retrieval config logged as MLflow params, encode_mode logged as tag
- All `MetricSummary` fields logged consistently in both paths (previously benchmark hand-picked a subset)
- `__init__.py` exposes both functions

Updated call sites:
- `benchmark.py` line 148: now passes `cfg` dict and `config.encode_mode.value`
- `ablation/experiment.py` line 26: updated import

### 4. MLflow DB cleaned
Removed 30 duplicate runs (92 → 62). Two ablation runs (`ablation__baseline__entity_boosted`, `ablation__no_entity__entity_boosted`) are pre-session and live in `rag-ablation` experiment — correct, leave them.

Deleted empty `mlruns/` stubs (were created before tracking URI was fixed).

### 5. Full benchmark complete — all 6 models × 10 configs

#### Clean results (fail < 0.10)

| Model | Best H@1 | Config | fail |
|-------|----------|--------|------|
| BAAI/bge-base-en-v1.5 | 0.898 | entity_boosted | 0.01 |
| intfloat/e5-large-v2 | 0.808 | entity_boosted | 0.03 |
| BAAI/bge-large-en-v1.5 | 0.795 | entity_boosted | 0.05 |
| BAAI/bge-m3 | 0.771 | entity_boosted | 0.05 |

#### Broken/excluded models
- `Snowflake/snowflake-arctic-embed-l` — vector_default H@1=0.296, fail=0.53. Not an encoding bug. Genuinely poor retrieval on this dataset.
- `microsoft/mpnet-base` — vector_default H@1=0.065, fail=0.81. Sentence similarity model, not a retrieval model. Drop both.

#### Key findings (consistent across all 4 good models)
- `entity_boosted` == `entity_section_boosted` — section signal adds nothing
- `entity_category_boosted` consistently hurts vs entity alone
- `entity_full_boosted` worst of entity configs — topic + category both harmful
- bge-base wins despite being smallest — larger dims don't compensate on this dataset
- 89.8% vs 92.2% gap (bge-base vs commit `68fd559`) still unresolved — suspect `_expand_urls` shifting entity boundaries

---

## Remaining work

### Immediate: sparse benchmarks (blocked)
4 SPLADE models exist in `experiments/_embeddings_gpu/`:
- `sparse_prithivida__Splade_PP_en_v1.npy`
- `sparse_prithivida__Splade_PP_en_v2.npy`
- `sparse_naver__splade-cocondenser-ensembledistil.npy`
- `sparse_naver__splade_v2_max.npy`

Blocker: Qdrant sparse vector collections must be created separately — different from dense. Qdrant supports named sparse vectors alongside dense in same collection. Pipeline does not have sparse ingestion or retrieval wired yet.

Steps needed:
1. Add sparse collection creation to `ingest_models.py` — `VectorParams` → `SparseVectorParams`
2. Add sparse retrieval config in `retrieval_configs.json`
3. Add sparse retrieval path in retrievers
4. Copy GPU sparse embeddings to cache with correct naming convention
5. Benchmark

### Deferred
- **ColBERT** — `colbert_colbert-ir__colbertv2.0.npy` exists. Needs `colbert-ai` package for proper multi-vector eval.
- **NER embeddings** — `ner_Babelscape__wikineural-multilingual-ner.npy`, `ner_dslim__bert-base-NER.npy` — for spaCy replacement experiments, not wired.
- **Topic embeddings** — `topic_sentence-transformers__all-MiniLM-L6-v2.npy` — for BERTopic experiments, not wired.
- **89.8% vs 92.2% gap** — run `baseline_no_url_expand` experiment in MLflow (known issue #1)
- **Noisy reclassified primaries** — don't set `ner_primary_entity` when `classification_source == 'rules'` (known issue #2)
- **`reranker_results_dir` misnamed** — rename to `benchmark_results_dir`, 3 callers (known issue #4)
- **`es_index` hardcoded** — `benchmark_config.py` line 61, should read from `Paths.defaults()` (known issue #5)
- **Cross-encoder reranking** — infrastructure exists, never benchmarked against QA collection (known issue #7)

---

## GPU embeddings — status

| File | Status |
|------|--------|
| `dense_BAAI__bge-large-en-v1.5.npy` | ✓ benchmarked (copied as `bge_large_en_v1_5_qa.npy`) |
| `dense_BAAI__bge-m3.npy` | ✓ benchmarked |
| `dense_intfloat__e5-large-v2.npy` | ✓ benchmarked |
| `dense_microsoft__mpnet-base.npy` | ✓ benchmarked — dropped (poor) |
| `dense_Snowflake__snowflake-arctic-embed-l.npy` | ✓ benchmarked — dropped (poor) |
| `sparse_*` (4 files) | ✗ not wired |
| `colbert_*` | ✗ not wired |
| `ner_*` (2 files) | ✗ not wired |
| `topic_*` | ✗ not wired |

---

## Architecture changes this session

### `BenchmarkConfig` (benchmark_config.py)
- `encode_mode` default now reads from `defaults["production_encode_mode"]` in `from_defaults()`
- `check_encode_mode_cache_consistency` model_validator — raises at startup with actionable error if cache missing for configured encode_mode + model

### `src/rag_pipeline/mlflow/tracking.py` (new)
```python
log_benchmark_run(
    cfg_name: str,
    cfg_dict: dict,          # full retrieval config — logged as MLflow params
    summary: MetricSummary,
    results: list[QueryResult],
    model_entry: dict,
    encode_mode: str,        # logged as tag — prevents silent mismatch
    tags: dict | None = None,
) -> None

log_ablation_run(
    result: ExperimentResult,
    tags: dict | None = None,
) -> None
```

MLflow experiments:
- `rag-retrieval` — benchmark runs
- `rag-ablation` — ablation runs
- Tracking URI: `sqlite:///experiments/mlflow/mlflow.db` (set at module import)

### MLflow DB state
- 62 runs in `rag-retrieval` (6 models × 10 configs + 2 early bge-base pre-session)
- 2 runs in `rag-ablation` (pre-session ablation baseline/no_entity)

HANDOFF












# RAG-a-muffin Handoff — 2026-05-31

## Branch
`mlflow-tracking` (branched from `main` at start of this session)

## Benchmarks in progress
Running all 6 models × all retrieval configs. Results pending. bge-base partial results:

| Config | H@1 | H@5 | MRR | p50ms |
|--------|-----|-----|-----|-------|
| entity_boosted | 89.8% | 98.5% | 0.9377 | 9.9 |
| entity_category_boosted | 89.8% | 98.5% | 0.9377 | 11.8 |
| entity_section_boosted | 89.8% | 98.5% | 0.9377 | 9.8 |
| entity_category_section_boosted | 89.8% | 98.5% | 0.9377 | 12.1 |
| entity_topic_boosted | 87.3% | 98.3% | 0.9205 | 12.0 |
| entity_full_boosted | 87.3% | 98.3% | 0.9205 | 13.9 |
| vector_default | 80.3% | 95.9% | 0.8719 | 9.2 |
| bm25_balanced | 66.3% | 86.0% | 0.7442 | 10.3 |
| bm25_default | 58.3% | 75.4% | 0.6608 | 11.7 |

Key finding: `topic` signal hurts (-2.5% H@1). Entity alone is optimal. Category and section add no value over entity alone.

Pre-session baseline: 92.2% H@1 at commit `68fd559`. Current 89.8% gap is under investigation (see known issues).

---

## What was done this session

### 1. Pydantic migration: `QueryResult`, `MetricSummary`, `SearchResult`
`src/rag_pipeline/ingestion/benchmark_types.py` — all three converted from `@dataclass(frozen=True)` to `BaseModel, frozen=True`.
- `to_dict()` → `model_dump()`
- `benchmark_persistence.py` — removed `asdict` import, all calls → `model_dump()`
- New fields on `QueryResult`: `ner_primary_entity`, `ner_entities`, `rank`, `hit_at_1`, `hit_at_3`, `hit_at_5`
- These are populated in `_build_query_result` in `evaluation.py`

### 2. `load_topic_assignments` migrated to Pydantic
`src/rag_pipeline/ingestion/benchmark_loader.py` — returns `dict[str, DocNERInfo]` instead of `dict[str, dict]`.
All downstream consumers updated to use attribute access:
- `evaluation.py` — `_build_query_context` now constructs `DocNERInfo()` default on miss
- `evaluation.py` — `entity_freq_map` uses `v.ner_primary_entity` not `v.get(...)`
- `benchmark_config.py` — `get_topic_map` return type updated

### 3. MLflow fully wired
- `src/rag_pipeline/mlflow/logger.py` — `log_benchmark_run()` logs aggregate metrics + per-query JSONL artifact
- `src/rag_pipeline/ingestion/benchmark.py` — calls `log_benchmark_run` after each config run
- `configs/paths.json` — `mlflow_dir: experiments/mlflow`
- Tracking URI: `sqlite:///experiments/mlflow/mlflow.db`
- Experiment name: `rag-retrieval`
- Per-query artifact fields: `query_id, query_text, expected_id, course, topic, subtopic, query_type, ner_primary_entity, ner_entities, rank, hit_at_1, hit_at_3, hit_at_5, hit_ids, hit_scores, latency_ms`

### 4. New embedding models added
5 new dense models registered in `configs/models.json` and ingested:
| Model | dims | Collection |
|-------|------|-----------|
| BAAI/bge-large-en-v1.5 | 1024 | faqs_bge_large_en_v1_5_qa |
| microsoft/mpnet-base | 768 | faqs_mpnet_base_qa |
| intfloat/e5-large-v2 | 1024 | faqs_e5_large_v2_qa |
| BAAI/bge-m3 | 1024 | faqs_bge_m3_qa |
| Snowflake/snowflake-arctic-embed-l | 1024 | faqs_snowflake_arctic_embed_l_qa |

GPU embeddings stored in `experiments/_embeddings_gpu/`. Copied to `experiments/embeddings/` with `_qa` suffix for cache lookup. Topic assignments aliased from `bge-base` using `copy.deepcopy` (shared references corrupt JSON — learned this the hard way).

### 5. Qdrant payload indexes created
All collections now have keyword/integer indexes on: `course`, `ner_primary_entity`, `ner_category`, `ner_entities`, `topic`. Previously unindexed = full payload scan on every query.

### 6. Bug fixes
- `es_retrievers.py` — `str()` coerce on `es_id` (2 docs had integer es_id from unknown source)
- `qdrant_retrievers.py` — same `str()` coerce on `es_id`
- `composite_retrievers.py` — `run_hybrid_dbsf` wrong course attribution (`hit_courses[0]` → `hit_courses[rank-1]`)
- `evaluation.py` — hardcoded `top_k=5` in test retrieval → `rc.top_k`
- `benchmark_config.py` — hardcoded `es_index='faqs'` noted (not yet fixed)

### 7. Experiments directory reorganized
```
experiments/
  embeddings/          # .npy cache files
  _embeddings_gpu/     # GPU-trained raw files (gitignored)
  mlflow/              # MLflow tracking DB
  onnx_cache/
  reranker_models/
  reranker_training/
  results/             # all benchmark output (new)
    archive/           # pre-MLflow results
    ablation/          # ablation output
  topic_assignments_all.json
```
`configs/paths.json` updated: `reranker_results_dir` and `ablation_results_dir` → `experiments/results/`

### 8. `pipeline.yaml` created
Documents full dataflow, config sources, Pydantic models, planned experiments, known issues. Lives at project root.

### 9. Tests added
- `tests/test_topic_assignments.py` — JSON integrity, `ner_entities` presence, no shared references
- `tests/test_pydantic_models.py` — `DocNERInfo`, `QueryResult`, `MetricSummary` validation

---

## Architecture & dataflow

### Full pipeline
```
topic_modeling.py
  └─ _expand_urls(text)            # pre-process URL tokens (regex: actual URLs only)
  └─ _tag_ner(questions)           # spaCy NER → full entity list
  └─ _build_assignments(...)       # ner_entities, ner_categories per doc
  └─ reclassify loop               # 257 docs reclassified; noisy primaries (known issue)
  └─ writes: experiments/topic_assignments_{slug}.json

topic_merge.py
  └─ merges per-model files → output/topic_assignments_all.json
  └─ IMPORTANT: always copy.deepcopy when aliasing models
  └─ IMPORTANT: validate JSON after every write
  └─ IMPORTANT: commit before running ablation (reads git HEAD)

ingest_models.py
  └─ _load_ner_map → dict[str, DocNERInfo]
  └─ _prepare_payload → Qdrant payload with ner_entities list
  └─ IMPORTANT: use --no-skip-existing when payload schema changed

benchmark.py
  └─ load_topic_assignments → dict[str, DocNERInfo]
  └─ _build_query_context → reads ner_* from DocNERInfo
  └─ _run_retrieval → filters by entity_freq_threshold (999 = off)
  └─ run_entity_category_boosted_retrieval → should clauses per entity
  └─ log_benchmark_run → MLflow
```

### Qdrant payload fields
```
es_id, question, answer, course, section,
ner_category, ner_primary_entity,
ner_entities (list[str]),
topic, subtopic
```
All filter fields are now indexed (keyword/integer).

### Qdrant collections
| Collection | Model | dims | encode_mode |
|---|---|---|---|
| faqs_bge_base_en_v1_5_qa | BAAI/bge-base-en-v1.5 | 768 | qa |
| faqs_bge_large_en_v1_5_qa | BAAI/bge-large-en-v1.5 | 1024 | qa |
| faqs_mpnet_base_qa | microsoft/mpnet-base | 768 | qa |
| faqs_e5_large_v2_qa | intfloat/e5-large-v2 | 1024 | qa |
| faqs_bge_m3_qa | BAAI/bge-m3 | 1024 | qa |
| faqs_snowflake_arctic_embed_l_qa | Snowflake/snowflake-arctic-embed-l | 1024 | qa |

---

## Pydantic models — single sources of truth

| Model | Path | Key rule |
|---|---|---|
| `DocNERInfo` | `core/models/topics.py` | Add NER fields here only |
| `FAQDocument` | `core/models/faq.py` | `id` always str, validated non-empty |
| `QueryResult` | `ingestion/benchmark_types.py` | frozen, includes rank/hit_at_k/ner_* |
| `MetricSummary` | `ingestion/benchmark_types.py` | frozen, `to_dict()` = `model_dump()` |
| `SearchResult` | `ingestion/benchmark_types.py` | frozen, `es_id` always str-coerced |
| `BenchmarkConfig` | `ingestion/benchmark_config.py` | Pydantic, reads from defaults.json |

**Do not** use `.get()` on `DocNERInfo` — use attribute access.

---

## Path resolution — always use `Paths`
```python
from rag_pipeline.core.paths import Paths
Paths.clean_jsonl()
Paths.topic_assignments()            # output/topic_assignments_all.json
Paths.topics_default_output()        # experiments/topic_assignments.json (flat)
Paths.topics_experiments_dir()
Paths.ablation_results_dir()         # experiments/results/ablation
Paths.reranker_results_dir()         # experiments/results (misnamed, see known issues)
Paths.mlflow_dir()                   # experiments/mlflow (auto-created)
Paths.mlflow_db()                    # experiments/mlflow/mlflow.db
Paths.collection_for_model(model, EncodeMode.qa)
Paths.defaults()["entity_freq_threshold"]   # never hardcode
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
p.write_text(t.replace(old, new, 1))
print("OK")
EOF
```
If assert fails: `cat -A file | sed -n 'N,Mp'` to get exact whitespace.

### Full pipeline re-run
```bash
# 1. Topic modeling
rm src/rag_pipeline/eda/topics/experiments/topic_assignments_BAAI_bge_base_en_v1.5.json
uv run python -m rag_pipeline.eda.topics.core.topic_modeling \
    --embedding-model "BAAI/bge-base-en-v1.5" --run-all

# 2. Merge (always copy.deepcopy for aliases)
python -c "
import json, copy
from rag_pipeline.eda.topics.core.topic_merge import TopicMerger
from pathlib import Path
TopicMerger().merge()
# Re-add aliases
p = Path('src/rag_pipeline/eda/topics/output/topic_assignments_all.json')
data = json.load(open(p))
source = copy.deepcopy(data['results']['BAAI/bge-base-en-v1.5'])
for model in ['BAAI/bge-large-en-v1.5','microsoft/mpnet-base','intfloat/e5-large-v2','BAAI/bge-m3','Snowflake/snowflake-arctic-embed-l']:
    data['results'][model] = copy.deepcopy(source)
    if model not in data['metadata']['models_merged']:
        data['metadata']['models_merged'].append(model)
text = json.dumps(data, indent=2)
json.loads(text)  # validate
p.write_text(text)
print('OK')
"

# 3. Validate
python -c "import json; json.load(open('src/rag_pipeline/eda/topics/output/topic_assignments_all.json')); print('valid')"

# 4. Ingest (--no-skip-existing when payload schema changed)
uv run python -m rag_pipeline.ingestion.ingest_models \
    --model "BAAI/bge-base-en-v1.5" --encode-mode qa --no-skip-existing

# 5. Benchmark (results go to experiments/results/, tracked in MLflow)
uv run python -m rag_pipeline.ingestion.benchmark \
    --model "BAAI/bge-base-en-v1.5" \
    --collection faqs_bge_base_en_v1_5_qa

# 6. Ablation (commit first)
git add -A && git commit -m "..."
uv run python -m rag_pipeline.ablation flow --configs entity_boosted --rerun
```

---

## GPU embeddings

Raw files in `experiments/_embeddings_gpu/` (gitignored). Copied to `experiments/embeddings/` with `_qa` suffix for cache.

Categories trained on GPU:
- **dense**: standard SentenceTransformer encode on answers
- **sparse**: SPLADE — `log(1+relu(logits))` max-pooled, vocab-sized vectors (~30k dims). **Not yet wired into Qdrant** — needs sparse vector collection support.
- **colbert**: mean-pooled token embeddings. Needs `colbert-ai` package for proper multi-vector eval.
- **ner**: encoded questions (not answers) — for spaCy replacement experiments
- **topic**: encoded questions — for BERTopic experiments

---

## Known issues & open experiments

### 1. 89.8% vs 92.2% gap (immediate)
64 docs changed `ner_primary_entity` vs baseline commit `68fd559`. Suspect: `_expand_urls` shifting entity boundaries.
Plan: run `baseline_no_url_expand` experiment in MLflow.

### 2. Noisy reclassified primaries (257 docs)
`extract_entity()` sets keyword signals as `ner_primary_entity` (e.g. `ask good questions`, `could not`).
Fix: don't set `ner_primary_entity` when `classification_source == 'rules'`.

### 3. Sparse retrieval not wired
SPLADE embeddings exist. Qdrant sparse vector collections need to be created separately — different from dense collections. Qdrant supports named sparse vectors alongside dense.

### 4. `reranker_results_dir` misnamed
`Paths.reranker_results_dir()` returns `experiments/results` — actually the general benchmark output. Rename to `benchmark_results_dir` (3 callers).

### 5. `es_index` hardcoded default
`benchmark_config.py` line 61: `es_index: str = 'faqs'`. Should read from `Paths.defaults()`.

### 6. Integer `es_id` in ES
2 docs indexed with integer `es_id`. Source unknown (likely old notebook). Coerced to str at read time. Real fix: reindex those 2 docs.

### 7. Cross-encoder reranking (deferred)
`cross-encoder/ms-marco-MiniLM-L-6-v2` infrastructure exists. Never benchmarked against QA collection.

### 8. Housekeeping
- Model registry Pydantic model (`core/models/registry.py` skeleton exists)
- `GENERIC_ENTITIES` centralization
- Cluster entity voting (unblocked)
- `llm_ner.py` — keep as option

---

## Ablation summary (pre-session, bge-base QA collection)

| Experiment | Delta H@1 | Meaning |
|---|---|---|
| no_entity | -11.9% | entity is load-bearing |
| no_cluster | -3.5% | cluster fallback matters |
| empty_patterns | -3.5% | hand-crafted patterns matter |
| no_rules | -2.8% | rules matter |
| no_generic_entity | -3.2% | generic entities still useful |
| topic signal | -2.5% | topic hurts, not helps |

Redundant signals: `ner_category`, low-confidence topic nulling.
HANDOFF
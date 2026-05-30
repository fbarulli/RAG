cat > HANDOFF_2026_05_30.md << 'EOF'
# RAG-a-muffin Handoff — 2026-05-30

## Key finding this session
QA encoding (question + answer at index time) outperforms question-only by ~8.6% H@1.
Production collection is now `faqs_bge_base_en_v1_5_qa`.
All ablation results are now against the QA collection.

## Current baseline (QA collection / entity_boosted)
| Config | H@1 | H@5 | MRR |
|--------|-----|-----|-----|
| entity_boosted | 92.2% | 98.7% | 0.9529 |

### By query type
| query_type | H@1 |
|------------|-----|
| chaos_monkey | 86.4% |
| creative_student | 92.5% |
| grounded_analyst | 95.2% |
| original | 100.0% |

---

## What was done this session

### 1. Deleted `ingest_qdrant.py`
Orphaned file — fully superseded by `ingest_models.py`. Had two bugs:
- hardcoded QA encoding regardless of mode
- no `encode_mode` awareness

Ablation (`experiment.py`) and corpus sampler already use `ingest_models.py`.

### 2. `EncodeMode` enum
`src/rag_pipeline/core/models/encode_mode.py`:
```python
class EncodeMode(str, Enum):
    question = "question"
    qa       = "qa"
    answer   = "answer"

    @property
    def suffix(self) -> str:
        return {"qa": "_qa", "answer": "_answer"}.get(self.value, "")

    def encode_text(self, question: str, answer: str) -> str:
        ...
```
Exported from `core/models/__init__.py`.
Used in `BenchmarkConfig.encode_mode`, `ingest_models.py`, `Experiment.encode_mode`.

**Note:** `Paths.collection_for_model()` accepts both `EncodeMode` and `str` via
`hasattr(encode_mode, "suffix")` fallback — avoids circular import
(`core/models/` imports `Paths` in deferred fashion; `paths.py` cannot import from `core/models/`).

### 3. `production_encode_mode` in `defaults.json`
```json
"production_encode_mode": "qa"
```
Read by `create_ablation_parser()` and `Experiment` default factory.
Ablation now benchmarks against the QA collection by default.

### 4. Ablation fixes
- `_production_defaults()` was deleted but still referenced → replaced with direct `Paths.defaults()` calls
- `_results_dir()` was deleted but still referenced → replaced with `Paths.ablation_results_dir()`
- `asdict(result)` → `result.model_dump()` (Pydantic migration remnant)
- `compare._load()` now globs `{name}__*_query_results.jsonl` instead of exact match
- `Experiment` now has `encode_mode: EncodeMode` field
- Benchmark subprocess call passes `--collection` derived from `Paths.collection_for_model(self.model, self.encode_mode)`
- Both ingest subprocess calls pass `--encode-mode "{self.encode_mode.value}"`

### 5. `reranker_results_dir` naming
Still misnamed — used as general benchmark output dir, not reranker-specific.
Three callers: `benchmark_config.py`, `experiment.py`, `paths.py`.
Rename deferred — cosmetic only, non-trivial due to `paths.json` key change.

---

## How to make changes correctly

### Adding a new encode mode
1. Add variant to `EncodeMode` in `src/rag_pipeline/core/models/encode_mode.py`
2. Add `suffix` mapping in `EncodeMode.suffix`
3. Add text prep in `EncodeMode.encode_text()`
4. Add to CLI `choices` in `configs/benchmark_cli.py` (two places: `create_ingestion_parser` and `create_ablation_parser`)
5. Re-ingest: `uv run python -m rag_pipeline.ingestion.ingest_models --model "BAAI/bge-base-en-v1.5" --encode-mode <new_mode>`

### Adding a new retrieval config
1. Add entry to `configs/retrieval_configs.json`
2. Add handler in `src/rag_pipeline/ingestion/benchmark_metrics_data/evaluation.py` `_run_retrieval()`
3. Implement retriever in `composite_retrievers.py` or `qdrant_retrievers.py`
4. Test: `uv run python -m rag_pipeline.ingestion.benchmark --model "BAAI/bge-base-en-v1.5" --configs <new_config>`

### Adding a new ablation experiment
Add to `fast` or `slow` list in `src/rag_pipeline/ablation/cli.py`:
```python
("my_experiment", Patch(null_entity=True)),  # fast — no rerun
("my_rerun_exp",  Patch(skip_cluster=True)), # slow — needs rerun
```
Patch flags defined in `src/rag_pipeline/core/models/ablation.py`.

### Changing production defaults
Edit `configs/defaults.json`:
```json
"production_model":       "BAAI/bge-base-en-v1.5",
"production_config":      "entity_boosted",
"production_encode_mode": "qa"
```
All scripts read these at startup via `Paths.defaults()`.

### Path resolution
Always use `Paths` — never hardcode paths:
```python
from rag_pipeline.core.paths import Paths
Paths.clean_jsonl()          # corpus
Paths.topic_assignments()    # NER/topic assignments
Paths.ablation_results_dir() # ablation JSONL + meta files
Paths.reranker_results_dir() # benchmark output (misnamed, but don't rename yet)
Paths.collection_for_model(model_name, encode_mode)  # Qdrant collection name
Paths.defaults()             # reads configs/defaults.json
```

### Model registry
`configs/models.json` — source of truth for embedding models.
Access via:
```python
from rag_pipeline.ingestion.benchmark_loader import load_model_registry, get_model_entry
```
Never hardcode `"BAAI/bge-base-en-v1.5"` — use `Paths.defaults()["production_model"]`.

### Making edits to source files
Use Python str.replace, not sed — sed is brittle with special characters:
```bash
uv run python - << 'EOF'
from pathlib import Path
p = Path("src/rag_pipeline/...")
text = p.read_text()
text = text.replace("old string", "new string")
p.write_text(text)
print("done")
EOF
```

---

## What is left

### 1. Entity frequency tiers (next priority)
High-frequency entities (`error`, `docker`, `homework`, `gcp` etc. — >10 docs)
get the same `should` boost as precise entities (`leaderboard` — 3 docs).
Proposed fix: skip entity boost when `entity_count > threshold`.
Entity counts available from topic assignments at benchmark time.

Generic entities (>10 docs): error, docker, homework, gcp, dbt, course, python,
project, aws, model, jupyter, kestra, mlflow, spark, pandas, valueerror, mage,
bigquery, failed, terraform, github, openai, pipenv, certificate, dlt.

Implementation: precompute entity frequency map at benchmark startup,
pass to retriever, skip `should` clause for generic entities.

### 2. Two-step RRF variant
Fuse entity-filtered results with pure vector results via RRF.
Naturally handles generic entities — precise entities dominate fusion,
generic ones get diluted. No new infrastructure needed.
Add as new retrieval config in `retrieval_configs.json`.

### 3. Cross-encoder reranking (CPU-feasible)
`cross-encoder/ms-marco-MiniLM-L-6-v2` runs on CPU ~10-20ms per query.
Infrastructure already exists (`onnx_bench.py`, `benchmark_reranker.py`).
Benchmark against QA collection — never properly evaluated.

### 4. Remaining handoff items (from 2026-05-29)
See previous handoff for: model registry Pydantic model, GENERIC_ENTITIES
centralization, remaining dataclass→Pydantic migrations, multi-category
tagging, cluster entity voting, folder restructure.

---

## Commands to resume

```bash
# Verify imports
uv run python -c "
from rag_pipeline.core.models import FAQDocument, ProviderConfig, MultiLLMResult, TopicAssignment, TopicAssignments, Patch, ExperimentResult, EncodeMode
from rag_pipeline.core.schemas import FAQDocument
from rag_pipeline.ablation.experiment import Experiment
print('OK')
"

# Run ablation (QA collection, entity_boosted)
uv run python -m rag_pipeline.ablation flow --configs entity_boosted --rerun

# Full report
uv run python -m rag_pipeline.ablation report

# Benchmark a specific collection
uv run python -m rag_pipeline.ingestion.benchmark \
    --model "BAAI/bge-base-en-v1.5" \
    --collection faqs_bge_base_en_v1_5_qa \
    --configs entity_boosted

# Ingest with specific encode mode
uv run python -m rag_pipeline.ingestion.ingest_models \
    --model "BAAI/bge-base-en-v1.5" \
    --encode-mode qa
```

---

## Ablation results summary (QA collection)

### Top performers
| Experiment | Patch | H@1 | MRR |
|---|---|---|---|
| baseline | baseline | 92.2% | 0.9529 |
| no_category | no_category | 92.2% | 0.9529 |
| no_topics | no_topics | 92.2% | 0.9529 |

### Key regressions
| Experiment | Delta |
|---|---|
| no_entity | -11.9% |
| no_cluster | -3.5% |
| empty_patterns | -3.5% |
| no_rules | -2.8% |
| no_generic_entity | -3.2% |

Entity is the dominant signal. Category and topics are redundant.
EOF
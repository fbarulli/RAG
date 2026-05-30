# RAG-a-muffin Handoff — 2026-05-30 (updated)

## Key findings this session
- QA encoding outperforms question-only by ~8.6% H@1. Production collection: `faqs_bge_base_en_v1_5_qa`
- Entity boost is the dominant retrieval signal (-11.9% without it)
- Entity frequency filtering does NOT help — every threshold hurts vs no filtering
- spaCy NER is the wrong tool — it's rule-based, single-entity, hand-crafted patterns
- Next: replace spaCy NER with LLM-based multi-entity extraction

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
Orphaned, superseded by `ingest_models.py`. Had two bugs: hardcoded QA encoding, no encode_mode awareness. Ablation and corpus sampler already use `ingest_models.py`.

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

**Circular import note:** `paths.py` cannot import from `core/models/` — `core/models/` imports
`Paths` in deferred fashion inside methods. `collection_for_model()` accepts both `EncodeMode`
and `str` via `hasattr(encode_mode, "suffix")` fallback.

### 3. `production_encode_mode` in `defaults.json`
```json
"production_encode_mode": "qa",
"entity_freq_threshold": 999
```
All scripts read these via `Paths.defaults()`. No hardcoded fallbacks in retrieval logic.

### 4. Ablation fixes (Pydantic migration remnants)
- `_production_defaults()` deleted but referenced → replaced with `Paths.defaults()` calls
- `_results_dir()` deleted but referenced → replaced with `Paths.ablation_results_dir()`
- `asdict(result)` → `result.model_dump()`
- `compare._load()` now globs `{name}__*_query_results.jsonl`
- `Experiment` has `encode_mode: EncodeMode` field
- Benchmark subprocess passes `--collection` from `Paths.collection_for_model(self.model, self.encode_mode)`
- Both ingest subprocess calls pass `--encode-mode "{self.encode_mode.value}"`

### 5. Entity frequency tier experiment (tried, failed)
Added `entity_freq_threshold` to `RetrievalConfig` and `defaults.json`.
Grid search over thresholds 5–999 showed every threshold hurts vs 999 (no filtering).
Threshold set to 999 (effectively disabled) — generic entities like `docker`, `aws`
are still useful signals even at high frequency.
Infrastructure kept in place for future experimentation.

### 6. `reranker_results_dir` naming
Still misnamed — used as general benchmark output dir.
Three callers: `benchmark_config.py`, `experiment.py`, `paths.py`.
Rename deferred — cosmetic only.

---

## Why entity boosting works (and its limits)
Entity boost adds a Qdrant `should` clause — docs matching the query's entity get a
score boost. Works well for precise entities (`leaderboard` → 3 docs). Less
discriminating for generic entities (`aws` → 28 docs, `docker` → 78 docs) but still
net positive.

Root cause of remaining failures: spaCy NER assigns ONE entity per doc, taking
`ents[0]` and discarding the rest. A doc about a Docker + AWS error gets `docker` OR
`aws`, not both. Queries mentioning the other signal miss the boost entirely.

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
"production_model":         "BAAI/bge-base-en-v1.5",
"production_config":        "entity_boosted",
"production_encode_mode":   "qa",
"entity_freq_threshold":    999
```
All scripts read these at startup via `Paths.defaults()`. Never hardcode.

### Path resolution — always use `Paths`
```python
from rag_pipeline.core.paths import Paths
Paths.clean_jsonl()                              # corpus
Paths.topic_assignments()                        # NER/topic assignments
Paths.ablation_results_dir()                     # ablation JSONL + meta
Paths.reranker_results_dir()                     # benchmark output (misnamed)
Paths.collection_for_model(model, encode_mode)   # Qdrant collection name
Paths.defaults()                                 # configs/defaults.json
```

### Model registry — never hardcode model names
```python
from rag_pipeline.ingestion.benchmark_loader import load_model_registry, get_model_entry
# or
Paths.defaults()["production_model"]
```

### Editing source files — use Python str.replace, not sed
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

### 1. LLM-based multi-entity NER (next priority)
Replace spaCy NER pipeline with LLM structured extraction.
Returns `entities: list[str]` and `categories: list[str]` per doc.
Validated by Pydantic. Removes all hand-crafted pattern machinery.

Current spaCy pipeline (`topic_modeling.py`):
- `_tag_ner()` → `ents[0].text.lower()` — discards all but first entity
- `ClassificationRules.reclassify()` — hand-crafted rules to fix spaCy mistakes
- `extract_missed_terms()` + `suggest_patterns()` — manual pattern discovery

Proposed replacement:
```python
class NERResult(BaseModel):
    entities: list[str]
    categories: list[str]  # TOOL | ERROR | CONCEPT | LANGUAGE | ADMIN
    primary_entity: str | None
    primary_category: str
```
LLM call via existing `llm_client`/`gem_client` infrastructure.
Batch processing — 1207 docs, runs once per corpus update.

Schema changes needed:
- `FAQDocument`: add `ner_entities: list[str]`, `ner_categories: list[str]`
- Qdrant payload: same fields
- Retriever `should` clause: iterate list instead of single value
- `TopicAssignment` in `core/models/topics.py`: add list fields

### 2. Two-step RRF variant
Fuse entity-filtered results with pure vector results via RRF.
No new infrastructure needed — add as new retrieval config.

### 3. Cross-encoder reranking (CPU-feasible)
`cross-encoder/ms-marco-MiniLM-L-6-v2` runs on CPU ~10-20ms/query.
Infrastructure exists (`onnx_bench.py`, `benchmark_reranker.py`).
Never benchmarked against QA collection.

### 4. Remaining from 2026-05-29 handoff
- Model registry Pydantic model (`core/models/registry.py` skeleton exists)
- `GENERIC_ENTITIES` centralization
- Remaining dataclass → Pydantic migrations (see previous handoff)
- Cluster entity voting (depends on multi-entity schema)
- Folder restructure of `ingestion/` (deferred, cosmetic)
- Rename `reranker_results_dir` → `benchmark_results_dir`

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

# Benchmark specific collection
uv run python -m rag_pipeline.ingestion.benchmark \
    --model "BAAI/bge-base-en-v1.5" \
    --collection faqs_bge_base_en_v1_5_qa \
    --configs entity_boosted

# Ingest with encode mode
uv run python -m rag_pipeline.ingestion.ingest_models \
    --model "BAAI/bge-base-en-v1.5" \
    --encode-mode qa

# Grid search entity threshold
uv run python - << 'EOF'
# see previous session for full script
EOF
```

---

## Ablation summary (QA collection)

### Key regressions vs baseline (92.2% H@1)
| Experiment | Delta | Meaning |
|---|---|---|
| no_entity | -11.9% | entity is load-bearing |
| no_cluster | -3.5% | cluster fallback matters |
| empty_patterns | -3.5% | hand-crafted patterns matter |
| no_rules | -2.8% | rules matter |
| no_generic_entity | -3.2% | generic entities still useful |

### Redundant signals (no impact)
- `ner_category` — identical to baseline
- `topic` assignments — identical to baseline
- Low-confidence topic nulling — identical to baseline

Entity is the only signal that matters. Everything else is noise at current scale.
# Session Notes — 2026-05-29

## Goal
Consolidate two ingestion paths (`ingest_qdrant.py` and `ingest_models.py`) and validate
encode mode (`question` vs `qa`) with a real benchmark comparison.

---

## 1. Ablation Migration: `ingest_qdrant.py` → `ingest_models.py`

**Decision:** migrate. `ingest_models.py` is strictly better — batching, tqdm, atomic cache
writes, skip_existing, hard NER failure, multi-model support, correct encode_mode wiring.

**Arg mapping:**

| `ingest_qdrant.py` | `ingest_models.py` |
|---|---|
| `--model` | `--models` |
| `--input` | `--clean-path` |
| `--host` | `--qdrant-host` |
| `--port` | `--qdrant-port` |

All args already existed in `configs/benchmark_cli.py` — no new args needed.

**Files changed:**
- `src/rag_pipeline/ablation/experiment.py` — 2 call sites
- `src/rag_pipeline/ablation/corpus_sampler.py` — 2 call sites (fraction re-ingest + restore)

---

## 2. Bug Fixes in `ingest_models.py`

### Missing stdlib imports
`tempfile` and `shutil` were used but never imported. Added both.

### Embedding cache key ignores encode_mode
Cache key was `{short_name}.npy` regardless of encode mode. QA ingest was loading
question-mode vectors from cache, producing identical collections.

**Fix:** cache key is now `{short_name}_qa.npy` when `encode_mode == 'qa'`,
`{short_name}.npy` otherwise. Applied to both load and save paths.

---

## 3. Benchmark Results

### Question mode (`faqs_bge_base_en_v1_5`)
| Config | H@1 | MRR |
|---|---|---|
| entity_boosted | 83.6% | 0.9023 |
| vector_default | 70.8% | 0.7906 |

### QA mode (`faqs_bge_base_en_v1_5_qa`)
| Config | H@1 | MRR |
|---|---|---|
| entity_boosted | **92.2%** | **0.9529** |
| vector_default | **80.3%** | **0.8719** |

**QA encode mode is uniformly better across all configs (+8-10pp H@1).**
The `__q_only` collection is orphaned from pipeline naming and can be deleted.

---

## 4. `BenchmarkConfig` — Dataclass → Pydantic

**Why:** project standard is Pydantic everywhere.

**Gotcha:** `cached_property` descriptors (`qdrant_client`, `es_client`) are picked up
as fields unless opted out:
```python
model_config = ConfigDict(ignored_types=(cached_property,))
```

**Changes:**
- `@dataclass` → `BaseModel`
- `encode_mode: str` → `encode_mode: Literal['question', 'qa']` — validated at construction
- `__repr__`: `__dataclass_fields__` → `model_fields`
- `_bool_flag` in `merge_args`: was returning `None` when flag absent; Pydantic rejects
  `None` for `bool` fields. Fixed to fall back to `getattr(self, attr)`.

---

## 5. Open / Next

- [ ] `benchmark_report.py` — column widths not dynamic (breaks on long config names),
  no per-query-type breakdown. `query_results_map` passed but unused.
- [ ] Verify `results_map` key format in `benchmark.py` before rewriting report.
- [ ] `ablation/TODO.md` line 195 still references `ingest_qdrant` — update or delete.
- [ ] Delete orphaned `faqs_bge_base_en_v1_5__q_only` collection from Qdrant.








# RAG-a-muffin Handoff — 2026-05-29

## What was done this session

### 1. Root-cause fix: integer IDs in corpus
Two raw markdown files had unquoted integer IDs in YAML frontmatter:
- `data/raw/machine-learning-zoomcamp/module-4/030_6739977244_predict-log-proba-vs-predict-proba.md`
- `data/raw/machine-learning-zoomcamp/module-9/041_9506089527_lambda-oci-manifest.md`

**Fix:** Quoted the IDs in frontmatter (`id: '6739977244'`).

**Guard added in `parse.py`:** After `doc_id = frontmatter.get('id', '')`, added:
```python
if not isinstance(doc_id, str):
    return (None, f"'id' must be a string, got {type(doc_id).__name__}: {doc_id!r} — quote it in frontmatter")
```
Also changed `question` non-string handling from silent coercion to hard failure.
Corpus regenerated: `just run parse && just run dedup`.

---

### 2. Ablation report: per-query-type columns
- Added `_qt_breakdown(name, cfg)` to `report.py` — reads JSONL directly from `ablation_results_dir`.
- Fixed `compare._load()` — was pointing at `reranker_results_dir()` instead of `ablation_results_dir()`.
- Extracted `breakdown_from_jsonl(jsonl_path)` into `compare.py` as a reusable function.
- Report now renders two stacked tables (TOP / BOTTOM performers) with inline columns: `chaos`, `creative`, `grounded`, `original`.
- Removed the old `--- Per-query-type breakdown vs baseline ---` block.

---

### 3. Pydantic migration — central model registry

#### New directory: `src/rag_pipeline/core/models/`
| File | Contents |
|------|----------|
| `faq.py` | `FAQDocument` |
| `llm.py` | `ProviderConfig`, `MultiLLMResult` |
| `topics.py` | `TopicAssignment`, `TopicAssignments` |
| `ablation.py` | `Patch`, `ExperimentResult`, `GENERIC_ENTITIES` |
| `__init__.py` | Re-exports all of the above |

`core/schemas.py` is now a **backwards-compatibility shim** — all existing imports still work.

#### `TopicAssignments` consolidation
- Deleted `eda/topics/core/topic_assignments.py` (duplicate, nothing imported it).
- Moved `iter_assignments()` and `get_sample()` onto `core/models/topics.TopicAssignments`.

#### `ablation/experiment.py`
- `Patch`, `ExperimentResult` removed from file — now imported from `core.models`.
- `Experiment` migrated from `@dataclass` to `BaseModel` with `Field(default_factory=...)`.
- `git_commit` field removed from `ExperimentResult`.

#### All dataclasses in `schemas.py` migrated to Pydantic
`ProviderConfig`, `MultiLLMResult`, `TopicAssignment`, `TopicAssignments` — all now `BaseModel`.

---

### 4. Hardcoded model string fixes
Two files hardcoded `'BAAI/bge-base-en-v1.5'` as a dict key:
- `eda/topics/classification/ner_from_config.py`
- `eda/topics/classification/entity_pattern_learner.py`

**Fix:**
```python
# Before
assignments = data['results']['BAAI/bge-base-en-v1.5']['assignments']

# After
assignments = data['results'][Paths.defaults()['production_model']]['assignments']
```

---

### 5. `cleaning/README.md` updated
Reflects current filenames (`just run` commands), Pydantic schema docs, public function signatures, and the ID quoting rule.

---

## What is left

### 1. Encode mode ablation (next priority)
Add `--encode-mode question|qa` to `benchmark_cli` and `create_ingestion_parser`.
Wire through `BenchmarkConfig`, `ingest_qdrant`, `ingest_models`.
`Paths.collection_for_model` needs `encode_mode` param — Q+A gets `_qa` suffix.
Cache keys: question-only uses existing short name, Q+A uses `qa_` prefix.

### 2. Remaining hardcoded strings
The following still hardcode `BAAI/bge-base-en-v1.5` or `entity_boosted` — audit each:
```bash
grep -rn "BAAI/bge-base-en-v1.5" src/ --include="*.py" | grep -v ".pyc\|test\|docstring\|#"
```
Fix pattern — replace with:
```python
from rag_pipeline.core.paths import Paths
_d = Paths.defaults()
model  = _d["production_model"]   # "BAAI/bge-base-en-v1.5"
config = _d["production_config"]  # "entity_boosted"
```

### 3. `models.json` — model registry Pydantic model
`configs/models.json` already exists with a full model registry. Add a Pydantic model for it in `core/models/`:

```python
# core/models/registry.py
from pydantic import BaseModel

class EmbeddingModel(BaseModel, frozen=True):
    name: str
    short_name: str
    collection: str
    es_index: str
    dims: int
    trust_remote_code: bool = False
    enabled: bool = True
    tier: str = "balanced"
    winner: bool = False
    description: str = ""

class ModelRegistry(BaseModel, frozen=True):
    models: list[EmbeddingModel]
    embeddings_cache_dir: str

    @classmethod
    def load(cls) -> "ModelRegistry":
        from rag_pipeline.core.paths import Paths
        import json
        with open(Paths.models_config()) as f:
            return cls(**json.load(f))

    def production(self) -> EmbeddingModel:
        from rag_pipeline.core.paths import Paths
        name = Paths.defaults()["production_model"]
        return next(m for m in self.models if m.name == name)

    def enabled_models(self) -> list[EmbeddingModel]:
        return [m for m in self.models if m.enabled]
```

Add `models_config` to `Paths` if not already there:
```python
@classmethod
def models_config(cls) -> Path:
    return cls._resolve(cls._cfg.get("models_config", "configs/models.json"))
```

Export `EmbeddingModel`, `ModelRegistry` from `core/models/__init__.py`.

### 4. `GENERIC_ENTITIES` centralization
Currently defined in `core/models/ablation.py`. Any module that needs it should import from there:
```python
from rag_pipeline.core.models import GENERIC_ENTITIES
```
Audit for any other copies:
```bash
grep -rn "GENERIC_ENTITIES\|generic_entities" src/ --include="*.py" | grep -v ".pyc"
```

### 5. Remaining dataclass → Pydantic migrations
These are lower priority — migrate when touching the module:
| File | Classes |
|------|---------|
| `ingestion/benchmark_types.py` | `QueryResult`, `BenchmarkSummary` etc. |
| `ingestion/benchmark_config.py` | `BenchmarkConfig` |
| `answer_generation/config.py` | pipeline config dataclasses |
| `answer_generation/runner.py` | `PipelineResult` |
| `core/gem_client.py` | client config |
| `core/llm_client.py` | client config |

Migration pattern — replace:
```python
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class Foo:
    x: str
    y: int = 0

asdict(foo)
```
With:
```python
from pydantic import BaseModel

class Foo(BaseModel, frozen=True):
    x: str
    y: int = 0

foo.model_dump()
```

### 6. Multi-category tagging
`FAQDocument` and Qdrant payload hold one `ner_category` and one `ner_primary_entity`.
Schema change: `ner_categories: list[str]`, `ner_entities: list[str]`.
Retriever `should` clause iterates list instead of single value.

### 7. Cluster entity voting
If cluster stays as fallback after rules, add `ner_primary_entity` majority vote
alongside existing category vote.

### 8. Folder restructure (deferred)
`src/rag_pipeline/ingestion/` is too flat. Proposed:
```
ingestion/
├── corpus.py
├── ner_map.py
├── encoder.py
├── qdrant/
│   ├── collection.py
│   ├── points.py
│   └── ingest.py
└── benchmark/
```

### 9. Commit
```bash
git add -A
git commit -m "feat: pydantic model registry, ablation report improvements, corpus ID fix"
```

---

## Commands to resume

```bash
# Verify imports
python -c "
from rag_pipeline.core.models import FAQDocument, ProviderConfig, MultiLLMResult, TopicAssignment, TopicAssignments, Patch, ExperimentResult
from rag_pipeline.core.schemas import FAQDocument
from rag_pipeline.ablation.experiment import Experiment
print('OK')
"

# Run ablation
uv run python -m rag_pipeline.ablation flow --configs entity_boosted vector_default --rerun

# Full report
uv run python -m rag_pipeline.ablation report

# Rebuild corpus from scratch
just run parse
just run dedup
```

---

## Baseline (current best)
| Config | H@1 | MRR |
|--------|-----|-----|
| no_cluster / entity_boosted | 92.4% | 0.9540 |

### By query type (no_cluster / entity_boosted)
| query_type | H@1 |
|------------|-----|
| chaos_monkey | 86.4% |
| creative_student | 92.5% |
| grounded_analyst | 95.9% |
| original | 100.0% |
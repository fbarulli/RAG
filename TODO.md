# Structural Cleanup Handoff
_Generated: 2026-05-27_

---

## Priority 1 — Easy, High Impact (do first)

### 1.1 Fix 25 `src.rag_pipeline` imports → `rag_pipeline`
The package is installed as `rag_pipeline` (via `where = ["src"]` in pyproject.toml).
The `src.` prefix is wrong and only works due to sys.path hacks.

**Files to fix:**
```
ablation/corpus_sampler.py          lines 31-32
ablation/experiment.py              lines 30-31
src/rag_pipeline/eda/topics/config.py                          line 10
src/rag_pipeline/eda/topics/pipeline.py                        lines 5-8
src/rag_pipeline/eda/topics/classification/topic_rules.py      lines 13-14
src/rag_pipeline/eda/topics/classification/topic_ner.py        lines 16-17
src/rag_pipeline/eda/topics/core/topic_merge.py                line 7
src/rag_pipeline/eda/topics/core/topic_modeling.py             lines 14-23
src/rag_pipeline/eda/topics/core/topic_subtopics.py            line 24
src/rag_pipeline/eda/topics/core/topic_loader.py               line 6
src/rag_pipeline/eda/topics/core/topic_cluster.py              line 5
src/rag_pipeline/eda/topics/core/topic_assignments.py          line 6
```

**Fix script:**
```bash
find . -path ./.venv -prune -o -name "*.py" -print | xargs grep -l "from src\.rag_pipeline" \
  | grep -v __pycache__ | while read f; do
    sed -i 's/from src\.rag_pipeline/from rag_pipeline/g' "$f"
    sed -i 's/import src\.rag_pipeline/import rag_pipeline/g' "$f"
done
```

**Test:**
```bash
grep -rn "src\.rag_pipeline" --include="*.py" . | grep -v __pycache__ | grep -v .venv
# should return nothing
uv run python -m rag_pipeline.eda.topics.core.topic_modeling --help
uv run python -m ablation run --help
```

---

### 1.2 Standardize logging import
Two valid import paths exist — use `rag_pipeline.logging` everywhere (the canonical one).
`rag_pipeline.core.logging` is a re-export shim — fine to keep but don't mix.

**Files using `rag_pipeline.core.logging` (should use `rag_pipeline.logging`):**
```
src/rag_pipeline/ingestion/benchmark_loader.py
src/rag_pipeline/ingestion/embedding_cache.py
src/rag_pipeline/ingestion/benchmark.py
src/rag_pipeline/ingestion/benchmark_multi_model.py
src/rag_pipeline/ingestion/benchmark_report.py
src/rag_pipeline/ingestion/ingest_qdrant.py
src/rag_pipeline/ingestion/ingest_es.py
src/rag_pipeline/ingestion/benchmark_config.py
src/rag_pipeline/ingestion/ingest_models.py
src/rag_pipeline/cleaning/stratified_test_split.py
src/rag_pipeline/cleaning/load_llm_queries.py
src/rag_pipeline/cleaning/dedup.py
```

**Fix script:**
```bash
find src -name "*.py" | grep -v __pycache__ | xargs \
  sed -i 's/from rag_pipeline\.core\.logging import get_logger/from rag_pipeline.logging import get_logger/g'
```

**Test:**
```bash
grep -rn "rag_pipeline.core.logging" --include="*.py" src/ | grep -v __pycache__
# should return nothing
python3 -c "from rag_pipeline.logging import get_logger; print(get_logger('test'))"
```

---

### 1.3 Remove sys.path.insert hacks
The package is installed — no path manipulation needed.

**Files:**
```
ablation/cli.py              line 23
ablation/corpus_sampler.py   lines 28-29
ablation/experiment.py       lines 26-28
src/rag_pipeline/eda/topics/config.py   line 8
tests/test_cleaning.py       line 5
```

**Fix:** Delete the `sys.path.insert` blocks. The `sys` import can stay if used elsewhere, remove if not.

**Test:**
```bash
uv run python -m ablation run --help
uv run python -m ablation report
uv run python -c "from rag_pipeline.eda.topics.config import TopicsConfig; print('ok')"
```

---

### 1.4 Fix pyproject.toml project name
```toml
# current
name = "llm"

# fix to
name = "rag-pipeline"
```

Also add `ablation` to packages so it's importable without sys.path:
```toml
[tool.setuptools.packages.find]
where = ["src", "."]
include = ["rag_pipeline*", "configs*", "ablation*"]
```

**Test:**
```bash
uv pip install -e . --quiet
python3 -c "import ablation; print('ok')"
```

---

### 1.5 Move entity_patterns.json to configs/
Currently at `src/rag_pipeline/eda/topics/entity_patterns.json` — it's hand-edited config,
not generated code. Should live alongside `retrieval_configs.json`.

```bash
mv src/rag_pipeline/eda/topics/entity_patterns.json configs/entity_patterns.json
```

Then update `Paths.entity_patterns()` in `core/paths.py` to point to `configs/entity_patterns.json`.

**Test:**
```bash
python3 -c "from rag_pipeline.core.paths import Paths; print(Paths.entity_patterns().exists())"
uv run python -m rag_pipeline.eda.topics.core.topic_modeling --help  # should not crash on import
```

---

## Priority 2 — Medium, Structural

### 2.1 Move ablation into src/rag_pipeline/
`ablation/` imports from `rag_pipeline`, is part of the project, but lives outside `src/`.
Move to `src/rag_pipeline/ablation/` and update `__main__.py` entry point.

```
ablation/ → src/rag_pipeline/ablation/
```

Update pyproject.toml entry points if needed. The `uv run python -m ablation` invocation
stays the same since `ablation` would still be a package.

---

### 2.2 Separate test queries from corpus assignments
`topic_assignments_all.json` now contains both corpus docs (1204) and test queries (470)
mixed together. This was added this session for query-side NER enrichment.

**Fix:** Store test query NER tags in a separate file:
```
output/topic_assignments_all.json      ← corpus docs only (restore to 1204 entries)
output/test_query_ner_tags.json        ← {query_id: {ner_category, ner_primary_entity}}
```

Update `benchmark_loader.py` `_build_query_context` to load from `test_query_ner_tags.json`
for query-side enrichment instead of `topic_map`.

---

### 2.3 Add Paths validation
`Paths` silently returns non-existent paths. Add existence checks for critical paths:

```python
@classmethod
def topic_assignments(cls) -> Path:
    p = cls._resolve("topic_assignments")
    if not p.exists():
        raise FileNotFoundError(
            f"Topic assignments not found at {p}. "
            "Run: uv run python -m rag_pipeline.eda.topics.core.topic_modeling"
        )
    return p
```

Apply to: `topic_assignments()`, `entity_patterns()`, `input_file()`.

---

### 2.4 Document the embedding strategy in collection name
Current: `faqs_bge_base_en_v1_5` — doesn't indicate `question+answer` concat strategy.
Add to `models.json`:
```json
"embedding_strategy": "question_answer_concat"
```
And log it clearly at ingest time.

---

## Priority 3 — Ablation as a proper pipeline step

### Current state (ad-hoc)
- Run manually after changes
- Results in `ablation/results/` with no provenance
- No connection to the main benchmark
- No pass/fail criteria

### Target state
Ablation becomes step `p04_ablation` in the pipeline, run after `benchmark`:

```
ingest_qdrant   → ingest corpus
benchmark       → establish baseline
p04_ablation        → run all ablations, compare to baseline, fail if regression > threshold
evaluation      → LLM judge on winners
```

**Changes needed:**

1. **Add `ablation/` to the pipeline runner** (`run_clean_pipeline.py` or a new `run_pipeline.py`)
2. **Add baseline comparison to ablation report** — report should print delta vs stored baseline
   and exit non-zero if any previously-winning config regresses by >2pp
3. **Store baseline in `experiments/baseline.json`** — written by `benchmark`, read by ablation
4. **Add `--ci` flag to ablation** — suppresses interactive output, exits 0/1 for pass/fail
5. **Add provenance to results** — each result file should record git commit, timestamp, corpus size

**Proposed ablation config** (`configs/ablation_config.json`):
```json
{
  "regression_threshold_pp": 2.0,
  "configs_to_test": ["entity_boosted"],
  "model": "BAAI/bge-base-en-v1.5",
  "experiments": [
    {"name": "no_entity",       "flags": ["--null-entity"]},
    {"name": "no_category",     "flags": ["--null-category"]},
    {"name": "no_topics",       "flags": ["--null-topics"]},
    {"name": "skip_ner",        "flags": ["--skip-ner"]},
    {"name": "skip_cluster",    "flags": ["--skip-cluster"]},
    {"name": "skip_rules",      "flags": ["--skip-rules"]},
    {"name": "empty_patterns",  "flags": ["--empty-entity-patterns"]}
  ]
}
```

---

## Priority 4 — Tests

### Current state
`tests/` has 4 files, 2 are debug scripts (`debug_dedup_grouping.py`, `inspect_comparisons.py`),
not actual tests.

### Minimum viable test suite to add:

| Test | What it checks |
|---|---|
| `test_paths.py` | All `Paths.*()` methods return existing files |
| `test_imports.py` | All modules import without error (catches wrong import paths) |
| `test_ablation_patch.py` | `Patch.apply_to_assignments()` correctly nulls fields |
| `test_retrieval_configs.py` | All configs in `retrieval_configs.json` have valid `search_type` |
| `test_topic_assignments.py` | Assignments file has expected schema, no test IDs mixed in |

Run with:
```bash
uv run pytest tests/ -v
```

---

## Fix order recommendation

```
Day 1 (30 min):
  1.1 Fix src. imports      ← one script, immediate
  1.2 Standardize logging   ← one script, immediate  
  1.3 Remove sys.path hacks ← manual, 5 files
  1.4 Fix pyproject name    ← 2 lines
  1.5 Move entity_patterns  ← mv + one Paths change
  → run full test suite to confirm nothing broke

Day 2 (2 hours):
  2.1 Move ablation into src/
  2.2 Separate test queries from corpus assignments
  2.3 Add Paths validation

Day 3 (3 hours):
  3.x Ablation as pipeline step
  4.x Minimum test suite
```
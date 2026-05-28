# Structural Cleanup Handoff
_Updated: 2026-05-28_

---

## Conventions (established this session — follow these everywhere)

### Paths
- **Single source of truth**: all filesystem paths live in `configs/paths.json`, resolved via `Paths` class in `src/rag_pipeline/core/paths.py`
- **Never hardcode paths** in module bodies, dataclass defaults, or CLI defaults. No `Path(__file__).resolve().parent / "results"` outside of `paths.py` itself
- **Never use `ROOT =`** or similar module-level path constants — they were sys.path hacks and are all gone
- Use `Paths._resolve("key")` for paths that may not exist yet; use `Paths._require(Paths._resolve("key"), "hint")` for paths that must exist at call time
- To add a new path: add the key to `configs/paths.json`, add a classmethod to `Paths`, done

**Current Paths methods:**
```
Paths.base()                  → project root (pyproject.toml location)
Paths.raw_dir()               → data/raw
Paths.processed_dir()         → data/processed
Paths.experiments_dir()       → experiments/
Paths.clean_jsonl()           → data/processed/clean.jsonl          [_require]
Paths.test_jsonl()            → data/processed/test.jsonl
Paths.topic_assignments()     → .../output/topic_assignments_all.json [_require]
Paths.retrieval_configs()     → configs/retrieval_configs.json
Paths.reranker_results_dir()  → experiments/reranker_benchmarks/    [_require]
Paths.ablation_results_dir()  → experiments/ablation_results/
Paths.input_file(stage)       → from input_mapping in paths.json
Paths.output_file(stage)      → from output_mapping in paths.json
Paths.topics_dir()            → src/rag_pipeline/eda/topics/
Paths.topics_experiments_dir()→ .../topics/experiments/
Paths.topics_output_dir()     → .../topics/output/
Paths.topics_rules_dir()      → .../topics/rules/
Paths.entity_patterns()       → configs/entity_patterns.json        [_require]
Paths.topics_default_output() → .../experiments/topic_assignments.json [_require]
Paths.stopwords_path()        → .../tfidf_analysis/stopwords/stopwords_pass2.txt
Paths.defaults()              → configs/defaults.json (full dict)
Paths.topic_modeling_defaults()→ defaults.json["topic_modeling"] slice
```

### Logging
- Always import from `rag_pipeline.logging`: `from rag_pipeline.logging import get_logger`
- Never use `rag_pipeline.core.logging` (re-export shim, kept but not used directly)
- Always: `logger = get_logger(__name__)` at module level

### CLI / argparse
- All argparse **parser factories** live in `configs/benchmark_cli.py`
- Each script/module calls the appropriate factory; no `ArgumentParser(...)` construction outside `benchmark_cli.py`
- Defaults for `--model` and `--configs` / `--config` always read from `Paths.defaults()` at factory call time — never hardcoded strings
- Available factories:
  ```
  create_base_parser()            → shared path/qdrant/es/model/tuning/flag args
  create_ingestion_parser()       → base + --model (singular)
  create_benchmark_parser()       → base + --model + --collection + --configs + --query-type
  create_multi_benchmark_parser() → base only
  create_generation_parser()      → base + generation args
  create_topic_modeling_parser()  → standalone topic modeling args
  create_ablation_parser()        → run/compare/report subcommands + patch flags
  ```

### Imports
- Package is installed as `rag_pipeline` (via `where = ["src"]` in pyproject.toml)
- Always `from rag_pipeline.x import y` — never `from src.rag_pipeline.x import y`
- No `sys.path.insert` anywhere — package is installed editable via `uv pip install -e .`
- `ablation` now lives at `src/rag_pipeline/ablation/` — import as `rag_pipeline.ablation`

### Defaults / config
- Runtime defaults (model, config name, qdrant host/port, batch sizes) live in `configs/defaults.json`
- Access via `Paths.defaults()` — returns the full dict; slice as needed
- Never duplicate these values as Python constants in module bodies

---

## What's done

### Day 1 ✓
- Fixed 25 `src.rag_pipeline` imports → `rag_pipeline`
- Standardised logging import → `rag_pipeline.logging` everywhere
- Removed all `sys.path.insert` hacks (5 files) and all orphaned fragments they left behind
- Fixed `pyproject.toml` project name (`llm` → `rag-pipeline`)
- Moved `entity_patterns.json` → `configs/` and updated `paths.json`

### Day 2 ✓
- Moved `ablation/` → `src/rag_pipeline/ablation/`
- Separated test query NER tags from corpus assignments (2.2)
- Added `Paths._require()`, `Paths.defaults()`, `Paths.ablation_results_dir()` (2.3)
- Fixed all syntax errors left by botched sys.path removal (6 files across corpus_sampler, experiment, cli, test_cleaning, topic_ner_diagnostics)

### Day 3 (partial) ✓
- Rewrote `corpus_sampler.py` — all hardcoded paths/defaults replaced with `Paths`
- Rewrote `experiment.py` — `RESULTS_DIR` constant gone, `Experiment.model` and `Experiment.configs` defaults from `Paths.defaults()`
- Rewrote `report.py` — `RESULTS_DIR` constant gone, uses `Paths.ablation_results_dir()`
- Added `create_ablation_parser()` to `configs/benchmark_cli.py`
- Rewrote `cli.py` — now just command dispatch, all arg definitions in `benchmark_cli.py`

---

## Todo

### Priority 1 — Remaining hardcoded paths/constants (do next)

#### 1.1 Fix `run_clean_pipeline.py`
Currently has module-level constants that bypass `Paths` entirely:
```python
PROJECT   = Path(__file__).parent.parent          # → Paths.base()
TOPIC_DIR = PROJECT / 'rag_pipeline/p02_eda/...'  # → Paths.topics_experiments_dir()
BENCH_DIR = PROJECT / 'rag_pipeline/experiments'  # → Paths.reranker_results_dir()
TEST_FILE = PROJECT / 'rag_pipeline/p01_data_...' # → Paths.test_jsonl()
MODELS    = ['BAAI/bge-base-en-v1.5', ...]        # → load from configs/models.json
JUDGE_MODEL  = 'BAAI/bge-base-en-v1.5'            # → Paths.defaults()["production_model"]
JUDGE_CONFIG = 'entity_boosted'                    # → Paths.defaults()["production_config"]
```
Also uses `rag_pipeline.p02_eda._topic_merge` — check if this module path is still valid
after the eda restructure, or if it should be `rag_pipeline.eda.topics.core.topic_merge`.

Also: `_build_payload_indexes()` hardcodes `host='localhost', port=6333`
→ `Paths.defaults()["qdrant"]["host/port"]`

#### 1.2 Audit remaining files for hardcoded paths
Run this to find remaining offenders:
```bash
grep -rn "Path(__file__)" --include="*.py" src/ | grep -v __pycache__ | grep -v "paths.py"
grep -rn "BAAI/bge-base-en-v1.5\|entity_boosted" --include="*.py" src/ | grep -v __pycache__ | grep -v "test_"
```

#### 1.3 `compare.py` — check for hardcoded RESULTS_DIR
Not reviewed this session. Likely has the same `RESULTS_DIR = Path(__file__).resolve().parent / "results"` pattern.
```bash
head -20 src/rag_pipeline/ablation/compare.py
```

---

### Priority 2 — Ablation as a proper pipeline step (Day 3 remainder)

#### 2.1 Add `ablation_config.json` to configs/
```json
{
  "regression_threshold_pp": 2.0,
  "configs_to_test": ["entity_boosted"],
  "experiments": [
    {"name": "no_entity",      "flags": ["--null-entity"]},
    {"name": "no_category",    "flags": ["--null-category"]},
    {"name": "no_topics",      "flags": ["--null-topics"]},
    {"name": "skip_ner",       "flags": ["--skip-ner"]},
    {"name": "skip_cluster",   "flags": ["--skip-cluster"]},
    {"name": "skip_rules",     "flags": ["--skip-rules"]},
    {"name": "empty_patterns", "flags": ["--empty-entity-patterns"]}
  ]
}
```
Add `Paths.ablation_config()` → `configs/ablation_config.json`.

#### 2.2 Baseline comparison in report
- `benchmark` step writes `experiments/baseline.json` after each run
- `ablation report` reads it and prints delta vs baseline
- Exit non-zero if any previously-winning config regresses by > `regression_threshold_pp`

#### 2.3 Add `--ci` flag to ablation CLI
- Suppresses interactive output
- Exits 0 (pass) / 1 (fail) based on regression threshold
- Add to `create_ablation_parser()` in `benchmark_cli.py`

#### 2.4 Add provenance to result files
Each `*_meta.json` should include:
```json
{
  "git_commit": "abc123",
  "timestamp": "...",
  "corpus_size": 1204
}
```
Write this in `Experiment.run()` using `subprocess.run(["git", "rev-parse", "HEAD"])`.

#### 2.5 Connect ablation to pipeline runner
Add step to `run_clean_pipeline.py` after benchmark:
```python
def run_ablation() -> None:
    logger.info("Step 6/7: Ablation (regression check)")
    run([sys.executable, "-m", "rag_pipeline.ablation", "run", "--name", "baseline"], step="ablation")
```

---

### Priority 3 — Tests (when ready)

| Test | What it checks |
|---|---|
| `test_paths.py` | All `Paths.*()` methods return existing files |
| `test_imports.py` | All modules import without error |
| `test_ablation_patch.py` | `Patch.apply_to_assignments()` correctly nulls fields |
| `test_retrieval_configs.py` | All configs in `retrieval_configs.json` have valid `search_type` |
| `test_topic_assignments.py` | Assignments file has expected schema, no test IDs mixed in |

```bash
uv run pytest tests/ -v
```

---

## Suggested order

```
Next session (1–2 hours):
  1.3  Check compare.py
  1.2  Audit grep for remaining hardcoded paths
  1.1  Fix run_clean_pipeline.py
  → bash audit.sh

Following session (2–3 hours):
  2.1  ablation_config.json + Paths.ablation_config()
  2.2  Baseline comparison
  2.3  --ci flag
  2.4  Provenance in result files
  2.5  Connect to pipeline runner
```
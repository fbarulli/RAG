Good output. Here's the analysis and exactly what to fix:

---

## 1.3 — `compare.py` fix (straightforward)

```bash
sed -n '1,30p' src/rag_pipeline/ablation/compare.py
```

We already know line 15 is `RESULTS_DIR = Path(__file__).resolve().parent / "results"`. The fix:

```bash
# Check what _load and the rest of the file look like before editing
cat src/rag_pipeline/ablation/compare.py
```

Then apply:
```python
# Remove this line:
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Replace the _load function's path line with:
from rag_pipeline.core.paths import Paths
# ...
path = Paths.reranker_results_dir() / f"{name}_query_results.jsonl"
```

But confirm the full file first — paste the output of `cat src/rag_pipeline/ablation/compare.py`.

---

## 1.2 — Audit results, ranked by priority

**Fix now** (functional bugs / real hardcoded paths):

| File | Line | Issue |
|---|---|---|
| `compare.py` | 15 | `RESULTS_DIR` constant → `Paths.reranker_results_dir()` |
| `answer_generation/runner.py` | 209 | `'BAAI/bge-base-en-v1.5'` default → `Paths.defaults()["production_model"]` |
| `answer_generation/retriever.py` | 39 | same model default in `__init__` |
| `evaluation/llm_judge.py` | 30 | `DEFAULT_MODEL` constant → `Paths.defaults()` |
| `evaluation/p05_evaluation/_failure_analysis.py` | 12 | hardcoded open() path → `Paths.topic_assignments()` |
| `eda/topics/classification/tfidf_course_analysis.py` | 89 | hardcoded `clean.jsonl` path → `Paths.clean_jsonl()` |
| `eda/topics/config.py` | 6 | `project_root = Path(__file__).parents[4]` → `Paths.base()` |

**Skip for now** (not path bugs):

- `entity_boosted` hits in `evaluation.py`, `retrievers.py`, `runner.py` — those are **search type string literals** used in conditionals, not config defaults. Leave them.
- `BAAI/bge-base-en-v1.5` in docstrings, `--help` examples, `TEST_MODELS` list, `ner_from_config.py`/`entity_pattern_learner.py` dict key lookups — these are either documentation or data-dependent keys, not defaults to centralise.
- `core/gem_client.py` and `core/llm_config.py` — these use `Path(__file__)` to find sibling config files (providers.json etc.), which is the **correct pattern** for self-contained core modules. Leave them.

---

Run these next to get the full content of the files that need editing:

```bash
cat src/rag_pipeline/ablation/compare.py
cat src/rag_pipeline/answer_generation/runner.py | grep -n "BAAI\|production_model\|retrieval_model" | head -20
sed -n '35,45p' src/rag_pipeline/answer_generation/retriever.py
sed -n '25,35p' src/rag_pipeline/evaluation/llm_judge.py
sed -n '8,20p' src/rag_pipeline/evaluation/p05_evaluation/_failure_analysis.py
sed -n '55,95p' src/rag_pipeline/eda/topics/classification/tfidf_course_analysis.py
cat src/rag_pipeline/eda/topics/config.py
```

Paste all of that and I'll write the exact diffs.

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
  
  

Following session (2–3 hours):
  2.1  ablation_config.json + Paths.ablation_config()
  2.2  Baseline comparison
  2.3  --ci flag
  2.4  Provenance in result files
  2.5  Connect to pipeline runner
```
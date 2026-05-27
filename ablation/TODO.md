---

## How to run the pipeline
```bash
# 1. topic modeling (single model)
uv run python -m rag_pipeline.eda.topics.core.topic_modeling \
  --embedding-model "BAAI/bge-base-en-v1.5" \
  --output src/rag_pipeline/eda/topics/experiments/topic_assignments_BAAI_bge_base_en_v1.5.json

# 2. merge (no main() — call directly)
uv run python -c "
from src.rag_pipeline.eda.topics.core.topic_merge import TopicMerger
TopicMerger().merge()
"

# 3. ingest
uv run python -m src.rag_pipeline.ingestion.p00_ingest_qdrant --model "BAAI/bge-base-en-v1.5"

# 4. benchmark
uv run python -m rag_pipeline.ingestion.p03_benchmark \
  --model "BAAI/bge-base-en-v1.5" \
  --configs entity_boosted vector_default

# 5. benchmark filtered by query type
uv run python -m rag_pipeline.ingestion.p03_benchmark \
  --model "BAAI/bge-base-en-v1.5" \
  --configs entity_boosted \
  --query-type original grounded_analyst creative_student
```

---

## Paths — single source of truth
```python
from src.rag_pipeline.core.paths import Paths

Paths.topic_assignments()        # output/topic_assignments_all.json
Paths.topics_experiments_dir()   # src/.../topics/experiments/
Paths.entity_patterns()          # src/.../topics/entity_patterns.json
Paths.input_file("eda")          # data/processed/clean.jsonl
Paths.topic_modeling_defaults()  # configs/defaults.json → topic_modeling section
Paths.stopwords_path()           # experiments/tfidf_analysis/stopwords/stopwords_pass2.txt
Paths.collection_for_model(m)    # derives Qdrant collection name from model name
```

Adding a new path: add to `configs/paths.json` → add `@classmethod` to `core/paths.py` → use everywhere.

---

## Config files
| File | Owns |
|---|---|
| `configs/paths.json` | All filesystem paths |
| `configs/defaults.json` | Runtime defaults — thresholds, batch sizes, hosts |
| `configs/retrieval_configs.json` | Retrieval strategy definitions |
| `configs/benchmark_cli.py` | All argparse factories — never inline |
| `src/rag_pipeline/eda/topics/entity_patterns.json` | NER signals — edit here only, never in Python |

---

## Standards
```python
from src.rag_pipeline.logging import get_logger
logger = get_logger(__name__)
logger.info("Processing %s docs=%d", model, len(docs))  # % formatting, never f-strings

# NEVER in library modules:
import logging; logging.basicConfig(...)   # entrypoint only
import sys; sys.path.insert(0, ...)        # Paths handles root resolution
Path(__file__).parent / "something"        # use Paths instead
defaults.get("key", hardcoded_fallback)    # fail loudly on missing config
```

---

## Ablation module (`ablation/`)
Self-contained, lives outside `src/`. Imports from main pipeline, owns all experiment state.

```bash
# run experiments
uv run python -m ablation run --name baseline
uv run python -m ablation run --name no_entity     --null-entity
uv run python -m ablation run --name no_category   --null-category
uv run python -m ablation run --name no_topics     --null-topics
uv run python -m ablation run --name skip_ner      --skip-ner          # re-runs topic modeling
uv run python -m ablation run --name skip_cluster  --skip-cluster      # re-runs topic modeling
uv run python -m ablation run --name skip_rules    --skip-rules        # re-runs topic modeling
uv run python -m ablation run --name empty_patterns --empty-entity-patterns  # re-runs topic modeling
uv run python -m ablation run --name neither       --null-entity --null-category

# compare two experiments
uv run python -m ablation compare baseline__entity_boosted no_entity__entity_boosted
uv run python -m ablation compare baseline__entity_boosted no_entity__entity_boosted --show losses

# summary table with per-query-type breakdown
uv run python -m ablation report
```

**Patch types:**
| Flag | Requires topic modeling re-run | What it tests |
|---|---|---|
| `--null-entity` | No | remove entity from Qdrant payload |
| `--null-category` | No | remove category from Qdrant payload |
| `--null-topics` | No | remove topic_id from payload |
| `--skip-ner` | No | EntityRuler found nothing (null both) |
| `--skip-cluster` | Yes | disable cluster majority in reclassify |
| `--skip-rules` | Yes | disable keyword rules in reclassify |
| `--empty-entity-patterns` | Yes | wipe entity_patterns.json before NER |

Env var flags wired into `topic_rules.py`:
- `ABLATION_SKIP_CLUSTER=1` — skips cluster majority branch
- `ABLATION_SKIP_RULES=1` — skips keyword rules branch

**experiment.py** always restores original state (assignments + entity_patterns) in a `finally` block — safe to interrupt.

Results live in `ablation/results/` — never committed.

---

## Bugs fixed this session
| File | Fix |
|---|---|
| `topic_modeling.py` | Removed invalid `stopwords=` kwarg from BERTopic |
| `topic_modeling.py` | `prob` scalar: `float(probs[i].max()) if hasattr(probs[i], "max") else float(probs[i])` |
| `topic_modeling.py` | `generate_subtopics` → `build_subtopics` |
| `topic_modeling.py` | Wired `ClassificationRules` as post-pass after `_build_assignments` |
| `topic_rules.py` | Added `ABLATION_SKIP_CLUSTER` / `ABLATION_SKIP_RULES` env var guards |
| `schemas.py` | Added missing `from pathlib import Path` |
| `retrievers.py` | Removed `ner_category` from Qdrant `should` clause (+3.7pp H@1) |
| `benchmark_cli.py` | Added `--query-type` filter to `create_benchmark_parser` |
| `p03_benchmark.py` | `_load_test_set` filters by `query_type` when `--query-type` passed |
| `entity_patterns.json` | ~60 new signals: MCP, Elastic Beanstalk, Prefect, Evidently, Faust, Flink, Pipenv, etc. |

---

## Immediate next steps
1. **Corpus sampler** — find minimum training size preserving distribution and H@1.
   Use `stratified_test_split.py` with `--n 600`, `--n 400`, `--n 200`.
   Note: course names inconsistent between corpus (`machine-learning-zoomcamp`) and test (`ml-zoomcamp`) — fix before splitting.
   Re-run full pipeline at each size. Add as `ablation/corpus_sampler.py`.

2. **Run remaining ablations** — `skip_cluster`, `skip_rules`, `empty_patterns`, `skip_ner`.
   These require topic modeling re-run (~5 min each).

3. **Commit current state** — many files modified, nothing committed this session.
   Key files to commit: `retrievers.py`, `topic_modeling.py`, `topic_rules.py`,
   `schemas.py`, `benchmark_cli.py`, `p03_benchmark.py`, `entity_patterns.json`,
   entire `ablation/` directory.
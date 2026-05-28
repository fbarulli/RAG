# ablation/

Self-contained ablation framework for RAG-a-muffin. Lives outside `src/` — imports from the main pipeline but owns all experiment state.

## Structure
```
ablation/
├── __init__.py
├── __main__.py         # enables: uv run python -m ablation <command>
├── cli.py              # argparse entrypoint
├── experiment.py       # patch → ingest → benchmark → save
├── compare.py          # query-by-query diff between two experiments
├── report.py           # summary table across all experiments
└── results/            # all outputs live here — never committed
    ├── baseline_meta.json
    ├── baseline__entity_boosted_query_results.jsonl
    └── ...
```

## Commands

```bash
# run experiments
uv run python -m ablation run --name baseline
uv run python -m ablation run --name no_category --null-category
uv run python -m ablation run --name no_entity --null-entity
uv run python -m ablation run --name neither --null-category --null-entity

# compare two experiments query-by-query (H@1 by default)
uv run python -m ablation compare baseline__entity_boosted no_category__entity_boosted
uv run python -m ablation compare baseline__entity_boosted no_category__entity_boosted --show losses

# summary table across all completed experiments
uv run python -m ablation report
```

## How it works

Each `run` command:
1. Loads `topic_assignments_all.json` from **git HEAD** — always a clean baseline
2. Applies the requested patch in memory (null entity, null category, or both)
3. Re-ingests into Qdrant
4. Runs `benchmark` for the specified configs
5. Copies per-query JSONL results into `ablation/results/`
6. Saves a metadata JSON with metrics
7. **Restores the original assignments** — always, even on failure

## Patch labels
| Flags | Patch label | What it tests |
|---|---|---|
| (none) | baseline | full pipeline |
| --null-entity | no_entity | category boost only |
| --null-category | no_category | entity boost only |
| --null-entity --null-category | neither | pure vector (should match vector_default) |

## Known results (BAAI/bge-base-en-v1.5, entity_boosted config, ner_category removed from should clause)
| Experiment | H@1 | MRR | Delta vs neither |
|---|---|---|---|
| baseline | 84.9% | 0.9025 | +4.6pp |
| no_category | 88.6% | 0.9231 | +8.3pp ← optimal |
| no_entity | 84.9% | 0.9025 | +4.6pp |
| neither | 80.3% | 0.8719 | — |

**Key finding**: `ner_category` in the Qdrant `should` clause hurts retrieval by 3.7pp when combined with `ner_primary_entity`. Entity alone is the signal. Category was removed from `retrievers.py` line 105-106.

## Dependencies
All imports come from the main pipeline:
```python
from src.rag_pipeline.core.paths import Paths
from src.rag_pipeline.logging import get_logger
```
Ingest and benchmark are called via subprocess to avoid argparse conflicts.
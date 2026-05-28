Good distribution — 335/470 have entity signal, category spread looks realistic.

On avoiding the mistakes — here's what actually went wrong and the systemic fix for each:

**1. Ablating fields that were never wired in**
The category/topic ablation was meaningless because the retriever never read those fields. Before designing any ablation, verify the field participates in the code path end-to-end.
→ *Fix:* Add a pre-flight check to `ablation run` that prints which payload fields each config actually uses, so you know what you're testing before running.

**2. Stale file confusion**
The "100% OTHER, 100% topic -1" panic was caused by reading a patched file left over from a failed ablation restore. Cost ~30 min of investigation.
→ *Fix:* The ablation `finally` block already restores, but add a checksum/doc-count sanity check after restore: `assert len(assignments) == expected_count`.

**3. `sed` for multi-line patches**
Every `sed` attempt on multi-line patterns failed. We switched to Python `str.replace` every time.
→ *Fix:* Never use `sed` for multi-line edits. Always use the Python heredoc pattern from the start.

**4. Signature changes not propagated**
`run_clustering_raw` return value changed (3-tuple → 4-tuple), `process_model` got a new param, `build_subtopics` args changed — each broke a different call site that wasn't updated atomically.
→ *Fix:* When changing a function signature, grep for all call sites first and update them in the same script. Never change a signature and assume there's only one caller.

**5. Test set IDs vs corpus IDs**
Spent time investigating why topic assignments weren't found for test queries — they used generated IDs, not corpus doc IDs. The lookup was by `expected_doc_id`, not query ID.
→ *Fix:* Document the ID semantics explicitly: corpus docs have stable IDs, generated test queries have synthetic IDs. The topic map is keyed by corpus doc ID. Any new benchmark feature touching NER needs to know this.

**6. Never verified the benchmark was actually using NER signal**
We assumed `entity_boosted` was working from day one. It was, but we didn't confirm it until deep into the ablation. The +12pp / -12pp confusion cost significant time.
→ *Fix:* Add a benchmark dry-run mode that prints, for the first 5 queries: query text, expected_doc_id, resolved ner_category, ner_primary_entity, and whether the entity boost fired. Makes the retrieval path transparent before running 453 queries.

The meta-pattern across all of these: **assumptions about what the code does weren't verified before building on top of them.** The right habit is to always trace one example end-to-end before running at scale.

Ready to write the new retrieval configs and retriever?





# RAG-a-muffin Ablation Handoff
_Last updated: 2026-05-27_

---

## Current state

### Ablation results so far (config: `entity_boosted`, 453 queries, all query types)

| Experiment | H@1 | MRR | ΔH@1 | Notes |
|---|---|---|---|---|
| baseline | 76.2% | 0.839 | — | full pipeline |
| no_entity | 80.3% | 0.872 | +4.1pp | removing entity boost helps |
| no_category | 88.6% | 0.923 | +12.4pp | removing category helps significantly |
| no_topics | 88.6% | 0.923 | +12.4pp | same as no_category |
| skip_ner | TBD | — | — | not yet run successfully |
| skip_cluster | TBD | — | — | rerun ablation, in progress |
| skip_rules | TBD | — | — | not yet run |
| empty_patterns | TBD | — | — | not yet run |

**Key finding:** The `entity_boosted` config's `should` clause on `ner_primary_entity` actively hurts
retrieval. 335/453 queries have an entity tag; the boost is firing but misfiring enough to cost ~12pp.
Removing NER enrichment entirely likely equals or beats the enriched baseline.

---

## Bugs fixed this session

| File | Fix |
|---|---|
| `topic_modeling.py` | `subtopic_min_size` not unpacked from defaults in `main()` |
| `topic_modeling.py` | `subtopic_min_size` missing from `process_model()` signature and call site |
| `topic_modeling.py` | `run_clustering_raw` now returns `embeddings` (4-tuple) |
| `topic_modeling.py` | `_apply_subtopics` signature fixed — was passing `topic_model/questions/topics`, now passes `assignments/questions/embeddings/subtopic_threshold/subtopic_min_size` |
| `topic_cluster.py` | `run_clustering_raw` encodes embeddings explicitly and returns them |
| `topic_subtopics.py` | `a.topic` → `a['topic']` (dicts passed, not dataclasses) |
| `ablation/cli.py` | Added 5 missing `add_argument` calls: `--null-topics`, `--skip-ner`, `--empty-entity-patterns`, `--skip-cluster`, `--skip-rules` |

---

## Corpus sampler results (`entity_boosted`, `grounded_analyst` queries)

Performance is good enough at reduced corpus sizes — exact numbers in:
`ablation/results/corpus_sampler/corpus_sampler__entity_boosted__BAAI_bge-base-en-v1.5.json`

Note: `original` query type gives H@1≈fraction (trivial — source doc is in corpus),
so always benchmark corpus size against `grounded_analyst` or `creative_student`.

---

## Outstanding work

### 1. Finish remaining ablations
```bash
# skip_cluster is in progress — if it crashed, retry:
uv run python -m ablation run --name skip_cluster     --skip-cluster        --configs entity_boosted
uv run python -m ablation run --name skip_rules       --skip-rules          --configs entity_boosted
uv run python -m ablation run --name empty_patterns   --empty-entity-patterns --configs entity_boosted
```

### 2. Run ablation report
```bash
uv run python -m ablation report
```

### 3. Re-run ablations against generated queries only
The current ablation results mix all query types. To isolate signal:
```bash
uv run python -m ablation run --name baseline_gen     --configs entity_boosted  # then filter in report
```
Or add `--query-type grounded_analyst creative_student` flag to the ablation benchmark call
(not yet implemented — would need to thread through `experiment.py` → `benchmark` call).

### 4. Investigate entity_boosted config
The `should` clause boosts on `ner_primary_entity` match — but this is hurting, not helping.
Options:
- Tune boost weight (currently implicit Qdrant default)
- Switch to a `must_not` or remove entity from `should` entirely
- Test a config variant that only uses entity for tie-breaking

### 5. Commit current state
Key files modified this session, none committed:
```
src/rag_pipeline/eda/topics/core/topic_modeling.py
src/rag_pipeline/eda/topics/core/topic_cluster.py
src/rag_pipeline/eda/topics/core/topic_subtopics.py
ablation/cli.py
ablation/corpus_sampler.py   ← new file
```

---

## Architecture notes

### Why entity_boosted hurts
- `run_entity_boosted_retrieval` adds a Qdrant `should` clause matching `ner_primary_entity` on the payload
- NER tags on test queries come from `topic_assignments_all.json` keyed by `expected_doc_id`
- 335/453 queries have a non-null entity tag → boost fires
- NER classification errors cause the boost to favor wrong documents
- `no_category` and `no_topics` both score identically (+12.4pp) suggesting the effect is
  not from category/topic directly but from a correlated change in the assignments file

### Topic map lookup chain
```
test.jsonl → expected_doc_id
           → benchmark_loader.load_test_set() maps to expected_id
           → benchmark_config.get_topic_map() → load_topic_assignments()
           → keyed by corpus doc ID in topic_assignments_all.json
           → ner_category, ner_primary_entity passed to retriever
```
453/470 test queries resolve to a topic map entry (17 missing = docs not in assignments).

### Ablation patch flow
```
ablation run → Patch.apply_to_assignments() patches topic_assignments_all.json in-place
             → benchmark reads patched file via Paths.topic_assignments()
             → finally block restores original from git
```
Rerun ablations (skip_cluster, skip_rules, empty_patterns) also re-run topic modeling
before ingesting — takes ~5 min each.

---

## Commands reference

```bash
# Run ablation
uv run python -m ablation run --name <name> [--null-entity] [--null-category] \
  [--null-topics] [--skip-ner] [--skip-cluster] [--skip-rules] \
  [--empty-entity-patterns] --configs entity_boosted

# Compare two experiments
uv run python -m ablation compare baseline__entity_boosted no_entity__entity_boosted
uv run python -m ablation compare baseline__entity_boosted no_entity__entity_boosted --show losses

# Summary table
uv run python -m ablation report

# Corpus size ablation (grounded_analyst queries only)
uv run python -m ablation.corpus_sampler --query-type grounded_analyst
uv run python -m ablation.corpus_sampler --fractions 1.0 0.8 0.6 0.4 0.2 --query-type creative_student
```


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
uv run python -m src.rag_pipeline.ingestion.ingest_qdrant --model "BAAI/bge-base-en-v1.5"

# 4. benchmark
uv run python -m rag_pipeline.ingestion.benchmark \
  --model "BAAI/bge-base-en-v1.5" \
  --configs entity_boosted vector_default

# 5. benchmark filtered by query type
uv run python -m rag_pipeline.ingestion.benchmark \
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
| `benchmark.py` | `_load_test_set` filters by `query_type` when `--query-type` passed |
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
   `schemas.py`, `benchmark_cli.py`, `benchmark.py`, `entity_patterns.json`,
   entire `ablation/` directory.
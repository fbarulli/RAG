```markdown
# RAG-a-muffin — Developer Handoff (updated)

## Benchmark status
| Config | H@1 | MRR | Notes |
|---|---|---|---|
| entity_boosted (README) | 85.9% | 0.9088 | Original run — files not recoverable |
| entity_boosted (current) | 80.3% | 0.8719 | **Entity boost contributing 0pp** |
| vector_default (current) | 80.3% | 0.8719 | Same as boosted — boost is broken |

## Why the boost is broken
Qdrant payloads contain `ner_category` and `ner_primary_entity` correctly:
```
LANGUAGE python / ADMIN course / TOOL git / TOOL vs code ...
```
But `configs/retrieval_configs.json` entity_boosted config has no `should` clause referencing those fields:
```json
"entity_boosted": {
    "name": "Entity Boosted",
    "search_type": "entity_boosted",
    "boost_question": 5.0,
    "boost_text": 5.0
}
```
The boost logic must live in the retrieval runner, not the config. Find it:
```bash
grep -rn "entity_boosted\|ner_category\|ner_primary_entity\|should" \
  src/rag_pipeline/ingestion/ 2>/dev/null | grep -v ".pyc"
```
That is the first thing to fix next session.

---

## Data flow
```
data/processed/clean.jsonl  (1207 docs, fields: id, question, answer, course, section)
  │
  ▼
BERTopic clustering                     topic_cluster.py::TopicCluster.run_clustering_raw()
  → topic_id, topic_probability          probs are per-topic arrays — use .max() or hasattr check
  → keywords (TF-IDF, list of tuples)
  │
  ▼
spaCy EntityRuler                       topic_modeling.py::_tag_ner()
  → ner_category                         TOOL/ERROR/CONCEPT/LANGUAGE/ADMIN/OTHER
  → ner_primary_entity                   e.g. "python", "docker" — only EntityRuler sets this
  │                                      keyword rules do NOT set this — produces garbage if tried
  ▼
_build_assignments()                    topic_modeling.py
  → one dict per doc, all fields merged
  │
  ▼
ClassificationRules.reclassify()        topic_rules.py — post-pass on OTHER docs only
  → cluster majority (≥0.8 confidence)  defers to embedding neighborhood
  → keyword rules (priority order)       ERROR > ADMIN > LANGUAGE > TOOL > CONCEPT
  → fallback OTHER                       genuinely ambiguous — correct, do not force
  signals loaded from entity_patterns.json — never hardcode terms in Python
  NOTE: only ner_category is updated — ner_primary_entity is NOT touched here
  │
  ▼
topic_assignments_BAAI_bge_base_en_v1.5.json   (per-model, in experiments/)
  │
  ▼
TopicMerger().merge()                   topic_merge.py — NO main(), call directly
  → topic_assignments_all.json          (output/ — single source for ingestion)
  │
  ▼
p00_ingest_qdrant.py                    indexes into Qdrant with NER fields as payload
  → collection: faqs_bge_base_en_v1_5  name derived from model at runtime
  │
  ▼
p03_benchmark.py                        453 queries from eval_queries_tiered.jsonl
  → H@1 / H@3 / H@5 / H@10 / MRR      expected_id field links query to correct doc
```

---

## How to run the pipeline
```bash
# 1. topic modeling (single model)
uv run python -m rag_pipeline.eda.topics.core.topic_modeling \
  --embedding-model "BAAI/bge-base-en-v1.5" \
  --output src/rag_pipeline/eda/topics/experiments/topic_assignments_BAAI_bge_base_en_v1.5.json

# 2. merge (must call directly — no main())
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
```

---

## Paths — single source of truth
All paths resolve through `Paths` → `configs/paths.json`. Never use `__file__` relative paths.

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

Adding a new path:
1. Add key to `configs/paths.json`
2. Add `@classmethod` to `src/rag_pipeline/core/paths.py`
3. Use `Paths.my_new_path()` everywhere

---

## Config — single source of truth
```python
# reading defaults
defaults = Paths.topic_modeling_defaults()
min_topic_size = args.min_topic_size or defaults["min_topic_size"]  # CLI overrides config

# never do this
defaults.get("key", hardcoded_fallback)  # fallbacks hide missing config — fail loudly
```

| File | Owns |
|---|---|
| `configs/paths.json` | All filesystem paths |
| `configs/defaults.json` | Runtime defaults — thresholds, batch sizes, hosts |
| `configs/retrieval_configs.json` | Retrieval strategy definitions |
| `configs/benchmark_cli.py` | All argparse parser factories — never inline |
| `src/rag_pipeline/eda/topics/entity_patterns.json` | All NER signals — edit here only |

---

## Standards (never break these)
```python
# logging
from src.rag_pipeline.logging import get_logger
logger = get_logger(__name__)
logger.info("Processing %s docs=%d", model, len(docs))  # % formatting, never f-strings

# NO bare logging in library modules
import logging; logging.basicConfig(...)  # entrypoint only

# NO sys.path manipulation
import sys; sys.path.insert(0, ...)  # Paths handles root resolution

# NO relative __file__ paths
Path(__file__).parent / "something"  # use Paths instead
```

---

## NER classification architecture
```
1. spaCy EntityRuler          sets ner_category + ner_primary_entity (entity text)
2. Cluster majority (≥0.8)    overrides OTHER via embedding neighborhood
3. Keyword rules               ERROR > ADMIN > LANGUAGE > TOOL > CONCEPT
4. Fallback OTHER              ~14 genuinely ambiguous docs — do not force
```

Source: `entity_patterns.json` (flat lists keyed by category)
```json
{ "TOOL": ["docker", "airflow", ...], "ERROR": ["access denied", ...] }
```

`ClassificationRules.load()` reads this. `ClassificationRules.reclassify(category, question, topic_id, all_assignments)` returns `(new_category, source)` where source ∈ `unchanged | cluster | rules | fallback`.

Whole-word matching for signals ≤3 chars (prevents `'r'` matching every word containing r).

---

## Key files touched this session
| File | Change |
|---|---|
| `topic_modeling.py` | Fixed BERTopic stopwords kwarg, prob scalar bug, wrong subtopics import, wired ClassificationRules |
| `topic_cluster.py` | Removed invalid `stopwords=` kwarg from BERTopic init |
| `topic_rules.py` | Added `extract_entity()` method (present but not called — leave it) |
| `schemas.py` | Added missing `from pathlib import Path` |
| `entity_patterns.json` | ~60 new signals: MCP, Elastic Beanstalk, Prefect, Evidently, Faust, Flink, Pipenv, etc. |

---

## Immediate next steps
1. **Find and fix the entity boost** — grep for where `entity_boosted` search_type is handled in the retrieval runner. The Qdrant `should` clause must reference `ner_primary_entity` payload field on inbound queries. This is why 85.9% → 80.3%.
2. **Confirm 85.9% is recoverable** — once boost is wired, re-ingest and benchmark. If still 80.3%, the original run used different entity assignments.
3. **Run ablation experiments** — one variable at a time:
   - Null out `ner_primary_entity` → benchmark (isolates entity contribution)
   - Null out `ner_category` → benchmark (isolates category filter contribution)
   - Disable cluster majority → benchmark
   - Disable keyword rules → benchmark
4. **Add `reclassify_source` to assignment schema** — useful for diagnostics.
5. **Add `main()` to `topic_merge.py`** — currently can't be run as a module.
```
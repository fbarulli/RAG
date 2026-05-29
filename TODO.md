```bash
cat > HANDOFF.md << 'EOF'
# Ablation Handoff — 2026-05-29

## Baseline (current best)
| Config | H@1 | MRR |
|---|---|---|
| entity_boosted | 88.8% | 0.9242 |

### By query type
| query_type | n | H@1 |
|---|---|---|
| chaos_monkey | 147 | 78.9% |
| creative_student | 120 | 90.8% |
| grounded_analyst | 147 | 93.2% |
| original | 49 | 100.0% |

---

## Completed Experiments
| Name | H@1 | MRR | Delta | Finding |
|---|---|---|---|---|
| no_generic_entity | 85.5% | 0.9054 | -3.3pp | Generic entities HELP, don't poison |
| low_conf_null_50 | 88.8% | 0.9242 | 0pp | Topic probability not driving retrieval |
| low_conf_null_40 | 88.8% | 0.9242 | 0pp | Confirmed — 47% docs below 0.5 but no effect |
| no_cluster | 92.4% | 0.9540 | +3.6pp | ← best result, cluster was poisoning chaos_monkey |
| no_cluster_no_rules | 88.8% | 0.9242 | 0pp | Rules + cluster cancel each other exactly |
| rules_first | 91.8% | 0.9496 | +3.0pp | Better than baseline, cluster still adds noise as fallback |

---

## Key Findings

### Cluster majority vote is harmful
`_cluster_majority_category` stamps the majority NER category onto every doc
in a BERTopic cluster at 80% confidence. BERTopic clusters by topical
proximity, not entity type — mixed-entity clusters propagate mislabels.

The env var fix (`ABLATION_SKIP_CLUSTER`) was a silent failure — env vars
passed via subprocess `env=` do not propagate correctly. Fixed by wiring
`--skip-cluster` and `--skip-rules` as explicit CLI flags through
`create_topic_modeling_parser` → `process_model` → `rules.reclassify`.

### Rules-first is the right order
`reclassify` now runs keyword rules before cluster fallback. Cluster only
fires on docs that rules can't classify. Still 0.6pp behind `no_cluster` —
cluster adds noise even as fallback because it votes on `ner_category` only,
not `ner_primary_entity`, so it can't reinforce the signal driving retrieval.

### Encoding mismatch discovered
- `ingest_qdrant.py` encodes Q+A — `f"{d.question} {d.answer}"`
- `ingest_models.py` encodes question-only — `[d.question for d in docs]`
- Ablation uses `ingest_qdrant`, benchmark uses `ingest_models`
- Every result so far compares different vector spaces
- Not yet fixed — encode_mode work is next

---

## Code fixes this session
| File | Fix |
|---|---|
| `ingest_qdrant.py` | Fixed `src.` prefix import; imports from `embedding_cache` |
| `ingest_es.py` | Imports from `embedding_cache` not `ingest_models` |
| `ingest_models.py` | Removed dead `shutil`/`tempfile` imports |
| `core/schemas.py` | `FAQDocument` migrated to Pydantic — validates empty fields |
| `core/paths.py` | Added `embeddings_cache_dir`; fixed `topics_default_output` to `_resolve` |
| `topic_rules.py` | Removed `os.getenv` — explicit `skip_cluster`/`skip_rules` bool params |
| `topic_modeling.py` | Fixed `or` pattern → `is not None`; `subtopic_min_size` from defaults |
| `ablation/experiment.py` | Replaced `_load_from_git` with direct read; CLI flags replace env vars |
| `benchmark_cli.py` | Added `--skip-cluster`, `--skip-rules` to topic modeling parser |
| `configs/defaults.json` | Added `subtopic_min_size: 5` |

---

## What is left

### 1. Encode mode ablation (next)
Add `--encode-mode question|qa` to `benchmark_cli` and `create_ingestion_parser`.
Wire through `BenchmarkConfig`, `ingest_qdrant`, `ingest_models`.
`Paths.collection_for_model` needs `encode_mode` param — Q+A gets `_qa` suffix
to avoid cache/collection collisions.
Cache keys: question-only uses existing short name, Q+A uses `qa_` prefix.

### 2. Multi-category tagging
`FAQDocument` and Qdrant payload currently hold one `ner_category` and one
`ner_primary_entity`. A document like "Why does my Docker container fail?"
is simultaneously TOOL and ERROR — the second signal is discarded.
Schema change: `ner_categories: list[str]`, `ner_entities: list[str]`.
Retriever `should` clause iterates the list instead of a single value.

### 3. Cluster entity voting
If cluster stays as fallback after rules, add `ner_primary_entity` majority
vote alongside the existing category vote. Docs without a keyword rule match
would inherit the dominant entity from cluster peers — giving entity boosting
more coverage on ambiguous docs.

### 4. Folder restructure (deferred, non-urgent)
`src/rag_pipeline/ingestion/` is too flat. Proposed:
```
ingestion/
├── corpus.py          — load_corpus only
├── ner_map.py         — unified NER loading (3 duplicates exist)
├── encoder.py         — encode_mode aware, uses embedding_cache
├── qdrant/
│   ├── collection.py
│   ├── points.py
│   └── ingest.py
└── benchmark/         — all benchmark_*.py files
```

### 5. Corpus sampling
Minimum corpus size investigation with `corpus_sampler.py`.
Fix naming mismatch: `ml-zoomcamp` vs `machine-learning-zoomcamp` in test set
(`_course_name_map` in `benchmark_loader.py` handles this but needs audit).

### 6. Commit
Once encode_mode ablation is done and multi-category tagging is tested,
commit restructured `src/` and updated `ablation/`.

---

## Commands to resume
```bash
# Run remaining ablation experiments
uv run python -m rag_pipeline.ablation run --name no_rules --skip-rules
uv run python -m rag_pipeline.ablation run --name optimized --null-generic-entity --null-low-confidence-topics

# Full report
uv run python -m rag_pipeline.ablation report

# Verify imports still clean after any changes
python -c "
from rag_pipeline.core.schemas import FAQDocument
from rag_pipeline.ingestion.ingest_qdrant import main
from rag_pipeline.ingestion.ingest_es import main
from rag_pipeline.ingestion.embedding_cache import load_cached_embeddings, save_embeddings_cache
print('OK')
"
```
EOF
echo "Done"
```
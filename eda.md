# RAG-a-muffin — Developer Handoff (updated 2026-05-29)

## Benchmark status (all vs vector_default baseline)
| Experiment | entity_boosted H@1 | vector_default H@1 | Delta vs vector |
|---|---|---|---|
| vector_default | — | 80.3% | — |
| no_generic_entity | 85.5% | — | +5.2pp |
| baseline | 88.8% | — | +8.5pp |
| no_noisy_entities | 89.2% | — | +8.9pp |
| no_rules | 89.4% | — | +9.1pp |
| **no_cluster** | **92.7%** | — | **+12.4pp** |

> ⚠️ `no_cluster` gain is from entity extraction fix becoming active on fresh topic modeling rerun,
> NOT from skipping cluster (cluster is a confirmed no-op). True fixed baseline not yet committed —
> run full rerun pipeline below to establish it.

### By query type — no_cluster (current best)
| query_type | n | entity_boosted H@1 | vector_default H@1 | Boost delta |
|---|---|---|---|---|
| chaos_monkey | 147 | 87.1% | 65.3% | +21.8pp |
| creative_student | 120 | 92.5% | 83.3% | +9.2pp |
| grounded_analyst | 147 | 95.9% | 87.1% | +8.8pp |
| original | 49 | 100.0% | 98.0% | +2.0pp |

---

## Critical finding: entity boost works, but entity quality varies

The boost has been firing correctly all along. `evaluation.py:_build_query_context()` looks up
`ner_primary_entity` from `topic_map` keyed by `expected_id` — the corpus document's assignment,
not the query file. The query file's missing NER fields (`eval_queries_tiered.jsonl` predates NER)
are irrelevant — benchmark joins correctly at runtime.

### Entity quality breakdown (current corpus, BAAI/bge-base-en-v1.5)
| Category | Docs | Unique entities | Avg docs/entity | Quality |
|---|---|---|---|---|
| TOOL | 541 | 84 | 6.4 | ✅ specific |
| LANGUAGE | 48 | 7 | 6.9 | ✅ specific |
| CONCEPT | 148 | 62 | 2.4 | ✅ mostly specific |
| ADMIN | 219 | 55 | 4.0 | ⚠️ generic tail |
| ERROR | 227 | 51 | 4.5 | ⚠️ generic tail |

### Top noisy entities (>20 docs, currently in GENERIC_ENTITIES set)
| Entity | Docs | Category | Action |
|---|---|---|---|
| error | 100 | ERROR | ❌ prune |
| homework | 66 | ADMIN | ❌ prune |
| course | 37 | ADMIN | ❌ prune |
| project | 28 | ADMIN | ❌ prune |
| model | 23 | CONCEPT | ❌ prune |
| docker | 78 | TOOL | ✅ keep |
| python | 34 | LANGUAGE | ✅ keep |
| gcp | 46 | TOOL | ✅ keep |
| dbt | 41 | TOOL | ✅ keep |
| aws | 28 | TOOL | ✅ keep |

Specific ERROR terms (`valueerror`, `typeerror`, `importerror`, `attributeerror`) are fine — keep them.

---

## NER classification architecture
```
1. spaCy EntityRuler          sets ner_category + ner_primary_entity (entity text)
2. Cluster majority (≥0.8)    overrides OTHER via embedding neighborhood
3. Keyword rules               ERROR > ADMIN > LANGUAGE > TOOL > CONCEPT
4. Fallback OTHER              ~17 genuinely ambiguous docs — do not force
```

## Pipeline stage verdicts

### ✅ spaCy EntityRuler — working as intended
- Sets `ner_category` + `ner_primary_entity` for 941/1207 docs
- `no_generic_entity` (-3.3pp from baseline) confirms these entities are real signal
- Removing them hurts — they are doing useful work

### ✅ Cluster majority vote — confirmed no-op (not broken, genuinely irrelevant)
- Only 17 docs reach cluster branch (those remaining OTHER after spaCy + rules)
- None have a clear cluster majority → `_cluster_majority_category` returns None for all 17
- Root cause: ≥0.8 threshold too aggressive for this dataset; most clusters are mixed-category
- 0pp delta is a real finding, not a bug

### ⚠️ Keyword rules — mildly hurting (+0.6pp when skipped)
- 249 docs reclassified by rules: CONCEPT:70, TOOL:62, ADMIN:61, ERROR:52, LANGUAGE:4
- Rules assign entities correctly but generic ERROR/ADMIN/CONCEPT terms poison the boost
- Specific TOOL/LANGUAGE rule matches are fine; the problem is generic category signals
- Fix: prune generic terms from `entity_patterns.json` rather than disabling rules entirely

### ✅ Topics/subtopics — confirmed no-op for current retriever
- `no_topics` 0pp delta — topic_id in payload not used by `entity_boosted` retriever
- `run_entity_boosted_retrieval` only uses `ner_primary_entity` and `ner_category` in should clause
- Topics may become useful with a different retrieval strategy (e.g. topic-filtered search)

---

## Bugs fixed this session

### 1. `ner_primary_entity` never populated after reclassification
**File:** `src/rag_pipeline/eda/topics/core/topic_modeling.py`
**Fix:** After `reclassify()` changes a category, call `rules.extract_entity(new_cat, question)`
and write result to `a["ner_primary_entity"]` if currently None. This was the root cause of the
apparent no_cluster gain — fresh topic modeling runs now produce correctly populated entities.

### 2. `classification_source` never written to assignment dict
**File:** `src/rag_pipeline/eda/topics/core/topic_modeling.py`
**Fix:** Added `a["classification_source"] = source` after reclassify call.
All docs were showing `classification_source: None`, making ablation diagnostics impossible.

### 3. `TopicMerger.merge()` globbed all stale per-model files
**File:** `src/rag_pipeline/eda/topics/core/topic_merge.py`
**Fix:** Added `only: Path | None = None` parameter. Ablation experiments were being drowned
out by 5 stale per-model files from May 23, producing identical results to baseline.

### 4. `TopicMerger` had no `__main__` entry point
**File:** `src/rag_pipeline/eda/topics/core/topic_merge.py`
**Fix:** Added `if __name__ == "__main__"` block with `--only` argparse flag.
`experiment.py` previously called it via `-c` string which broke on paths containing `/`.

### 5. `experiment.py` merge call used broken shell quoting
**File:** `src/rag_pipeline/ablation/experiment.py`
**Fix:** Replaced broken `-c "TopicMerger().merge(only=Path(\"...\"))"` with
`-m rag_pipeline.eda.topics.core.topic_merge --only "{out}"`.

---

## Immediate next steps

### 1. Establish true fixed baseline (required before any further ablation)
```bash
uv run python -m rag_pipeline.eda.topics.core.topic_modeling \
  --embedding-model "BAAI/bge-base-en-v1.5" \
  --output "src/rag_pipeline/eda/topics/experiments/topic_assignments_BAAI_bge_base_en_v1.5.json" && \
uv run python -m rag_pipeline.eda.topics.core.topic_merge \
  --only "src/rag_pipeline/eda/topics/experiments/topic_assignments_BAAI_bge_base_en_v1.5.json" && \
uv run python -m rag_pipeline.ingestion.ingest_qdrant --model "BAAI/bge-base-en-v1.5" && \
uv run python -m rag_pipeline.ingestion.benchmark \
  --model "BAAI/bge-base-en-v1.5" --configs entity_boosted vector_default
```

### 2. Run full ablation suite
```bash
# fast experiments only
uv run python -m rag_pipeline.ablation flow --configs entity_boosted vector_default

# include slow reruns
uv run python -m rag_pipeline.ablation flow --configs entity_boosted vector_default --rerun
```

---

## What is left

### 🔴 High priority
- [ ] **Establish true fixed baseline** — run full rerun pipeline above, commit result
- [ ] **Prune noisy entity patterns** — remove `error`, `homework`, `course`, `project`, `model`
  from `entity_patterns.json`. Keep specific ERROR terms (valueerror, typeerror, etc).
  Re-run baseline after pruning to measure gain.
- [ ] **Commit restructured src/ and ablation/**

### 🟡 Medium priority
- [ ] **Asymmetric NER (Question-First, Answer-Fallback)** — extract entity from Question first;
  fall back to Answer if Question is generic. Requires full topic modeling rerun across all models.
- [ ] **Cluster threshold** — 0.8 too aggressive; try 0.6 to see if cluster gains any real signal
- [ ] **Corpus sampling** — minimum corpus size with corpus_sampler.py.
  Fix naming mismatch: `ml-zoomcamp` vs `machine-learning-zoomcamp` in test set first.
- [ ] **Reranker integration** — Vector → Reranker pipeline with best Cross-Encoder

### 🟢 Low priority
- [ ] **`ingest_qdrant` payload-only mode** — add `--payload-only` flag to skip vector re-upload
  for experiments that only change NER payload fields (saves ~30s per experiment)
- [ ] **Pydantic Assignment schema** — silent None fields caused 2 bugs this session.
  Migrate after ablation suite complete; one focused PR with before/after benchmark.
- [ ] **Update RERANKER.md** — document entity boosting vs pure vector findings
- [ ] **`eval_queries_tiered.jsonl` NER stamping** — query file predates NER pipeline.
  Benchmark joins correctly at runtime via topic_map, so this is cosmetic only.

---

## Data flow
```
data/processed/clean.jsonl  (1207 docs)
  │
  ▼
BERTopic clustering                     topic_modeling.py
  → topic_id, topic_probability
  → keywords (TF-IDF)
  │
  ▼
spaCy EntityRuler                       _tag_ner()
  → ner_category + ner_primary_entity   TOOL/ERROR/CONCEPT/LANGUAGE/ADMIN/OTHER
  │                                     941 docs categorised here
  ▼
_build_assignments()                    266 docs stay OTHER
  │
  ▼
ClassificationRules.reclassify()        topic_rules.py — post-pass on OTHER docs only
  → cluster majority (≥0.8)            → 0 docs changed (no clear majority)
  → keyword rules                       → 249 docs changed (mildly hurts retrieval)
  → fallback OTHER                      → 17 docs stay OTHER
  NOTE: extract_entity() called after reclassify to populate ner_primary_entity ← fixed this session
  │
  ▼
topic_assignments_{slug}.json           per-model, experiments/
  │
  ▼
TopicMerger().merge(only=path)          topic_merge.py — always pass --only for ablation runs
  → topic_assignments_all.json          output/ — single source for ingestion
  │
  ▼
ingest_qdrant.py                        NER fields stored as Qdrant payload
  │
  ▼
benchmark.py                            453 queries → H@1 / MRR
  NER for boost looked up from topic_map[expected_id] at runtime — NOT from query file
```

---

## Key files modified this session
| File | Change |
|---|---|
| `topic_modeling.py` | `classification_source` write-back; `extract_entity` after reclassify |
| `topic_merge.py` | `only` param on `merge()`; `__main__` block with `--only` flag |
| `experiment.py` | Fixed merge call to `-m` module; fixed broken shell quoting |
| `cli.py` | Added `flow` command — runs full experiment suite in sequence |
| `benchmark_cli.py` | Added `flow` subparser with `--rerun` flag |
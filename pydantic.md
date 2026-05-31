**RAG-a-muffin Handoff — 2026-05-31 (Session 3)**

---

## Branch
`mlflow-tracking` (continued from Session 2)

## What was done this session

### 1. Fixed ablation pipeline end-to-end
- `BenchmarkConfig` syntax error fixed (unterminated f-string in validator)
- `QueryResult.rank/hit_at_1/hit_at_3/hit_at_5` converted to `computed_field` — were `Optional` defaulting to `None`, never populated
- `_build_query_result` — removed stale manual `rank=rank, hit_at_1=..., hit_at_3=..., hit_at_5=...` args
- `benchmark.py` `_run_config` — was using `model_entry['collection']` (static, wrong) → now uses `Paths.collection_for_model(model_entry['name'], config.encode_mode)`
- `benchmark.py` — added missing `cache_dir` and `model_name` to `evaluate_config` call
- `evaluate_config` signature — made `cache_dir: Path` and `model_name: str` required (no default) to prevent silent degradation
- `benchmark_multi_model.py` — same collection and cache_dir fixes applied
- `models.json` — removed static `collection` and `es_index` fields (were lies, caused bugs)
- `_save_query_results` — replaced manual field-by-field dict with `r.model_dump()` so computed fields (`rank`, `hit_at_*`, `ner_*`) are always serialized
- `benchmark.py` log line referencing `collection` before assignment fixed

### 2. Sparse retrieval wired (infrastructure only)
- `qdrant_retrievers.py` — added `run_sparse_retrieval` using `SparseVector`
- `retrievers.py` — exported `run_sparse_retrieval`
- `evaluation.py` — added `sparse` dispatch in `_run_retrieval`
- `retrieval_configs.json` — added 4 sparse configs
- `ingest_sparse.py` — new standalone script for sparse collection creation from `.npy` files
- **Blocked**: SPLADE needs query-time encoding; no SPLADE package available. Parked.

### 3. entity_patterns.json cleaned
Removed question starters and generic phrases that were being set as `ner_primary_entity`:
- `CONCEPT`: `why do`, `what is`
- `ERROR`: `could not`, `out of memory`, `warning`
- `ADMIN`: `call`

Result: no H@1 change (89.8%) — these were noise but not the gap cause.

### 4. topic_modeling.py fix — rules entity assignment
Changed: when `classification_source == 'rules'` and `ner_primary_entity` is None:
- If `ner_entities` already populated by spaCy → use `existing[0]` as primary
- If `ner_entities` empty → fall back to `rules.extract_entity()` as before

Result: no H@1 change — correct behavior but not the gap cause.

### 5. LLM NER pipeline
- `providers.json` restored from git history (`secret_agent_man` branch, commit `139fa88`)
- `ProviderConfig` — added `rpm: int = 60` and `tpm: int = 100000` fields
- `src/rag_pipeline/eda/topics/core/llm_ner.py` — new module, uses `call_with_fallback` to extract primary technical entity per doc, checkpoints every 50 docs, resumable
- Ran on full corpus (1207 docs): 868/1207 with entity (71.9%), ~400ms/doc via groq

### 6. Wikineural NER (GPU, Colab)
- Ran `Babelscape/wikineural-multilingual-ner` on GPU in Colab
- 996/1207 with entity (82.5%), faster but noisier
- Saved to `experiments/wikineural_ner_assignments.json`

### 7. Merged NER — LLM primary, wikineural fallback
- `experiments/llm_ner_merged.json` — LLM entity where available, wikineural fallback for 206 docs, 133 still None
- Coverage: 1074/1207 (89.0%)

### 8. New ablation patch: `use_llm_ner`
- `Patch.use_llm_ner: bool` — swaps `ner_primary_entity` and `ner_entities` from `llm_ner_merged.json`
- Loads map once before loop (not per-doc)
- CLI flag `--use-llm-ner` wired in `benchmark_cli.py` and `cli.py`

### 9. Benchmark results
| Experiment | H@1 | MRR | chaos | creative | grounded | original |
|---|---|---|---|---|---|---|
| baseline | 89.8% | 0.9377 | 83.0% | 90.8% | 93.2% | 98.0% |
| **llm_ner** | **90.9%** | **0.9441** | 81.6% | 95.0% | 93.9% | 100.0% |

**+1.1% H@1 — new ceiling.**

---

## Failure analysis (llm_ner, 42 failures)
- `chaos_monkey`: 27/147 — adversarial/vague queries, entity often None, pure vector fallback
- `grounded_analyst`: 9/147
- `creative_student`: 6/120

**Pattern**: many failures have correct answer at rank 2, not rank 1. Entity is correct but wrong doc ranked first within entity cluster.

Example:
```
query:    "installing wget on ubuntu macos windows"
entity:   wget (8 docs share this entity)
expected: "wget is not recognized as internal command"  score=0.747
got:      "wget - ERROR: cannot verify certificate (MacOS)"  score=0.751
```

The vector score breaks ties within the entity cluster incorrectly. Cross-encoder reranking (known issue #7) is the logical next step.

---

## Remaining work

### Immediate
- **Cross-encoder reranking** (known issue #7) — infrastructure exists (`cross-encoder/ms-marco-MiniLM-L-6-v2`), never benchmarked against QA collection. Would fix the rank-2 failures within entity clusters.
- **Improve LLM NER coverage** — 133 docs still have no entity. Second-pass with stronger model or different prompt.
- **Validate wikineural fallback quality** — 206 docs use wikineural; some are noisy (`google colab` as entity where LLM said none). LLM judge pass on these.
- **Run full ablation flow with `--use-llm-ner`** — currently only `entity_boosted` tested with LLM NER. Run all configs to confirm interaction effects.

### Deferred (from previous sessions)
- **Sparse retrieval** — SPLADE query encoding blocked, needs `transformers` SPLADE inference path
- **89.8% → 92.2% gap** — origin still unclear; may be resolved now at 90.9% but the original 92.2% baseline at commit `68fd559` is still unbeaten
- **Known issue #2** — noisy reclassified primaries partially addressed but not fully resolved
- **Known issue #4** — `reranker_results_dir` misnamed
- **Known issue #5** — `es_index` hardcoded
- **ColBERT** — needs `colbert-ai` package
- **Generate more test questions** — current 463 queries, stratified by type; more would improve statistical confidence on small deltas

---

## Key files changed this session
- `src/rag_pipeline/ingestion/benchmark_types.py` — computed fields on `QueryResult`
- `src/rag_pipeline/ingestion/benchmark_metrics_data/evaluation.py` — required `cache_dir`/`model_name`, sparse dispatch
- `src/rag_pipeline/ingestion/benchmark_metrics_data/qdrant_retrievers.py` — `run_sparse_retrieval`
- `src/rag_pipeline/ingestion/benchmark_metrics_data/retrievers.py` — export sparse
- `src/rag_pipeline/ingestion/benchmark.py` — collection fix, cache_dir, model_dump JSONL
- `src/rag_pipeline/ingestion/benchmark_multi_model.py` — collection fix
- `src/rag_pipeline/ingestion/ingest_sparse.py` — new
- `src/rag_pipeline/eda/topics/core/topic_modeling.py` — rules entity assignment fix
- `src/rag_pipeline/eda/topics/core/llm_ner.py` — new
- `src/rag_pipeline/core/models/llm.py` — `rpm`/`tpm` fields
- `src/rag_pipeline/core/models/ablation.py` — `use_llm_ner` patch
- `src/rag_pipeline/core/providers.json` — restored
- `configs/entity_patterns.json` — garbage signals removed
- `configs/models.json` — static `collection`/`es_index` removed
- `configs/benchmark_cli.py` — `--use-llm-ner` flag
- `configs/retrieval_configs.json` — 4 sparse configs added

---

## MLflow DB state
- `rag-retrieval`: 60+ runs (6 models × 10 configs)
- `rag-ablation`: baseline, no_entity, no_category, no_topics, no_generic_entity, low_conf_null_50, low_conf_null_40, no_cluster, no_rules, empty_patterns, llm_ner

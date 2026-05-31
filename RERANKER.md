# RAG-a-muffin Handoff — 2026-05-31 (Session 4)

---

## Branch
`mlflow-tracking` (continued from Session 3)

## What was done this session

### 1. Failure analysis module
New file: `src/rag_pipeline/ingestion/benchmark_metrics_data/failure_analysis.py`

- `FailureRecord` dataclass with `cluster_bucket` and `is_rank2` properties
- `triage()` — buckets failures by entity cluster size, returns `actionable_failures` (excludes `chaos_monkey`)
- Formatted terminal output: summary header, mode breakdown with bar charts, cluster table, per-query detail view
- CLI: `uv run python -m rag_pipeline.ingestion.benchmark_metrics_data.failure_analysis <jsonl> <assignments_json> [--detail <bucket>]`

### 2. Failure mode analysis (llm_ner, 42 failures)

| bucket | failures | rank-2 |
|---|---|---|
| no_entity | 15 | 9 |
| 2-5 | 8 | 6 |
| 6-15 | 2 | 2 |
| 16-40 | 2 | 2 |
| 40+ | 15 | 7 |

**Two distinct modes identified:**
- **Mode A — reranking (26 failures):** correct doc retrieved, wrong rank. Cross-encoder fix.
- **Mode B — routing (16 failures):** entity missing or rank > 2.
  - 10 of 15 no_entity failures are `chaos_monkey` — unreliable ground truth, not actionable
  - 5 no_entity `grounded_analyst` failures are real — all rank-2, valid expected_ids

### 3. Cross-encoder viability confirmed
- Tested `ms-marco-MiniLM-L-2-v2` on wget failure case — reranked correctly, 32ms for 2 pairs
- ONNX infrastructure fully wired end-to-end: `ONNXCrossEncoder` → `RerankerRunner` → `evaluate_with_reranker` → `_apply_reranking`
- `hit_answers` confirmed populated by `run_entity_boosted_retrieval` — cross-encoder receives real text, not empty strings
- `_apply_reranking` already skips only `search_type == 'vector'` — entity_boosted goes through it automatically

### 4. Reranker configs added to retrieval_configs.json
```json
"entity_boosted_tinybert":  { ...entity_boosted base..., "reranker": true, "reranker_name": "TinyBERT" }
"entity_boosted_minilm":    { ...entity_boosted base..., "reranker": true, "reranker_name": "MiniLM-L6" }
"entity_boosted_mxbai":     { ...entity_boosted base..., "reranker": true, "reranker_name": "mxbai-xsmall" }
```

### 5. bge-reranker-base removed from rerankers.json
Too large for CPU. Remaining models: TinyBERT, MiniLM-L6, mxbai-xsmall, AnswerDotAI-Trainic.

### 6. --sample-size wired into ablation CLI
- `configs/benchmark_cli.py` — `--sample-size` added to `create_ablation_parser()`
- `src/rag_pipeline/ablation/experiment.py` — `sample_size: int = 0` field on `Experiment`, passed as `--sample-size` in benchmark subprocess command
- `src/rag_pipeline/ablation/cli.py` — `sample_size=args.sample_size` passed to `Experiment()`

### 7. Reranker eager loading + progress log improvement
`src/rag_pipeline/ingestion/benchmark_metrics_data/evaluation.py`:
- Pre-loads reranker via `_get_runner()` before `_run_evaluation_loop` so model download/compile is visible in logs
- Progress log now includes `[reranker=<name>]` when reranking is active

---

## In progress at handoff
Reranker sample benchmark running:
```bash
uv run python -m rag_pipeline.ablation run \
    --name entity_boosted_rerankers_sample \
    --use-llm-ner \
    --configs entity_boosted_tinybert entity_boosted_minilm entity_boosted_mxbai \
    --sample-size 20
```
If still running, check: `experiments/results/ablation/entity_boosted_rerankers_sample_subprocess.log`

---

## Remaining work

### Immediate
- **Read sample results** — compare TinyBERT vs MiniLM-L6 vs mxbai H@1 and latency on the 20-query sample
- **Run full benchmark** on winning reranker with `--use-llm-ner`, no `--sample-size`
- **Warm ONNX cache** before full run if not already done: `uv run python -m rag_pipeline.ingestion.warm_onnx_cache`
- **5 grounded_analyst no_entity failures** — these have valid ground truth and no entity assigned; worth a manual look at the queries to understand if query-time NER could fix them

### Deferred (from previous sessions)
- **Improve LLM NER coverage** — 133 docs still have no entity
- **Validate wikineural fallback quality** — 206 docs use wikineural fallback
- **Run full ablation with `--use-llm-ner`** across all configs
- **Sparse retrieval** — SPLADE query encoding blocked
- **Known issue #2** — noisy reclassified primaries
- **Known issue #4** — `reranker_results_dir` misnamed
- **Known issue #5** — `es_index` hardcoded
- **ColBERT** — needs `colbert-ai` package
- **89.8% → 92.2% gap** — original 92.2% at commit `68fd559` still unbeaten; current ceiling 90.9%

---

## Key files changed this session
- `src/rag_pipeline/ingestion/benchmark_metrics_data/failure_analysis.py` — new
- `src/rag_pipeline/ingestion/benchmark_metrics_data/evaluation.py` — eager reranker load, progress log
- `src/rag_pipeline/ablation/experiment.py` — `sample_size` field, passed to benchmark subprocess
- `src/rag_pipeline/ablation/cli.py` — `sample_size` wired from args to `Experiment`
- `configs/benchmark_cli.py` — `--sample-size` in `create_ablation_parser()`
- `configs/retrieval_configs.json` — 3 reranker configs added
- `configs/rerankers.json` — `bge-reranker-base` removed

---

## Benchmark results so far

| Experiment | H@1 | MRR |
|---|---|---|
| baseline | 89.8% | 0.9377 |
| llm_ner | **90.9%** | **0.9441** |
| entity_boosted_rerankers_sample | pending |


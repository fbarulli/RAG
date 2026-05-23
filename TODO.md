Based on what I've seen, here are the most impactful directions ranked by likely ROI:
High impact, lower effort

Query expansion — before retrieval, expand the query with synonyms or related terms. Your entity boosting is already doing a light version of this with NER, but you could add LLM-generated query variants and take the best hit across all of them (you already have the LLM client)
Reciprocal Rank Fusion over rerankers — instead of picking one reranker, combine all their rankings with RRF. Your disagreement analysis will show if different rerankers are good at different query types — combining them could beat any single one
Score calibration — TinyBERT and MiniLM are outputting scores of 0.88-0.90 on failures, which suggests they're not well-calibrated for your domain. A simple sigmoid or min-max rescaling per reranker could help

Medium impact, medium effort

Fine-tune the best reranker — take your test set, generate hard negatives (the wrong answers the rerankers are currently ranking above the correct one), and fine-tune bge-reranker-base on your domain. This is probably the single highest ceiling move
Chunk-level answer enrichment — your answers are already clean but adding the question text to the indexed document (question + answer concatenated) before embedding could improve bi-encoder recall, which is the ceiling for reranker performance
Two-stage retrieval — retrieve 100 candidates instead of 40 with the bi-encoder, then rerank. Your current setup might be losing the correct doc before reranking even starts on harder queries

# Handoff — RAG-a-muffin Reranker Pipeline

## Current State
- All mypy errors fixed across all files
- Benchmark running (200 queries, 5 models zero-shot)
- Triples cached at `experiments/reranker_training/triples_sample_965.json`
- Training infrastructure fixed but **not yet run** — waiting on benchmark results to pick best base model
- W&B integration not yet built

## Correct Order of Operations
1. ✅ Fix mypy
2. ✅ Generate 965 triples (cached)
3. 🔄 Run 200-query benchmark → pick top 1-2 models by Hit@5/MRR
4. Update `default_model_key` in `configs/rerankers.json` to winner
5. Build W&B integration
6. Fine-tune winner on Colab T4 with W&B tracking
7. Add fine-tuned model to benchmark, run holdout, beat baseline Hit@5=94% / MRR=0.84

---

## Mypy Fixes Applied (all clean)
| File | What was fixed |
|---|---|
| `benchmark_metrics_data/retrievers.py` | `hit_answers=tuple()` on ES/RRF returns, `Optional[str]` on all `course_filter` params, `Filter` covariance via `# type: ignore[arg-type]`, dict-item noise suppressed |
| `benchmark_metrics_data/evaluation.py` | `QueryContext` TypedDict introduced, `_run_retrieval` reordered (bm25 first, then null guard), `Optional[int]` for topic/subtopic, `float(... or 0.0)` for `code_integrity_ref` |
| `onnx_bench.py` | `Optional[QueryResult]` return type, `Optional[str]` for `target_override`, `str()` cast on `course_filter` |
| `onnx_bench_runner.py` | `Optional[Path]` for `output_dir` with hard raise on None |
| `rerankers.py` | Hard raise on missing `model` key |
| `benchmark_config.py` | `_val()` return type `Any` instead of `object`, `Any` added to imports |
| `onnx_model_loader.py` / `onnx_cross_encoder.py` | `# type: ignore[import-untyped]` on stub-less imports |

---

## Bugs Fixed
| File | Bug |
|---|---|
| `configs/rerankers.json` | `course_filter: "all"` → `null` (was filtering to `course == "all"`, returning nothing) |
| `configs/rerankers.json` | `evaluation_strategy` → `eval_strategy` (renamed in sentence-transformers v5) |
| `reranking_evaluator.py` | `CrossEncoderRerankingEvaluator.from_input_examples()` → `CrossEncoderRerankingEvaluator(samples=...)` (API changed in v5) |
| `create_training_triples.py` | Sequential embedding → batch encode all queries upfront (significant CPU speedup) |

---

## Next: Build W&B Integration

### Install
```bash
uv add wandb
```

### What to instrument
1. **`reranking_training.py`** — log loss, eval metrics, hyperparams per run
2. **`onnx_bench_runner.py`** — log Hit@5, MRR, latency per model per run
3. **Sweeps** — search over `num_hard_negatives`, `lr`, `batch_size`, `num_train_epochs`

### Key config to add to `configs/rerankers.json`
```json
"wandb": {
  "project": "rag-a-muffin-reranker",
  "entity": "<your-wandb-username>",
  "enabled": true
}
```

### Hook into training (one line)
In `reranking_training.py`, add to `CrossEncoderTrainingArguments`:
```python
report_to="wandb",
run_name=f"{cfg['default_model_key']}-{sample_size}triples",
```

### Sweep config (after benchmark picks best model)
```yaml
program: src/rag_pipeline/ingestion/reranking/reranking_training.py
method: bayes
metric:
  name: eval/MRR@10
  goal: maximize
parameters:
  num_hard_negatives:
    values: [3, 5, 10]
  learning_rate:
    distribution: log_uniform_values
    min: 1e-5
    max: 5e-5
  per_device_train_batch_size:
    values: [8, 16]
  num_train_epochs:
    values: [2, 3, 5]
```

---

## Next: Fine-tune on Colab T4
Use `notebooks/reranker_training_colab.ipynb`. Ensure:
- Qdrant running and collection exists
- Triples pre-cached at `experiments/reranker_training/triples_sample_965.json` and committed
- `default_model_key` updated to benchmark winner
- W&B API key set in Colab secrets

Add fine-tuned model to `configs/rerankers.json`:
```json
{
  "name": "<winner>-finetuned",
  "model": "experiments/reranker_models/<winner>-finetuned",
  "max_length": 512,
  "reranker": true,
  "quantization": false
}
```

Then run holdout benchmark:
```bash
python -m rag_pipeline.ingestion.onnx_bench_runner --model <winner>-finetuned
```
Target: beat `entity_boosted` Hit@5=94.0% / MRR=0.84.

---

## Remaining Functional Tasks

### Fix hardcoded retrieval hyperparameters
`parse_runtime_hyperparameters` in `onnx_bench.py` returns hardcoded `boost_question=5.0`, `boost_text=5.0`, `rrf_k=60`. Should come from `configs/retrieval_configs.json`.

### OOD corpus warnings
17 test queries reference IDs not in corpus. Investigate `data/processed/test.jsonl` and either filter or handle gracefully in benchmark loop.

### Fix retrieval cache key collision risk
`_build_cache_key` in `onnx_bench.py` hashes collection + model + query IDs but not retrieval config (boost values, filters). Changing hyperparameters won't invalidate cache. Add retrieval config to hash.

---

## Key File Map
| File | Purpose |
|---|---|
| `configs/rerankers.json` | All reranker model configs + training + inference settings |
| `configs/paths.json` | Single source of truth for all filesystem paths |
| `configs/defaults.json` | Qdrant, ES, benchmark, ingestion settings |
| `src/rag_pipeline/ingestion/onnx_bench_runner.py` | CLI entry point for benchmark |
| `src/rag_pipeline/ingestion/onnx_bench.py` | Core benchmark loop, retrieval cache, matrix eval |
| `src/rag_pipeline/ingestion/reranking/reranking_training.py` | Reranker fine-tuning entry point |
| `src/rag_pipeline/ingestion/reranking/reranking_evaluator.py` | CrossEncoderRerankingEvaluator setup |
| `src/rag_pipeline/ingestion/reranking/create_training_triples.py` | Triple generation with batch encoding + entity boosting |
| `src/rag_pipeline/ingestion/reranking/reranking_triples.py` | Triple cache management |
| `src/rag_pipeline/core/paths.py` | `Paths` class — all path resolution |
| `notebooks/reranker_training_colab.ipynb` | Colab GPU training notebook |


# Handoff — RAG-a-muffin Reranker Pipeline

## Current State
- All mypy errors fixed across all files
- Benchmark running (200 queries, 5 models zero-shot)
- Triples cached at `experiments/reranker_training/triples_sample_965.json`
- Training infrastructure fixed but **not yet run** — waiting on benchmark results to pick best base model
- W&B integration not yet built

## Correct Order of Operations
1. ✅ Fix mypy
2. ✅ Generate 965 triples (cached)
3. 🔄 Run 200-query benchmark → pick top 1-2 models by Hit@5/MRR
4. Update `default_model_key` in `configs/rerankers.json` to winner
5. Build W&B integration
6. Fine-tune winner on Colab T4 with W&B tracking
7. Add fine-tuned model to benchmark, run holdout, beat baseline Hit@5=94% / MRR=0.84

---

## Mypy Fixes Applied (all clean)
| File | What was fixed |
|---|---|
| `benchmark_metrics_data/retrievers.py` | `hit_answers=tuple()` on ES/RRF returns, `Optional[str]` on all `course_filter` params, `Filter` covariance via `# type: ignore[arg-type]`, dict-item noise suppressed |
| `benchmark_metrics_data/evaluation.py` | `QueryContext` TypedDict introduced, `_run_retrieval` reordered (bm25 first, then null guard), `Optional[int]` for topic/subtopic, `float(... or 0.0)` for `code_integrity_ref` |
| `onnx_bench.py` | `Optional[QueryResult]` return type, `Optional[str]` for `target_override`, `str()` cast on `course_filter` |
| `onnx_bench_runner.py` | `Optional[Path]` for `output_dir` with hard raise on None |
| `rerankers.py` | Hard raise on missing `model` key |
| `benchmark_config.py` | `_val()` return type `Any` instead of `object`, `Any` added to imports |
| `onnx_model_loader.py` / `onnx_cross_encoder.py` | `# type: ignore[import-untyped]` on stub-less imports |

---

## Bugs Fixed
| File | Bug |
|---|---|
| `configs/rerankers.json` | `course_filter: "all"` → `null` (was filtering to `course == "all"`, returning nothing) |
| `configs/rerankers.json` | `evaluation_strategy` → `eval_strategy` (renamed in sentence-transformers v5) |
| `reranking_evaluator.py` | `CrossEncoderRerankingEvaluator.from_input_examples()` → `CrossEncoderRerankingEvaluator(samples=...)` (API changed in v5) |
| `create_training_triples.py` | Sequential embedding → batch encode all queries upfront (significant CPU speedup) |

---

## Next: Build W&B Integration

### Install
```bash
uv add wandb
```

### What to instrument
1. **`reranking_training.py`** — log loss, eval metrics, hyperparams per run
2. **`onnx_bench_runner.py`** — log Hit@5, MRR, latency per model per run
3. **Sweeps** — search over `num_hard_negatives`, `lr`, `batch_size`, `num_train_epochs`

### Key config to add to `configs/rerankers.json`
```json
"wandb": {
  "project": "rag-a-muffin-reranker",
  "entity": "<your-wandb-username>",
  "enabled": true
}
```

### Hook into training (one line)
In `reranking_training.py`, add to `CrossEncoderTrainingArguments`:
```python
report_to="wandb",
run_name=f"{cfg['default_model_key']}-{sample_size}triples",
```

### Sweep config (after benchmark picks best model)
```yaml
program: src/rag_pipeline/ingestion/reranking/reranking_training.py
method: bayes
metric:
  name: eval/MRR@10
  goal: maximize
parameters:
  num_hard_negatives:
    values: [3, 5, 10]
  learning_rate:
    distribution: log_uniform_values
    min: 1e-5
    max: 5e-5
  per_device_train_batch_size:
    values: [8, 16]
  num_train_epochs:
    values: [2, 3, 5]
```

---

## Next: Fine-tune on Colab T4
Use `notebooks/reranker_training_colab.ipynb`. Ensure:
- Qdrant running and collection exists
- Triples pre-cached at `experiments/reranker_training/triples_sample_965.json` and committed
- `default_model_key` updated to benchmark winner
- W&B API key set in Colab secrets

Add fine-tuned model to `configs/rerankers.json`:
```json
{
  "name": "<winner>-finetuned",
  "model": "experiments/reranker_models/<winner>-finetuned",
  "max_length": 512,
  "reranker": true,
  "quantization": false
}
```

Then run holdout benchmark:
```bash
python -m rag_pipeline.ingestion.onnx_bench_runner --model <winner>-finetuned
```
Target: beat `entity_boosted` Hit@5=94.0% / MRR=0.84.

---

## Remaining Functional Tasks

### Fix hardcoded retrieval hyperparameters
`parse_runtime_hyperparameters` in `onnx_bench.py` returns hardcoded `boost_question=5.0`, `boost_text=5.0`, `rrf_k=60`. Should come from `configs/retrieval_configs.json`.

### OOD corpus warnings
17 test queries reference IDs not in corpus. Investigate `data/processed/test.jsonl` and either filter or handle gracefully in benchmark loop.

### Fix retrieval cache key collision risk
`_build_cache_key` in `onnx_bench.py` hashes collection + model + query IDs but not retrieval config (boost values, filters). Changing hyperparameters won't invalidate cache. Add retrieval config to hash.

---

## Key File Map
| File | Purpose |
|---|---|
| `configs/rerankers.json` | All reranker model configs + training + inference settings |
| `configs/paths.json` | Single source of truth for all filesystem paths |
| `configs/defaults.json` | Qdrant, ES, benchmark, ingestion settings |
| `src/rag_pipeline/ingestion/onnx_bench_runner.py` | CLI entry point for benchmark |
| `src/rag_pipeline/ingestion/onnx_bench.py` | Core benchmark loop, retrieval cache, matrix eval |
| `src/rag_pipeline/ingestion/reranking/reranking_training.py` | Reranker fine-tuning entry point |
| `src/rag_pipeline/ingestion/reranking/reranking_evaluator.py` | CrossEncoderRerankingEvaluator setup |
| `src/rag_pipeline/ingestion/reranking/create_training_triples.py` | Triple generation with batch encoding + entity boosting |
| `src/rag_pipeline/ingestion/reranking/reranking_triples.py` | Triple cache management |
| `src/rag_pipeline/core/paths.py` | `Paths` class — all path resolution |
| `notebooks/reranker_training_colab.ipynb` | Colab GPU training notebook |
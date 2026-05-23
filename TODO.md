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










Area
What’s slowing you down / could break?	Quick win	Longer‑term fix
Config loading	All values are read lazily from a giant JSON each worker.	Cache the parsed JSON (or the already‑instantiated RayTrainingConfig) on the driver and pass the dict to every worker – no extra file‑IO per worker.	Switch to a typed‑config library (Pydantic, Hydra, OmegaConf) → validation + defaults + single source of truth.
Dataset	Whole triples file is read into a Python list once per worker, then every __getitem__ returns a dataclass (extra Python objects).	Load the JSON once on the driver, keep it in a shared‑memory torch.utils.data.Dataset (or use ray.data), then let each worker use a DistributedSampler.	Store triples on disk in a columnar format (Parquet, Arrow) and lazily map‑style read with datasets or torchdata.
Collation / Tokenisation	Tokeniser runs on the CPU for every batch; the whole batch (B × (G + 1)) is padded twice (inside tokenizer + .view).	Use a fast tokenizer (use_fast=True), enable tokenizer.batch_encode_plus(..., return_tensors="pt", padding="longest") once and avoid the .view by pre‑computing G.	Move tokenisation offline (pre‑tokenise triples → torch.save tensors) or use torch.compile‑friendly tokeniser wrappers.
Mask handling	mask is built as a Python‑list‑of‑list → torch.tensor each batch → extra host‑to‑GPU copy.	Pre‑allocate a (max_batch, G) bool tensor inside the collate fn and fill it with torch.bool ops; use torch.stack on a pre‑created mask tensor.	Store masks together with pre‑tokenised tensors, so the collate function is essentially torch.utils.data.DataLoader’s default.
Loss	Two softmaxes (log_softmax on masked scores and softmax on hardness) per step; creates many temporary tensors.	Fuse the two by re‑using log_probs and compute hardness with logits[:,1:] - logits[:,0:1] → torch.softmax directly; optionally use torch.nn.functional.cross_entropy for the base term.	Write a custom CUDA kernel (or use torch.compile) for the whole loss; cache the neg_mask.float() once per batch.
Training loop	view/to(device) inside the inner loop; torch.cuda.amp.autocast is entered per batch (fine) but torch.cuda.amp.GradScaler is recreated on every worker each epoch (overkill).	Keep the tensors on the GPU once they are moved (input_ids = batch["input_ids"]; …); avoid repeated .to(device) by moving the whole batch at once (for k,v in batch.items(): batch[k]=v.to(device)).	Use torch.utils.checkpoint for gradient checkpointing if memory becomes the bottleneck, or switch to DeepSpeed/FSDP for multi‑GPU scaling.
DistributedSampler	shuffle=False and rely on Ray’s internal sampler – it does work, but you lose deterministic epoch‑wise shuffling.	Pass a torch.utils.data.DistributedSampler(dataset, shuffle=True, seed=cfg.seed) and set shuffle=False on the DataLoader.	Use Ray‑Train’s DataPipe API to let Ray handle sharding & shuffling centrally.
Logging / Reporting	ray.train.report is called every log_every_n_steps inside the inner loop, which may trigger a costly RPC on each worker.	Accumulate a small local list of (step, loss) and only report once per log_every_n_steps after the loop, or use trainer.report(metrics, checkpoint=None) which batches RPCs.	Use ray.train.get_reporter() and a separate thread that flushes metrics every N seconds.
Checkpointing	No checkpoints – a single failure forces you to restart from epoch 0.	Add trainer.save_checkpoint every N epochs (or every log_every_n_steps).	Use Ray‑Train’s built‑in Checkpoint + checkpoint_config + resume_from_checkpoint.







# Handoff — Ray Reranker Training Pipeline

## Current State

- Tunnel live: `https://began-verbal-load-low.trycloudflare.com` (re-run cloudflared on VM if session dropped)
- Qdrant reachable from Colab via `QdrantClient(url=..., prefer_grpc=False, https=True, port=443)`
- Collection: `faqs_bge_base_en_v1_5`
- Old checkpoint deleted: `experiments/reranker_models/MiniLM-L6-finetuned/`
- Core dump deleted: `/workspaces/RAG-a-muffin/core` (was 3.9G — onnx_bench OOM crash)
- HuggingFace cache cleared: `~/.cache/huggingface/` (was 1.9G)
- Ray training pipeline written and split into single-responsibility modules

---

## New Files — Where They Go

| File | Destination |
|---|---|
| `reranking_config_ray.py` | `src/rag_pipeline/ingestion/reranking/` |
| `reranking_dataset.py` | `src/rag_pipeline/ingestion/reranking/` |
| `reranking_loss.py` | `src/rag_pipeline/ingestion/reranking/` |
| `reranking_training_ray.py` | `src/rag_pipeline/ingestion/reranking/` |
| `benchmark_cli.py` | `configs/` (replace existing) |

---

## Config Changes Required

### `configs/rerankers.json`
Add the `ray_training` block at the top level alongside `training` and `inference`:

```json
"ray_training": {
    "model_key":              "bge-reranker-base",
    "output_subdir":          "bge-reranker-base-finetuned",
    "max_length":             512,
    "num_labels":             1,
    "max_negatives":          5,
    "dataloader_num_workers": 2,
    "epochs":                 3,
    "batch_size":             16,
    "lr":                     0.00002,
    "weight_decay":           0.01,
    "warmup_ratio":           0.1,
    "grad_clip":              1.0,
    "fp16":                   true,
    "log_every_n_steps":      50,
    "alpha":                  0.5,
    "num_workers":            1,
    "use_gpu":                true
}
```

> `lr` must be `0.00002` not `2e-5` — JSON does not support scientific notation.

### `configs/paths.json`
No changes needed. `Paths.experiments_dir()` already resolves triples and output paths correctly.

---

## Module Responsibilities

### `reranking_config_ray.py`
- `RayTrainingConfig` dataclass
- `from_rerankers_json()` — loads exclusively from JSON, no hardcoded fallbacks; raises `RuntimeError` with the exact missing key if anything is absent
- `apply_cli_overrides(args)` — maps argparse dest names → dataclass fields
- `to_dict()` — flat serialisable dict for Ray `train_loop_config`

### `reranking_dataset.py`
- `QueryGroup` dataclass
- `RerankerDataset` — loads triples JSON, pads short negative lists with the positive (masked in loss)
- `make_collate_fn()` — tokenises groups into `(B, G, L)` tensors with a bool mask

### `reranking_loss.py`
- `AdaptiveListwiseLoss(alpha)`
- Listwise cross-entropy with difference-based hard-negative weighting: `w_i = softmax(s_neg_i - s_pos)`
- NaN-safe: masked log-probs filled with `0.0` before weighted sum
- Logs an error if NaN is detected mid-training

### `reranking_training_ray.py`
- `train_loop_per_worker(config)` — Ray worker loop: model, dataloader, AMP, grad clip, checkpoint save on rank 0
- `main()` — loads config, applies CLI overrides, launches `TorchTrainer`

### `configs/benchmark_cli.py`
- Added `create_reranker_training_parser()` to the public API
- Groups: `model`, `data`, `training`, `loss`, `ray`
- All args default to `None` — JSON config is the source of truth, CLI only overrides

---

## Correct Order of Operations

1. ✅ Commit all four new `.py` files to `src/rag_pipeline/ingestion/reranking/`
2. ✅ Replace `configs/benchmark_cli.py`
3. ✅ Add `ray_training` block to `configs/rerankers.json`
4. ✅ Verify triples exist: `experiments/reranker_training/triples_sample_965.json`
5. 🔲 In Colab — clone repo, install deps, start tunnel, run training
6. 🔲 Download finetuned model, add to `configs/rerankers.json` models list
7. 🔲 Run holdout benchmark — target: beat `entity_boosted` Hit@5=94.0% / MRR=0.84

---

## Colab Setup (fresh session)

```python
# Cell 1 — install
!pip install ray[train] transformers torch qdrant-client -q

# Cell 2 — clone
!git clone https://github.com/<you>/rag-a-muffin.git
%cd rag-a-muffin

# Cell 3 — verify Qdrant tunnel
from qdrant_client import QdrantClient
QDRANT_URL = "https://began-verbal-load-low.trycloudflare.com"
client = QdrantClient(url=QDRANT_URL, prefer_grpc=False, https=True, port=443)
print(client.get_collections())

# Cell 4 — train (all config from rerankers.json)
!python -m rag_pipeline.ingestion.reranking.reranking_training_ray

# Cell 4a — with CLI overrides
!python -m rag_pipeline.ingestion.reranking.reranking_training_ray \
    --epochs 5 --alpha 0.3 --batch-size 32
```

---

## Loss Design Notes

```
L = L_base + alpha * L_penalty

L_base    = -log_softmax(scores)[0]           positive always at index 0
hardness  = s_neg_i - s_pos                   difference, not ratio (stable)
weights   = softmax(hardness)                  harder neg → higher weight
L_penalty = Σ_i  weights_i * -log_softmax(s)[i+1]

Padding   → scores masked -inf before softmax
          → log_probs masked 0.0 before penalty sum  (prevents 0*inf=NaN)
```

**Tuning `alpha`:** start at `0.5`. If training is unstable, lower to `0.3`. If easy negatives dominate, raise to `0.7`. Change in `configs/rerankers.json` only — no code edits needed.

---

## Remaining Tasks from Previous Handoff

| Task | File | Status |
|---|---|---|
| Fix hardcoded retrieval hyperparameters | `onnx_bench.py` | 🔲 |
| OOD corpus warnings (17 queries) | `data/processed/test.jsonl` | 🔲 |
| Fix retrieval cache key collision | `onnx_bench.py` | 🔲 |
| Add finetuned model to benchmark | `configs/rerankers.json` | 🔲 after training |

---

## Key File Map (full pipeline)

| File | Purpose |
|---|---|
| `configs/rerankers.json` | All reranker configs — models, training, ray_training, inference |
| `configs/paths.json` | Single source of truth for all filesystem paths |
| `configs/benchmark_cli.py` | All argparse parser factories |
| `src/.../reranking_config_ray.py` | RayTrainingConfig dataclass |
| `src/.../reranking_dataset.py` | QueryGroup, RerankerDataset, collate_fn |
| `src/.../reranking_loss.py` | AdaptiveListwiseLoss |
| `src/.../reranking_training_ray.py` | Ray training entry point |
| `src/.../reranking_training.py` | Original sentence-transformers trainer (keep for reference) |
| `src/.../create_training_triples.py` | Triple generation |
| `src/.../reranking_triples.py` | Triple cache management |
| `notebooks/reranker_training_colab.ipynb` | Colab notebook (superseded by ray script) |


Aspect
Train model‑by‑model (sequential)	Train all models together (loop over models)
Simplicity	One + one: you only have a single train_loop_per_worker. No need to carry a “model_id” field around.	Requires a small dispatcher that picks a model, loads its tokenizer, runs a forward pass, then repeats for the next model inside the same batch. More code, more chance of bugs.
Resource utilisation	Only one GPU (or one Ray worker) is active at a time → lower hardware utilisation, longer wall‑clock time.	You can run N workers in parallel, each with a different model, and let Ray schedule them on the available GPUs/CPU cores. Wall‑clock time drops roughly by a factor ≈ #workers (subject to I/O limits).
Isolation & reproducibility	Each run gets a clean Python process → no cross‑contamination of optimizer state, random seeds, or CUDA memory fragmentation.	You have to be careful to reset torch.manual_seed, empty the optimizer, and free CUDA memory between model switches.
Ease of hyper‑parameter sweep	You can hand‑off each model to Ray‑Tune as a separate trial → trivial to combine with Bayesian optimisation, early‑stopping, etc.	You would need to embed a nested hyper‑parameter search (model ↦ hyper‑params), which Ray‑Tune already supports but is more cumbersome to set up.
When it makes sense	• Only a handful of candidate models (≤ 3‑4)
• You want to keep the code‑base identical to the single‑model trainer you already have.	• Dozens of candidates (e.g. all HuggingFace models < 1 B parameters)
• You have a multi‑GPU node or a Ray cluster and care about wall‑clock time.

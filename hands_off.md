# RAG-a-muffin Handoff — 2026-06-01 (Session 5, updated)

---

`torch_dtype` is deprecated! Use `dtype` instead!

## Branch
`mlflow-tracking` (continued from Session 4)

## What was done this session

### 1. `--sample-size` parser bug fixed
`configs/benchmark_cli.py` — `--sample-size` was attached to the wrong parser level twice:
- Benchmark parser: moved from bare `parser.add_argument` → `g.add_argument` (behaviour group, line 72)
- Ablation parser: moved from bare `parser.add_argument` → `g.add_argument` (patch flags group, line 207)

### 2. tqdm added to evaluation loop
`src/rag_pipeline/ingestion/benchmark_metrics_data/evaluation.py`:
- Replaced manual `if idx % 10 == 0: logger.info(...)` with `tqdm(test_set, desc=f'Evaluating [{reranker_name}]', unit='q')`
- Shows per-query throughput and ETA live in terminal

### 3. Subprocess output now streams live
`src/rag_pipeline/ablation/experiment.py` — `_run()`:
- Was: `subprocess.run(..., capture_output=True)` — swallowed all output until completion
- Now: `tee -a <log_path>` so stdout/stderr stream live to terminal AND append to log file

### 4. `retrieval_k` reduced for reranker configs
`src/rag_pipeline/ingestion/benchmark_metrics_data/evaluation.py`:
- Was: `retrieval_k=top_k * 4 if use_reranker else top_k` (40 candidates at top_k=10)
- Now: `retrieval_k=top_k if use_reranker else top_k` (10 candidates)
- Rationale: H@5 is 95%+ so correct doc is almost always in top 10

### 5. Reranker pipeline — two critical bugs fixed

**Bug 1 — score_map key mismatch (reranker was doing nothing):**
`src/rag_pipeline/ingestion/benchmark_reranker.py`:
- Was building doc texts as `faq_question + answer` but `score_map` keyed by `answer` only → all candidates scored 0.0, sort stable, no reordering
- Fix: build `doc_texts` list, key score_map by index position
- `sorted_candidates = [c for _, c in sorted(enumerate(retrieved_candidates), key=lambda x: score_map.get(doc_texts[x[0]], 0.0), reverse=True)]`

**Bug 2 — FAQ question missing from reranker input:**
- Cross-encoder only saw answer text, not the richer FAQ question
- `src/rag_pipeline/ingestion/benchmark_types.py` — added `hit_questions: tuple[str, ...] = ()` to `SearchResult`
- `src/rag_pipeline/ingestion/benchmark_metrics_data/qdrant_retrievers.py` — `parse_qdrant_points` now populates `hit_questions`
- `src/rag_pipeline/ingestion/benchmark_metrics_data/evaluation.py` — candidates include `faq_question` key
- `src/rag_pipeline/ingestion/benchmark_reranker.py` — doc text = `faq_question + " " + answer`

### 6. Entity-boosted retrieval fallback added
`src/rag_pipeline/ingestion/benchmark_metrics_data/composite_retrievers.py`:
- `run_entity_category_boosted_retrieval` retries without entity filter if Qdrant returns 0 results
- Reduced reranker skips from ~39 to ~4 per full run

### 7. Quantization wiring (caches cleared, ready to activate)
- `src/rag_pipeline/ingestion/reranker_runner.py` — passes `quantize=model_config.get("quantization", False)` to `ONNXCrossEncoder`
- `src/rag_pipeline/ingestion/onnx_cross_encoder.py` — accepts and forwards `quantize` param
- `src/rag_pipeline/ingestion/onnx_model_loader.py` — runs `ORTQuantizer` with `avx512_vnni` int8 post-export, loads `model_quantized.onnx` if present
- **All three reranker ONNX caches deleted** — TinyBERT, MiniLM-L6, mxbai will all re-export as int8 on next run

### 8. Reranker config validation with Pydantic
`src/rag_pipeline/ingestion/reranker_config.py` — fully rewritten:
- `RerankerModelConfig`, `RerankerTrainingConfig`, `RerankerInferenceConfig`, `RerankerConfig` — all Pydantic models, no hardcoded defaults (all defaults live in `rerankers.json`)
- `load_reranker_config() -> RerankerConfig` — validates on load, raises on missing/wrong-typed fields
- `get_model_config(key) -> RerankerModelConfig` — raises `KeyError` with available keys instead of silent fallback

### 9. Reranker training pipeline
New file: `src/rag_pipeline/ingestion/reranker_training.py`
- `TrainingTriple` Pydantic model — validates triples on load, catches malformed data early
- `build_dataset()` — explodes triples into `(query, positive, negative)` pairs for `BinaryCrossEntropyLoss`; eval split uses full `hard_negatives` list for `CERerankingEvaluator`
- `train()` — `CrossEncoderTrainer` with `CrossEncoderTrainingArguments` (v4 API), all hyperparams from `rerankers.json` training section
- CLI via `configs/benchmark_cli.py` — `create_reranker_training_parser()` added
- Supports multiple models in one run: `--model-keys TinyBERT MiniLM-L6`

---

## In progress at handoff
Full training run on 965 triples:
```bash
uv run python -m rag_pipeline.ingestion.reranker_training \
    --model-keys TinyBERT \
    --triples experiments/reranker_training/triples_sample_965.json \
    --output-dir experiments/reranker_finetuned
```
Output: `experiments/reranker_finetuned/tinybert/final`

Also still pending from earlier in session:
```bash
uv run python -m rag_pipeline.ablation run \
    --name rerankers_v3_full \
    --use-llm-ner \
    --configs entity_boosted_tinybert entity_boosted_minilm
```
Check: `experiments/results/ablation/rerankers_v3_full_subprocess.log`

---

## Benchmark results

| Experiment | H@1 | MRR | notes |
|---|---|---|---|
| baseline | 89.8% | 0.9377 | |
| llm_ner | 90.9% | 0.9441 | |
| rerankers_v3 tinybert (sample 100) | **97.0%** | 0.9833 | score_map + faq_question fix |
| rerankers_v3 minilm (sample 100) | **97.0%** | 0.9833 | score_map + faq_question fix |
| rerankers_v3 mxbai (sample 100) | 95.0% | 0.9733 | score_map + faq_question fix |
| rerankers_v3_full | pending | | TinyBERT + MiniLM full corpus |

### Sample breakdown (rerankers_v3, n=100)
| query_type | tinybert H@1 | minilm H@1 |
|---|---|---|
| chaos_monkey | 93.3% | 96.7% |
| creative_student | 96.7% | 93.3% |
| grounded_analyst | 100.0% | 100.0% |
| original | 100.0% | 100.0% |

---

## Remaining work

### Immediate
- **Read rerankers_v3_full results** — TinyBERT vs MiniLM H@1 and latency on full 463 queries
- **Wire finetuned model into benchmark** — after training completes, add a reranker config entry pointing to `experiments/reranker_finetuned/tinybert/final` and run ablation
- **Run mxbai with quantization** — caches cleared, next mxbai run re-exports as int8; compare latency vs float32 baseline (~4.86s/q)
- **Commit working state** — reranker pipeline correct end-to-end, training pipeline functional

### Next lever — better training data
- Current triples use question title as query — consider generating answer-grounded queries: give LLM 2-3 answers from same topic/entity, ask for new questions from answer body (harder positives, better `chaos_monkey` coverage)
- Generation code lives in `src/rag_pipeline/evaluation/generate_diverse_test_queries.py`

### Deferred (from previous sessions)
- **Improve LLM NER coverage** — 133 docs still have no entity
- **Validate wikineural fallback quality** — 206 docs use wikineural fallback
- **Run full ablation with `--use-llm-ner`** across all configs
- **Sparse retrieval** — SPLADE query encoding blocked
- **Known issue #2** — noisy reclassified primaries
- **Known issue #4** — `reranker_results_dir` misnamed
- **Known issue #5** — `es_index` hardcoded
- **ColBERT** — needs `colbert-ai` package

---

## Key files changed this session
- `configs/benchmark_cli.py` — `--sample-size` moved to correct groups, `create_reranker_training_parser()` added
- `src/rag_pipeline/ablation/experiment.py` — `_run()` uses `tee` for live streaming
- `src/rag_pipeline/ingestion/benchmark_metrics_data/evaluation.py` — tqdm, `hit_questions`, `retrieval_k` fix
- `src/rag_pipeline/ingestion/benchmark_metrics_data/composite_retrievers.py` — entity filter fallback
- `src/rag_pipeline/ingestion/benchmark_metrics_data/qdrant_retrievers.py` — `hit_questions` in `parse_qdrant_points`
- `src/rag_pipeline/ingestion/benchmark_types.py` — `hit_questions` field on `SearchResult`
- `src/rag_pipeline/ingestion/benchmark_reranker.py` — score_map key fix, `faq_question + answer` doc text
- `src/rag_pipeline/ingestion/benchmark.py` — `top_k` in config log line
- `src/rag_pipeline/ingestion/reranker_runner.py` — `quantize` wired through
- `src/rag_pipeline/ingestion/onnx_cross_encoder.py` — `quantize` param forwarded
- `src/rag_pipeline/ingestion/onnx_model_loader.py` — int8 quantization, loads `model_quantized.onnx`
- `src/rag_pipeline/ingestion/reranker_config.py` — fully rewritten with Pydantic models, no silent fallbacks
- `src/rag_pipeline/ingestion/reranker_training.py` — new, full training pipeline
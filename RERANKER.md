# RAG-a-muffin Handoff — 2026-05-31 (Session 5)

---

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
- Rationale: H@5 is 95%+ so correct doc is almost always in top 10; extra candidates just slow reranking

### 5. `top_k` added to config log line
`src/rag_pipeline/ingestion/benchmark.py`:
- `logger.info(f'Config dict: {cfg} | top_k={config.top_k}')`

### 6. `hit_questions` added to reranker pipeline — **critical fix**
Previously the cross-encoder was scoring `(query, '')` — blank documents — because `hit_answers` was populated but the score_map key mismatch meant all candidates got score 0.0 and sort was stable (no reordering). Two bugs fixed:

**Bug 1 — blank documents:** `evaluate_with_reranker` was building doc texts as `faq_question + answer` but `score_map` was keyed by `answer` only. Fixed by keying score_map on the full doc text via index:
- `src/rag_pipeline/ingestion/benchmark_reranker.py`:
  - Build `doc_texts` list before scoring
  - `score_map = {doc: score for doc, score in doc_score_pairs}`
  - `sorted_candidates = [c for _, c in sorted(enumerate(retrieved_candidates), key=lambda x: score_map.get(doc_texts[x[0]], 0.0), reverse=True)]`

**Bug 2 — FAQ question missing:** Cross-encoder only saw the answer text, not the richer FAQ question. Fixed by:
- `src/rag_pipeline/ingestion/benchmark_types.py` — added `hit_questions: tuple[str, ...] = ()` to `SearchResult`
- `src/rag_pipeline/ingestion/benchmark_metrics_data/qdrant_retrievers.py` — `parse_qdrant_points` now populates `hit_questions=tuple(p.payload.get('question', '') for p in points)`
- `src/rag_pipeline/ingestion/benchmark_metrics_data/evaluation.py` — candidates now include `faq_question` key
- `src/rag_pipeline/ingestion/benchmark_reranker.py` — doc text = `faq_question + " " + answer`

### 7. Entity-boosted retrieval fallback added
`src/rag_pipeline/ingestion/benchmark_metrics_data/composite_retrievers.py`:
- `run_entity_category_boosted_retrieval` now retries without entity filter if Qdrant returns 0 results
- Was causing ~39 reranker skips per full run; now down to ~4

### 8. Quantization wiring (ready but cache not yet rebuilt)
- `src/rag_pipeline/ingestion/reranker_runner.py` — passes `quantize=model_config.get("quantization", False)` to `ONNXCrossEncoder`
- `src/rag_pipeline/ingestion/onnx_cross_encoder.py` — accepts and forwards `quantize` param to `ONNXModelLoader`
- `src/rag_pipeline/ingestion/onnx_model_loader.py` — accepts `quantize`, runs `ORTQuantizer` with `avx512_vnni` int8 config post-export, loads `model_quantized.onnx` if present
- mxbai cache deleted: `experiments/onnx_cache/ee9a6fdcfcbe791ab957991415a11469_mixedbread-ai_mxbai-rerank-xsmall-v1/`
- **Next run with mxbai will re-export and quantize automatically**

---

## In progress at handoff
Full corpus run on TinyBERT and MiniLM (the two sample leaders):
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
| rerankers_v3 tinybert (sample 100) | **97.0%** | 0.9833 | score_map fix applied |
| rerankers_v3 minilm (sample 100) | **97.0%** | 0.9833 | score_map fix applied |
| rerankers_v3 mxbai (sample 100) | 95.0% | 0.9733 | score_map fix applied |
| rerankers_v3_full | pending | | |

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
- **Run mxbai with quantization** — cache was deleted, next run will re-export as int8; compare latency vs float32 (was ~4.86s/q)
- **Commit working state** — reranker pipeline is now correct end-to-end

### Next lever — finetuning
- Training data ready at `experiments/reranker_training/triples_sample_965.json`
- Finetune bi-encoder (bge-base) on domain triples → better Qdrant recall
- Finetune cross-encoder (mxbai or TinyBERT) on domain triples → better reranking precision
- Consider generating answer-grounded queries: give LLM 2-3 answers from same topic/entity, ask for new questions from answer body (harder positives than current question-title-based generation)

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
- `configs/benchmark_cli.py` — `--sample-size` moved to correct parser groups
- `src/rag_pipeline/ablation/experiment.py` — `_run()` now uses `tee` for live streaming
- `src/rag_pipeline/ingestion/benchmark_metrics_data/evaluation.py` — tqdm, `hit_questions`, fallback retrieval
- `src/rag_pipeline/ingestion/benchmark_metrics_data/composite_retrievers.py` — entity filter fallback
- `src/rag_pipeline/ingestion/benchmark_metrics_data/qdrant_retrievers.py` — `hit_questions` in `parse_qdrant_points`
- `src/rag_pipeline/ingestion/benchmark_types.py` — `hit_questions` field on `SearchResult`
- `src/rag_pipeline/ingestion/benchmark_reranker.py` — score_map key fix, `faq_question + answer` doc text
- `src/rag_pipeline/ingestion/benchmark.py` — `top_k` in config log line
- `src/rag_pipeline/ingestion/reranker_runner.py` — `quantize` wired through
- `src/rag_pipeline/ingestion/onnx_cross_encoder.py` — `quantize` param forwarded
- `src/rag_pipeline/ingestion/onnx_model_loader.py` — int8 quantization via `ORTQuantizer`, loads `model_quantized.onnx`
# Reranker Benchmarking & Training

## Overview

This document covers the reranker evaluation pipeline for RAG-a-muffin — why we built it, how it works, what the data looks like, and what the next step (fine-tuning) looks like.

The goal is to find or train a cross-encoder reranker that improves over the baseline `entity_boosted` retrieval configuration, which currently achieves **Hit@5=94.0% / MRR=0.84** on 453 queries using `BAAI/bge-base-en-v1.5`.

---

## Baseline Retrieval Performance

Before introducing rerankers, the bi-encoder retrieval pipeline was benchmarked across several configurations:

| Config | Hit@5 | MRR | Notes |
|---|---|---|---|
| `entity_boosted` | 94.0% | 0.84 | **Production baseline** |
| `hybrid_rrf` | 92.7% | 0.81 | BM25 + vector via RRF |
| `vector_default` | 90.5% | 0.79 | Pure vector, no boosting |
| `bm25_balanced` | 87.9% | 0.76 | BM25 with balanced field weights |
| `bm25_default` | 77.0% | 0.68 | BM25 question-only |

The `entity_boosted` config uses Qdrant `should` clauses to soft-boost results that match the NER entity and category extracted from the query. This is the ceiling a reranker needs to beat to justify added latency.

---

## Why Rerank?

Bi-encoders compress query and document into independent vectors — fast, but lossy. A cross-encoder sees the full `(query, document)` pair at once, allowing fine-grained token-level interaction. For a FAQ domain where the correct answer is often semantically close to several wrong ones, this interaction matters.

The hypothesis: a well-trained cross-encoder can push the correct document to rank 1 even when the bi-encoder retrieves it at rank 3–5.

---

## Data

| Split | File | Size | Purpose |
|---|---|---|---|
| Corpus | `data/processed/clean.jsonl` | 1,204 docs | Source of truth — ingested into Qdrant |
| Train | `data/processed/train.jsonl` | 965 queries | Fine-tuning reranker |
| Test (holdout) | `data/processed/test.jsonl` | 470 queries | Final evaluation only |

### Train set schema
```json
{
  "id": "9f9a1b9e4f",
  "question": "Terraform: Teardown of BigQuery Dataset",
  "answer": "When running terraform destroy...",
  "course": "data-engineering-zoomcamp",
  "section": "module-1-terraform"
}
```

The train set contains `(question, answer)` pairs from 4 courses:
- `data-engineering-zoomcamp`
- `machine-learning-zoomcamp`
- `mlops-zoomcamp`
- `llm-zoomcamp`

Each `id` maps directly to an `es_id` in Qdrant, so the correct document can be retrieved for any training query.

### Test set schema
```json
{
  "id": "e8df9f0d12_orig",
  "question": "Docker: When trying to run a streamlit app...",
  "expected_doc_id": "e8df9f0d12",
  "course": "llm-zoomcamp",
  "answer": "...",
  "query_type": "original",
  "source": "llm_generated"
}
```

The test set has an explicit `expected_doc_id` — the ground truth document the retriever should rank first.

---

## Off-the-shelf Reranker Benchmark

Before fine-tuning, five pre-trained ONNX cross-encoders were evaluated to establish a baseline and identify which architecture is most promising for fine-tuning:

| Name | Model | Quantized |
|---|---|---|
| TinyBERT | `cross-encoder/ms-marco-TinyBERT-L-2-v2` | Yes |
| MiniLM-L6 | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Yes |
| mxbai-xsmall | `mixedbread-ai/mxbai-rerank-xsmall-v1` | Yes |
| AnswerDotAI-Trainic | `answerdotai/answerai-colbert-small-v1` | No |
| bge-reranker-base | `BAAI/bge-reranker-base` | Yes |

### How the benchmark works

1. For each query in the test set, the bi-encoder retrieves 40 candidate documents from Qdrant using `entity_boosted` retrieval
2. Each reranker scores all 40 `(query, answer)` pairs
3. Candidates are re-sorted by reranker score
4. Metrics are computed against `expected_doc_id`

### Run it

```bash
# Full test set
python -m rag_pipeline.ingestion._onnx_bench_runner

# Quick sample (N queries)
python -m rag_pipeline.ingestion._onnx_bench_runner --sample-size 20

# Single reranker
python -m rag_pipeline.ingestion._onnx_bench_runner --model bge-reranker-base
```

Results are written to `production_pipeline/experiments/reranker_benchmarks/`:
```
reranker_benchmarks/
├── benchmark_results.json              # per-model MetricSummary
├── benchmark_summary.txt               # human-readable table
├── reranker_benchmark_performance.json # ranked winner view
└── failure_analysis/
    ├── per_query_failures.json         # every missed query per reranker
    ├── per_query_successes.json        # every hit with rank found
    ├── topic_failure_rates.json        # failure rate grouped by topic
    ├── course_contamination.json       # top-1 from wrong course
    ├── hard_queries.json               # failed by ALL rerankers
    ├── reranker_disagreements.json     # rerankers disagree on rank-1
    └── score_distributions.json       # avg top-1 score on hits vs misses
```

### Expected result

Off-the-shelf MS-MARCO models perform poorly on this domain without fine-tuning. The benchmark is designed to confirm this and identify which model has the best architecture for domain adaptation — not to find a production-ready model.

The key diagnostic is `hard_queries.json` — queries that every reranker fails on point to either retrieval gaps (bi-encoder not returning the correct doc in top-40) or genuinely ambiguous queries.

---

## Fine-tuning Plan (Next Step)

The correct training loop is:

### 1. Generate training triplets

For each `(question, answer)` pair in `train.jsonl`:
- **Positive**: the correct answer (`answer` field, `id` matches `es_id` in Qdrant)
- **Hard negatives**: top-K documents retrieved by the bi-encoder that are NOT the correct answer

Hard negatives are more effective than random negatives because they force the reranker to learn subtle distinctions — the same kind of distinctions it will face at inference time.

### 2. Fine-tune

Target model: `BAAI/bge-reranker-base` (best balance of size and quality from the benchmark).

Training objective: cross-entropy over `(query, positive, [negatives])` using `sentence-transformers` `CrossEncoderTrainer`.

```python
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

model = CrossEncoder("BAAI/bge-reranker-base")
# train with (query, positive, hard_negatives) triplets from train.jsonl
```

### 3. Evaluate on holdout

```bash
python -m rag_pipeline.ingestion._onnx_bench_runner --model bge-reranker-base-finetuned
```

Compare against:
- Baseline `entity_boosted`: Hit@5=94.0%, MRR=0.84
- Off-the-shelf `bge-reranker-base` benchmark score

### Success criterion

Fine-tuned reranker must beat `entity_boosted` Hit@5 and MRR on the holdout test set to justify the added inference latency.

---

## Architecture Notes

### Candidate payload

Each candidate passed to the reranker contains:
```json
{
  "es_id": "e8df9f0d12",
  "question": "Docker: When trying to run a streamlit app...",
  "answer": "To resolve this issue: 1. Ensure you have created a Dockerfile...",
  "payload": {
    "course": "llm-zoomcamp",
    "section": "module-6",
    "ner_category": "TOOL",
    "ner_primary_entity": "docker",
    "topic": 19
  }
}
```

The reranker scores `(query_text, answer)` pairs — only the answer text, not the question or metadata. Future work could explore concatenating `question + answer` as the document representation.

### ONNX runtime

All rerankers are compiled to ONNX and run on CPU via `onnxruntime`. Quantization is applied where supported to reduce memory and improve throughput. Models are cached in `production_pipeline/experiments/onnx_cache/`.

### Retrieval ceiling

The reranker can only promote documents that the bi-encoder retrieves. With 40 candidates per query, the bi-encoder recall ceiling is the hard upper bound on reranker performance. If the correct document is not in the top-40, no reranker can fix it. The `hard_queries.json` failure analysis helps distinguish retrieval failures from reranking failures.

# Dataset Overview

## Training Data

### `data/processed/train.jsonl`
- **Size**: 965 items
- **Purpose**: Source for reranker training triples
- **Fields**: `id`, `question`, `answer`, `course`, `section`
- **Notes**: Primary training source. 963/965 items overlap with `clean.jsonl`.

### `data/processed/clean.jsonl`
- **Size**: 1,204 items
- **Purpose**: Full cleaned corpus — used for Qdrant ingestion and hard negative retrieval
- **Fields**: `id`, `question`, `answer`, `course`, `section`
- **Notes**: Superset of `train.jsonl` with 241 additional documents.

---

## Evaluation Data

### `data/processed/test.jsonl`
- **Size**: 470 items
- **Purpose**: Holdout benchmark set — never used for training
- **Fields**: `id`, `question`, `answer`, `course`, `expected_id`, `query_type`, `section`

### `data/processed/eval_queries_tiered.jsonl`
- **Size**: 2,058 items
- **Purpose**: Evaluation queries stratified by difficulty/type
- **Fields**: `id`, `question`, `answer`, `course`, `expected_id`, `query_type`, `section`
- **Notes**: `answer` field contains question paraphrases, not answer text — not suitable for triple generation.

---

## Generated Triples

All files in `experiments/reranker_training/`.

| File | Size | Notes |
|------|------|-------|
| `triples_sample_10.json` | 10 | Early smoke test |
| `triples_sample_50.json` | 50 | Early smoke test |
| `triples_sample_100.json` | 100 | Early smoke test |
| `triples_sample_200.json` | 200 | Convergence check |
| `triples_sample_200_stratified.json` | 200 | Stratified by topic/NER |
| `triples_sample_965.json` | 965 | Full training set — current ceiling |

**Triple format**:
```json
{
  "query": "...",
  "positive": "...",
  "hard_negatives": ["...", "...", "..."],
  "doc_id": "...",
  "course": "...",
  "topic": "...",
  "ner_category": "...",
  "ner_primary_entity": "..."
}
```

---

## Benchmark Results

### `experiments/reranker_benchmarks/entity_boosted_query_results.jsonl`
- **Purpose**: Pre-computed retrieval results from the `entity_boosted` config — used as input for finetuned reranker benchmarking
- **Baseline**: Hit@5=94.0%, MRR=0.8395 (`BAAI/bge-base-en-v1.5` + entity boosting)

---

## Vector Store

- **Qdrant collection**: `faqs_bge_base_en_v1_5`
- **Embedding model**: `BAAI/bge-base-en-v1.5`
- **Documents**: 1,204 (from `clean.jsonl`)
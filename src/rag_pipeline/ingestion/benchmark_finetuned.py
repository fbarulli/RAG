"""
benchmark_finetuned.py
======================
Benchmark a finetuned cross-encoder checkpoint (safetensors, not ONNX)
against the holdout test set and print Hit@5 / MRR vs entity_boosted baseline.

Usage:
    python -m rag_pipeline.ingestion.benchmark_finetuned \
        --model-path /path/to/bge-reranker-base-finetuned-test \
        --test-set   /path/to/test_set.jsonl \
        --clean      /path/to/clean.jsonl \
        --results    /path/to/existing_query_results.jsonl
"""
import argparse
import json
import logging
import time
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .benchmark_types import QueryResult
from .benchmark_metrics_data.aggregation import aggregate_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASELINE_HIT5 = 0.940
BASELINE_MRR  = 0.84


class FinetuneReranker:
    """Thin CPU wrapper around a saved cross-encoder checkpoint."""

    def __init__(self, model_path: str, max_length: int = 512, batch_size: int = 32):
        self.max_length = max_length
        self.batch_size = batch_size
        logger.info("Loading finetuned reranker from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    @torch.no_grad()
    def rerank(self, query: str, doc_ids: List[str], doc_texts: List[str]) -> List[str]:
        """Return doc_ids sorted by descending relevance score."""
        pairs = [(query, text) for text in doc_texts]
        all_scores = []

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            logits = self.model(**enc).logits.squeeze(-1)
            all_scores.extend(logits.tolist())

        ranked = sorted(zip(doc_ids, all_scores), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranked]


def rerank_query_results(
    results: List[QueryResult],
    reranker: FinetuneReranker,
    id_to_text: dict,
    top_k: int = 5,
) -> List[QueryResult]:
    """Apply reranker to each QueryResult, return new list with reranked hit_ids."""
    reranked = []
    for r in results:
        doc_texts = [id_to_text.get(id_, "") for id_ in r.hit_ids]
        start = time.perf_counter()
        new_hit_ids = reranker.rerank(r.query_text, list(r.hit_ids), doc_texts)
        latency_ms = (time.perf_counter() - start) * 1000

        reranked.append(QueryResult(
            query_id=r.query_id,
            query_text=r.query_text,
            expected_id=r.expected_id,
            course=r.course,
            topic=r.topic,
            subtopic=r.subtopic,
            query_type=r.query_type,
            hit_ids=tuple(new_hit_ids[:top_k]),
            hit_scores=tuple(0.0 for _ in new_hit_ids[:top_k]),
            latency_ms=r.latency_ms,
            reranker_latency_ms=latency_ms,
            code_integrity_ref=r.code_integrity_ref,
            code_integrity_retrieved=r.code_integrity_retrieved,
        ))
    return reranked


def load_query_results(path: Path) -> List[QueryResult]:
    """Load pre-computed QueryResult objects from a JSONL file."""
    results = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            results.append(QueryResult(
                query_id=d["query_id"],
                query_text=d["query_text"],
                expected_id=d["expected_id"],
                course=d["course"],
                topic=d.get("topic"),
                subtopic=d.get("subtopic"),
                query_type=d.get("query_type", "unknown"),
                hit_ids=tuple(d["hit_ids"]),
                hit_scores=tuple(d["hit_scores"]),
                latency_ms=d["latency_ms"],
                reranker_latency_ms=d.get("reranker_latency_ms", 0.0),
                code_integrity_ref=d.get("code_integrity_ref", 1.0),
                code_integrity_retrieved=d.get("code_integrity_retrieved"),
            ))
    logger.info("Loaded %d QueryResults from %s", len(results), path)
    return results


def load_id_to_text(clean_path: Path) -> dict:
    """Build {es_id: answer} map from clean.jsonl."""
    mapping = {}
    with clean_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            mapping[doc["id"]] = doc.get("answer", "")
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--results",    required=True, help="JSONL of pre-computed QueryResults from entity_boosted retrieval")
    parser.add_argument("--clean",      required=True, help="clean.jsonl for id→text lookup")
    parser.add_argument("--model-name", default="bge-reranker-base-finetuned")
    parser.add_argument("--top-k",      type=int, default=5)
    args = parser.parse_args()

    reranker   = FinetuneReranker(args.model_path)
    results    = load_query_results(Path(args.results))
    id_to_text = load_id_to_text(Path(args.clean))

    reranked = rerank_query_results(results, reranker, id_to_text, top_k=args.top_k)
    summary  = aggregate_metrics(reranked, config_name="finetuned_reranker", model_name=args.model_name)

    hit5 = summary.hit_rate_5
    mrr  = summary.mrr

    print(f"\n{'='*45}")
    print(f"  Model : {args.model_name}")
    print(f"  N     : {summary.num_queries}")
    print(f"  Hit@5 : {hit5:.3f}  (baseline {BASELINE_HIT5:.3f}, Δ{hit5 - BASELINE_HIT5:+.3f})")
    print(f"  MRR   : {mrr:.3f}  (baseline {BASELINE_MRR:.3f}, Δ{mrr - BASELINE_MRR:+.3f})")
    print(f"  Hit@1 : {summary.hit_rate_1:.3f}")
    print(f"  Hit@3 : {summary.hit_rate_3:.3f}")
    print(f"  NDCG@10: {summary.ndcg_10:.3f}")
    print(f"{'='*45}\n")


if __name__ == "__main__":
    main()
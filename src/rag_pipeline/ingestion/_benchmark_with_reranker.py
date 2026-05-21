"""
rag_pipeline/p04_ingestion/_benchmark_with_reranker.py
Helper to run benchmark with different rerankers (one at a time).
"""
import logging
from typing import Dict
from ._benchmark_reranker import evaluate_with_reranker
logger = logging.getLogger(__name__)

def run_benchmark_with_reranker(query: str, candidates: list, reranker_name: str=None, top_k: int=5):
    """One-off helper to test in p04 style"""
    reranked_ids, metrics = evaluate_with_reranker(query=query, retrieved_candidates=candidates, reranker_name=reranker_name, top_k=top_k)
    logger.info(f"Reranker: {metrics.get('reranker_name')} | Latency: {metrics.get('reranker_latency_ms')}ms")
    return (reranked_ids, metrics)
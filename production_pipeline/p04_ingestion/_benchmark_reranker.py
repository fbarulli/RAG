"""
production_pipeline/p04_ingestion/_benchmark_reranker.py
Integration point for testing multiple rerankers in p04.
"""

import time
from typing import List, Dict, Tuple
import logging

from ._rerankers import load_rerankers
from ._reranker_runner import run_reranking

logger = logging.getLogger(__name__)


def evaluate_with_reranker(
    query: str,
    retrieved_candidates: List[Dict],      # list of full payload dicts from Qdrant
    reranker_name: str = None,
    top_k: int = 5
) -> Tuple[List[str], Dict]:
    """
    Run one reranker and return reranked ids + metrics.
    """
    start = time.perf_counter()
    
    rerankers = load_rerankers(reranker_name)
    if not rerankers:
        return [c.get('es_id') for c in retrieved_candidates[:top_k]], {"reranker": "none", "latency_ms": 0.0}
    
    reranker_config = rerankers[0]
    
    reranked_ids, rerank_latency = run_reranking(
        reranker_config=reranker_config,
        query=query,
        candidates=retrieved_candidates,
        top_k=top_k
    )
    
    total_latency = (time.perf_counter() - start) * 1000
    
    metrics = {
        "reranker_name": reranker_config["name"],
        "reranker_model": reranker_config["model"],
        "reranker_latency_ms": round(rerank_latency, 1),
        "total_rerank_time_ms": round(total_latency, 1)
    }
    
    return reranked_ids, metrics

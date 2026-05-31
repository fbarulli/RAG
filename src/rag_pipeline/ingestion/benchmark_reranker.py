"""
rag_pipeline/ingestion/benchmark_reranker.py
Integration point for reranker evaluation in the benchmark pipeline.
Uses RerankerRunner as the single reranking backend.
"""
import time
import logging
from typing import Dict, List, Optional, Tuple

from .reranker_runner import RerankerRunner

logger = logging.getLogger(__name__)

_runner_cache: Dict[str, RerankerRunner] = {}


def _get_runner(reranker_name: str) -> Optional[RerankerRunner]:
    if reranker_name not in _runner_cache:
        try:
            _runner_cache[reranker_name] = RerankerRunner(model_key=reranker_name)
        except Exception as e:
            logger.error(f"Failed to load reranker '{reranker_name}': {e}")
            return None
    return _runner_cache[reranker_name]


def evaluate_with_reranker(
    query: str,
    retrieved_candidates: List[Dict],
    reranker_name: Optional[str] = None,
    top_k: int = 5,
) -> Tuple[List[str], Dict]:
    """Run reranking and return (reranked_ids, metrics)."""
    _no_rerank = {"reranker_name": "none", "reranker_latency_ms": 0.0}

    if not retrieved_candidates:
        return ([], _no_rerank)

    if reranker_name is None:
        ids = [c.get("es_id", "") for c in retrieved_candidates[:top_k]]
        return (ids, _no_rerank)

    runner = _get_runner(reranker_name)
    if runner is None:
        ids = [c.get("es_id", "") for c in retrieved_candidates[:top_k]]
        return (ids, _no_rerank)

    try:
        start = time.perf_counter()
        start_rerank = time.perf_counter()
        doc_score_pairs = runner.rerank(
            query=query,
            documents=[(c.get("faq_question", "") + " " + (c.get("answer") or c.get("text", ""))).strip() for c in retrieved_candidates],
        )
        latency_ms = (time.perf_counter() - start_rerank) * 1000
        score_map = {doc: score for doc, score in doc_score_pairs}
        sorted_candidates = sorted(retrieved_candidates, key=lambda c: score_map.get(c.get("answer") or c.get("text", ""), 0.0), reverse=True)
        reranked_ids = [c.get("es_id", "") for c in sorted_candidates[:top_k]]
        total_ms = (time.perf_counter() - start) * 1000
        metrics = {
            "reranker_name": reranker_name,
            "reranker_model": runner.model_name,
            "reranker_latency_ms": round(latency_ms, 1),
            "total_rerank_time_ms": round(total_ms, 1),
        }
        return ([rid for rid in reranked_ids if rid], metrics)
    except Exception as e:
        logger.error(f"Reranking failed for '{reranker_name}': {e}", exc_info=True)
        ids = [c.get("es_id", "") for c in retrieved_candidates[:top_k]]
        return (ids, {"reranker_name": "error", "reranker_latency_ms": 0.0})

"""
rag_pipeline/p04_ingestion/_benchmark_reranker.py
Public Functions for Rerank Evaluation Integration:

def evaluate_with_reranker(query: str, retrieved_candidates: List[Dict], reranker_name: str = None, top_k: int = 5) -> Tuple[List[str], Dict]:
    Run one reranker and return reranked ids + metrics.
    I/O: query (str), retrieved_candidates (List[Dict]), reranker_name (str), top_k (int) -> Tuple[List[str], Dict]
rag_pipeline/p04_ingestion/_benchmark_reranker.py
Integration point for testing multiple rerankers in p04.
"""
import time
import threading
from functools import lru_cache
from typing import List, Dict, Tuple, Optional
import logging
from ._rerankers import load_rerankers
from ._reranker_runner import run_reranking
logger = logging.getLogger(__name__)
_LOADED_RERANKER_CACHE: Dict[str, dict] = {}
_CACHE_LOCK = threading.Lock()

def _clear_reranker_cache() -> None:
    """Clear the reranker cache, e.g. after a config reload."""
    with _CACHE_LOCK:
        _LOADED_RERANKER_CACHE.clear()
    _load_rerankers_cached.cache_clear()

@lru_cache(maxsize=32)
def _load_rerankers_cached(reranker_name: str) -> list:
    """
    Load rerankers by name with lru_cache.
    Provides cache_info() for free — useful for monitoring hit rates.
    Only suitable if load_rerankers() is pure (same input -> same output).
    """
    logger.info(f'Loading reranker: {reranker_name}')
    return load_rerankers(reranker_name)

def _get_reranker_entry(reranker_name: str) -> Optional[dict]:
    """
    Fetch a resolved reranker entry from the thread-safe cache, populating it
    on first access. Returns None if no reranker is found for the given name.
    """
    if reranker_name in _LOADED_RERANKER_CACHE:
        return _LOADED_RERANKER_CACHE[reranker_name]
    with _CACHE_LOCK:
        if reranker_name in _LOADED_RERANKER_CACHE:
            return _LOADED_RERANKER_CACHE[reranker_name]
        rerankers = _load_rerankers_cached(reranker_name)
        if not rerankers:
            return None
        if len(rerankers) > 1:
            logger.debug(f"Multiple rerankers found for '{reranker_name}', using first: {rerankers[0].get('name')}")
        config = rerankers[0]
        entry = {'config': config, 'display_name': config.get('short_name') or config.get('name'), 'model': config.get('model')}
        _LOADED_RERANKER_CACHE[reranker_name] = entry
        return entry

def _safe_slice_ids(candidates: List[Dict], top_k: int) -> List[str]:
    """
    Extract es_ids from candidates up to top_k, with a warning if fewer are available.
    """
    ids = [c.get('es_id') for c in candidates[:top_k]]
    if len(ids) < top_k:
        logger.warning(f'Only {len(ids)} candidate(s) available, fewer than top_k={top_k}.')
    return ids

def evaluate_with_reranker(query: str, retrieved_candidates: List[Dict], reranker_name: Optional[str]=None, top_k: int=5) -> Tuple[List[str], Dict]:
    """
    Run reranking using existing reranker modules with caching.

    Args:
        query: The search query string.
        retrieved_candidates: List of candidate dicts, each expected to have an 'es_id' key.
        reranker_name: Name of the reranker to use. If None, returns original order.
        top_k: Number of top results to return.

    Returns:
        A tuple of (list of es_ids, metrics dict).
    """
    _no_rerank = {'reranker_name': 'none', 'reranker_latency_ms': 0.0}
    if not retrieved_candidates:
        logger.warning('Empty candidates list provided.')
        return ([], _no_rerank)
    if reranker_name is None:
        logger.warning('No reranker_name provided. Returning original order.')
        return (_safe_slice_ids(retrieved_candidates, top_k), _no_rerank)
    try:
        entry = _get_reranker_entry(reranker_name)
        if entry is None:
            logger.warning(f"No reranker found for '{reranker_name}'. Returning original order.")
            return (_safe_slice_ids(retrieved_candidates, top_k), _no_rerank)
        start = time.perf_counter()
        reranked_ids, rerank_latency = run_reranking(reranker_config=entry['config'], query=query, candidates=retrieved_candidates, top_k=top_k)
        total_latency = (time.perf_counter() - start) * 1000
        metrics = {'reranker_name': entry['display_name'], 'reranker_model': entry['model'], 'reranker_latency_ms': round(rerank_latency, 1), 'total_rerank_time_ms': round(total_latency, 1)}
        return (reranked_ids, metrics)
    except Exception as e:
        logger.error(f'Error during reranking with {reranker_name}: {e}', exc_info=True)
        return (_safe_slice_ids(retrieved_candidates, top_k), {'reranker_name': 'error', 'reranker_latency_ms': 0.0})
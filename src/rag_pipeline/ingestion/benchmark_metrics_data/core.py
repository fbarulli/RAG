"""
Core metric formulas - no I/O, no retrieval logic.
"""
import math
import re
from statistics import mean, quantiles
from typing import Optional

def check_code_integrity(text: str) -> float:
    """
    Return 1.0 if *text* contains no code fences, or the fraction of fenced
    blocks that are non-empty (a proxy for "complete" blocks).
    
    Handles language hints with special characters: c++, c#, bash-script, etc.
    """
    blocks = re.findall('```[\\w\\+\\#\\-]*\\s*\\n?(.*?)```', text, re.DOTALL)
    if not blocks:
        return 1.0
    complete = sum((1 for b in blocks if b.strip()))
    return complete / len(blocks)

def compute_latency_percentiles(latencies: list[float]) -> dict[str, float]:
    """Return p50 / p95 / p99 from *latencies* (milliseconds).
    
    Uses the "nearest rank" method: percentile p corresponds to ceil(p * n)
    """
    if not latencies:
        return {'p50': 0.0, 'p95': 0.0, 'p99': 0.0}
    if len(latencies) == 1:
        v = latencies[0]
        return {'p50': v, 'p95': v, 'p99': v}
    n = len(latencies)
    s = sorted(latencies)

    def nearest_rank(p: float) -> float:
        """Return value at nearest rank for percentile p (0-1)."""
        idx = max(0, min(int(math.ceil(p * n)) - 1, n - 1))
        return s[idx]
    return {'p50': nearest_rank(0.5), 'p95': nearest_rank(0.95), 'p99': nearest_rank(0.99)}

def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0

def compute_map(hits: tuple[str, ...], expected_id: str) -> float:
    """Mean Average Precision for single ground truth document."""
    if expected_id not in hits:
        return 0.0
    rank = hits.index(expected_id) + 1
    return 1.0 / rank


def compute_recall_at_k(hits: tuple[str, ...], expected_id: str, k: int) -> bool:
    """Recall@K for single ground truth document."""
    return expected_id in hits[:k]

def compute_reciprocal_rank(hits: tuple[str, ...], expected_id: str) -> float:
    if expected_id in hits:
        return 1.0 / (hits.index(expected_id) + 1)
    return 0.0

def compute_ndcg_at_k(hits: tuple[str, ...], expected_id: str, k: int) -> float:
    """
    NDCG@k for single binary-relevance ground truth.
    
    For a single relevant document at rank r (1-indexed):
        DCG = 1 / log2(r + 1)
        IDCG = 1 / log2(1 + 1) = 1.0
        NDCG = DCG / IDCG = DCG
    
    Returns 0.0 if relevant doc not found in top k.
    """
    if expected_id not in hits:
        return 0.0
    rank = hits.index(expected_id) + 1
    if rank > k:
        return 0.0
    dcg = 1.0 / math.log2(rank + 1)
    idcg = 1.0 / math.log2(2)
    return dcg / idcg
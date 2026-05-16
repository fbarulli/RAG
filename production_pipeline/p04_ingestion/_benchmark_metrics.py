"""
_benchmark_metrics.py
=====================
Compute retrieval metrics for benchmark evaluation.

Single responsibility: metric computation logic.
No I/O, no reporting, no model loading, no heuristic guessing.
Stratification relies exclusively on factual topic assignments.

Functions:
    evaluate_config(...) -> list[QueryResult]
    aggregate_metrics(...) -> MetricSummary
    aggregate_metrics_by_topic(...) -> list[MetricSummary]
"""
import math
import re
import time
from collections import defaultdict
from statistics import mean, quantiles, stdev, mean
from typing import Optional


from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams

from ._benchmark_types import MetricSummary, QueryResult


# ---------------------------------------------------------------------------
# Core logic: Utilities
# ---------------------------------------------------------------------------

def check_code_integrity(text: str) -> float:
    """
    Check if code blocks in text are complete and well-formed.
    
    Returns 1.0 if no code blocks present (neutral), or ratio of complete blocks.
    """
    blocks = re.findall(r"```(?:\w+)?\s*(.*?)```", text, re.DOTALL)
    if not blocks:
        return 1.0
    complete = sum(1 for b in blocks if b.strip() and "\n" in b)
    return complete / len(blocks)


def compute_latency_percentiles(latencies: list[float]) -> dict[str, float]:
    """Compute p50, p95, p99 latency using proper percentile calculation."""
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    try:
        if len(latencies) >= 2:
            q = quantiles(latencies, n=100)
            return {"p50": q[49], "p95": q[94], "p99": q[98]}
        val = latencies[0]
        return {"p50": val, "p95": val, "p99": val}
    except Exception:
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        return {
            "p50": sorted_lat[math.ceil(n * 0.50) - 1],
            "p95": sorted_lat[math.ceil(n * 0.95) - 1],
            "p99": sorted_lat[math.ceil(n * 0.99) - 1],
        }


def safe_mean(values: list[float]) -> float:
    """Compute mean, returning 0.0 for empty lists."""
    return mean(values) if values else 0.0


# ---------------------------------------------------------------------------
# Core logic: Metric formulas
# ---------------------------------------------------------------------------

def compute_hit_rate(hits: tuple[str, ...], expected_id: str, k: int) -> bool:
    """Check if expected_id appears in top-k results."""
    return expected_id in hits[:k]


def compute_reciprocal_rank(hits: tuple[str, ...], expected_id: str) -> float:
    """Compute 1/rank if found, else 0."""
    if expected_id in hits:
        return 1.0 / (hits.index(expected_id) + 1)
    return 0.0


def compute_ndcg_at_k(hits: tuple[str, ...], expected_id: str, k: int) -> float:
    """Compute NDCG@k for binary relevance (1 if correct, 0 otherwise)."""
    if expected_id not in hits:
        return 0.0
    rank = hits.index(expected_id) + 1
    if rank > k:
        return 0.0
    dcg = 1.0 / math.log2(rank + 1)
    idcg = 1.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Core logic: Retrieval execution (config-aware)
# ---------------------------------------------------------------------------

def run_retrieval_query(
    client, 
    collection: str, 
    query_vector: list, 
    course_filter: str, 
    config: dict, 
    top_k: int
) -> tuple[tuple[str, ...], Optional[str], tuple[float, ...], float]:
    """
    Execute a single retrieval query against Qdrant, applying config options.
    
    Returns:
        hit_ids: Tuple of document IDs
        top_answer: The full answer text of the #1 result (or None)
        scores: Tuple of relevance scores
        latency_ms: Execution time in milliseconds
    """
    start = time.perf_counter()
    
    must_conditions = []
    if course_filter:
        must_conditions.append(FieldCondition(key="course", match=MatchValue(value=course_filter)))
    
    if config.get("filters"):
        for cond in config["filters"].get("must", []):
            key = cond.get("key")
            match = cond.get("match", {})
            if key and "value" in match:
                must_conditions.append(FieldCondition(key=key, match=MatchValue(value=match["value"])))
    
    query_filter = Filter(must=must_conditions) if must_conditions else None
    effective_limit = config.get("limit", top_k)
    
    search_params = None
    if config.get("hnsw_ef"):
        search_params = SearchParams(hnsw_ef=config["hnsw_ef"])
    
    result = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=effective_limit,
        query_filter=query_filter,
        score_threshold=config.get("score_threshold"),
        search_params=search_params,
        with_payload=True,
        with_vectors=False,
    )
    
    latency_ms = (time.perf_counter() - start) * 1000
    points = result.points
    
    hit_ids = tuple(p.payload.get("es_id", "") for p in points)
    scores = tuple(float(p.score) if p.score is not None else 0.0 for p in points)
    top_answer = points[0].payload.get("answer", None) if points else None
    
    return hit_ids, top_answer, scores, latency_ms


# ---------------------------------------------------------------------------
# Core logic: Evaluation & Aggregation
# ---------------------------------------------------------------------------

def evaluate_config(client, collection: str, model, test_set: list[dict], topic_map: dict, config: dict, top_k: int) -> list[QueryResult]:
    """
    Evaluate a single config/model combination against the test set.
    Uses factual topic/subtopic assignments from topic_map.
    """
    results = []
    
    for test in test_set:
        query = test["query"]
        expected_id = test["expected_id"]
        course = test["course"]
        ref_answer = test["answer"]
        
        topic_info = topic_map.get(expected_id, {})
        topic = topic_info.get("topic", -1)
        subtopic = topic_info.get("subtopic")
        
        query_vector = model.encode(query, convert_to_numpy=True).tolist()
        
        hit_ids, top_answer, scores, latency_ms = run_retrieval_query(
            client=client,
            collection=collection,
            query_vector=query_vector,
            course_filter=course,
            config=config,
            top_k=top_k,
        )
        
        code_int_ref = check_code_integrity(ref_answer)
        code_int_ret = check_code_integrity(top_answer) if top_answer else None
        
        results.append(QueryResult(
            query_id=test["query_id"],
            query_text=query,
            expected_id=expected_id,
            course=course,
            topic=topic,
            subtopic=subtopic,
            hit_ids=hit_ids,
            hit_scores=scores,
            latency_ms=latency_ms,
            code_integrity_ref=code_int_ref,
            code_integrity_retrieved=code_int_ret,
        ))
    
    return results





def aggregate_metrics(results: list[QueryResult], config_name: str, model_name: str,
                     topic: Optional[int] = None, subtopic: Optional[int] = None) -> MetricSummary:
    """Aggregate per-query results into summary metrics with all new diagnostics."""
    if not results:
        return MetricSummary(
            config_name=config_name,
            model_name=model_name,
            topic=topic,
            subtopic=subtopic,
            num_queries=0
        )

    n = len(results)

    # Core metrics
    hit_1 = sum(1 for r in results if compute_hit_rate(r.hit_ids, r.expected_id, 1))
    hit_3 = sum(1 for r in results if compute_hit_rate(r.hit_ids, r.expected_id, 3))
    hit_5 = sum(1 for r in results if compute_hit_rate(r.hit_ids, r.expected_id, 5))
    hit_10 = sum(1 for r in results if compute_hit_rate(r.hit_ids, r.expected_id, 10))

    mrr = safe_mean([compute_reciprocal_rank(r.hit_ids, r.expected_id) for r in results])
    ndcg = safe_mean([compute_ndcg_at_k(r.hit_ids, r.expected_id, 10) for r in results])

    # Latency & Code Integrity
    latencies = [r.latency_ms for r in results]
    lat_pcts = compute_latency_percentiles(latencies)

    code_int_ref = safe_mean([r.code_integrity_ref for r in results])
    code_int_ret_vals = [r.code_integrity_retrieved for r in results if r.code_integrity_retrieved is not None]
    code_int_ret = safe_mean(code_int_ret_vals) if code_int_ret_vals else None

    # === NEW METRICS ===
    ranks = []
    failures = []
    failure_sims = []
    cross_course_errors = 0

    for r in results:
        # Rank of the correct document
        if r.expected_id in r.hit_ids:
            rank = r.hit_ids.index(r.expected_id) + 1
            ranks.append(rank)
        else:
            ranks.append(11)                    # penalty value
            failures.append(r)
            if r.hit_scores and len(r.hit_scores) > 0:
                failure_sims.append(r.hit_scores[0])

        # Cross-course contamination (top-1 from different course)
        # Currently this will be near zero because of course filtering
        # You can improve this later by storing course info in the payload
        if r.hit_ids:
            pass  # TODO: enhance when document course is available

    # Calculate new metrics
    cross_course_contamination = cross_course_errors / n if n > 0 else 0.0
    rank_std = stdev(ranks) if len(ranks) >= 2 else 0.0
    failure_count = len(failures)
    avg_failure_similarity = mean(failure_sims) if failure_sims else None

    return MetricSummary(
        config_name=config_name,
        model_name=model_name,
        topic=topic,
        subtopic=subtopic,
        num_queries=n,
        hit_rate_1=hit_1 / n,
        hit_rate_3=hit_3 / n,
        hit_rate_5=hit_5 / n,
        hit_rate_10=hit_10 / n,
        mrr=mrr,
        ndcg_10=ndcg,
        latency_p50=lat_pcts["p50"],
        latency_p95=lat_pcts["p95"],
        latency_p99=lat_pcts.get("p99", 0.0),
        avg_code_integrity_ref=code_int_ref,
        avg_code_integrity_retrieved=code_int_ret,
        # New metrics
        cross_course_contamination=cross_course_contamination,
        rank_std=rank_std,
        failure_count=failure_count,
        avg_failure_similarity=avg_failure_similarity,
    )
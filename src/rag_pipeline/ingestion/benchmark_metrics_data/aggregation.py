"""
Aggregation of per-query results into metric summaries.
"""
from statistics import stdev, mean
from typing import Optional
from ..benchmark_types import MetricSummary, QueryResult
from .core import compute_recall_at_k, compute_reciprocal_rank, compute_ndcg_at_k, compute_map, compute_latency_percentiles, safe_mean

def aggregate_metrics(results: list[QueryResult], config_name: str, model_name: str, topic: Optional[int]=None, subtopic: Optional[int]=None) -> MetricSummary:
    """Collapse per-query results into a :class:`MetricSummary`."""
    if not results:
        return MetricSummary(config_name=config_name, model_name=model_name, topic=topic, subtopic=subtopic, num_queries=0)
    n = len(results)
    recall_counts = {1: 0, 3: 0, 5: 0, 10: 0}
    mrr_sum = 0.0
    ndcg_sum = 0.0
    ndcg_1_sum = 0.0
    ndcg_5_sum = 0.0
    map_sum = 0.0
    latencies = []
    code_int_ref_sum = 0.0
    code_int_ret_vals = []
    ranks = []
    failure_sims = []
    failure_count = 0
    cross_course_count = 0
    for r in results:
        for k in [1, 3, 5, 10]:
            if compute_recall_at_k(r.hit_ids, r.expected_id, k):
                recall_counts[k] += 1
        mrr_sum += compute_reciprocal_rank(r.hit_ids, r.expected_id)
        ndcg_sum   += compute_ndcg_at_k(r.hit_ids, r.expected_id, 10)
        ndcg_1_sum += compute_ndcg_at_k(r.hit_ids, r.expected_id, 1)
        ndcg_5_sum += compute_ndcg_at_k(r.hit_ids, r.expected_id, 5)
        map_sum    += compute_map(r.hit_ids, r.expected_id)
        latencies.append(r.latency_ms)
        code_int_ref_sum += r.code_integrity_ref
        if r.code_integrity_retrieved is not None:
            code_int_ret_vals.append(r.code_integrity_retrieved)
        if r.expected_id in r.hit_ids:
            rank = r.hit_ids.index(r.expected_id) + 1
            ranks.append(rank)
        else:
            ranks.append(11)
            failure_count += 1
            if r.hit_scores:
                failure_sims.append(r.hit_scores[0])
        if r.hit_ids and r.hit_courses:
            top_course = r.hit_courses[0]
            if top_course and top_course != r.course:
                cross_course_count += 1
    hit_rate_1 = recall_counts[1] / n
    hit_rate_3 = recall_counts[3] / n
    hit_rate_5 = recall_counts[5] / n
    hit_rate_10 = recall_counts[10] / n
    mrr = mrr_sum / n
    ndcg   = ndcg_sum   / n
    ndcg_1 = ndcg_1_sum / n
    ndcg_5 = ndcg_5_sum / n
    map_score = map_sum / n
    lat_pcts = compute_latency_percentiles(latencies)
    code_int_ref = code_int_ref_sum / n
    code_int_ret = safe_mean(code_int_ret_vals) if code_int_ret_vals else None
    rank_std = stdev(ranks) if len(ranks) >= 2 else 0.0
    avg_failure_similarity = mean(failure_sims) if failure_sims else None
    cross_course_contamination = cross_course_count / n
    return MetricSummary(config_name=config_name, model_name=model_name, topic=topic, subtopic=subtopic, num_queries=n, hit_rate_1=hit_rate_1, hit_rate_3=hit_rate_3, hit_rate_5=hit_rate_5, hit_rate_10=hit_rate_10, mrr=mrr, ndcg_1=ndcg_1, ndcg_5=ndcg_5, ndcg_10=ndcg, map_score=map_score, latency_p50=lat_pcts['p50'], latency_p95=lat_pcts['p95'], latency_p99=lat_pcts['p99'], avg_code_integrity_ref=code_int_ref, avg_code_integrity_retrieved=code_int_ret, cross_course_contamination=cross_course_contamination, rank_std=rank_std, failure_count=failure_count, avg_failure_similarity=avg_failure_similarity)

def aggregate_metrics_by_topic(results: list[QueryResult], config_name: str, model_name: str) -> list[MetricSummary]:
    """Return one :class:`MetricSummary` per (topic, subtopic) combination."""
    from collections import defaultdict
    groups: dict[tuple, list[QueryResult]] = defaultdict(list)
    for r in results:
        groups[r.topic, r.subtopic].append(r)
    return [aggregate_metrics(group, config_name, model_name, topic=t, subtopic=st) for (t, st), group in sorted(groups.items())]
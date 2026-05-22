"""
rag_pipeline/ingestion/_onnx_bench_failure_analysis.py

Failure analysis and diagnostics for ONNX Cross-Encoder benchmark runs.
RESPONSIBILITY: Produces per-query, topic-level, course-level, and
cross-reranker diagnostic reports from raw QueryResult lists.
No metric computation, no retrieval logic — pure analysis and serialization.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from .benchmark_types import MetricSummary, QueryResult

logger = logging.getLogger(__name__)


def _build_query_failure_records(results, reranker_name):
    records = []
    for r in results:
        if r.expected_id not in r.hit_ids:
            records.append({
                "reranker": reranker_name,
                "query_id": r.query_id,
                "query_text": r.query_text,
                "expected_id": r.expected_id,
                "course": r.course,
                "topic": r.topic,
                "subtopic": r.subtopic,
                "top1_returned": r.hit_ids[0] if r.hit_ids else None,
                "top1_course": r.hit_courses[0] if r.hit_courses else None,
                "top1_score": r.hit_scores[0] if r.hit_scores else None,
                "num_hits": len(r.hit_ids),
                "latency_ms": round(r.latency_ms, 2),
            })
    return records


def _build_query_success_records(results, reranker_name):
    records = []
    for r in results:
        if r.expected_id in r.hit_ids:
            rank = r.hit_ids.index(r.expected_id) + 1
            score_at_rank = r.hit_scores[rank - 1] if r.hit_scores else None
            records.append({
                "reranker": reranker_name,
                "query_id": r.query_id,
                "query_text": r.query_text,
                "expected_id": r.expected_id,
                "course": r.course,
                "topic": r.topic,
                "rank": rank,
                "score_at_rank": score_at_rank,
                "latency_ms": round(r.latency_ms, 2),
            })
    return records


def _build_topic_failure_rates(results, reranker_name):
    groups = defaultdict(list)
    for r in results:
        groups[(r.topic, r.subtopic)].append(r)
    records = []
    for (topic, subtopic), group in sorted(groups.items()):
        n = len(group)
        failures = sum(1 for r in group if r.expected_id not in r.hit_ids)
        hit5 = sum(1 for r in group if r.expected_id in r.hit_ids[:5])
        records.append({
            "reranker": reranker_name,
            "topic": topic,
            "subtopic": subtopic,
            "num_queries": n,
            "failure_count": failures,
            "failure_rate": round(failures / n, 4),
            "hit_rate_5": round(hit5 / n, 4),
        })
    return sorted(records, key=lambda x: x["failure_rate"], reverse=True)


def _build_course_contamination(results, reranker_name):
    records = []
    for r in results:
        if not r.hit_ids or not r.hit_courses:
            continue
        top_course = r.hit_courses[0]
        if top_course and top_course != r.course:
            records.append({
                "reranker": reranker_name,
                "query_id": r.query_id,
                "query_text": r.query_text,
                "expected_course": r.course,
                "top1_course": top_course,
                "top1_id": r.hit_ids[0],
                "top1_score": r.hit_scores[0] if r.hit_scores else None,
                "expected_id": r.expected_id,
                "expected_found": r.expected_id in r.hit_ids,
            })
    return records


def _build_hard_queries(results_by_reranker):
    failure_map = {}
    reranker_names = list(results_by_reranker.keys())
    for reranker_name, results in results_by_reranker.items():
        for r in results:
            if r.expected_id not in r.hit_ids:
                if r.query_id not in failure_map:
                    failure_map[r.query_id] = {
                        "query_id": r.query_id,
                        "query_text": r.query_text,
                        "expected_id": r.expected_id,
                        "course": r.course,
                        "topic": r.topic,
                        "failed_rerankers": [],
                    }
                failure_map[r.query_id]["failed_rerankers"].append(reranker_name)
    hard = [v for v in failure_map.values() if len(v["failed_rerankers"]) == len(reranker_names)]
    return sorted(hard, key=lambda x: x["query_id"])


def _build_reranker_disagreements(results_by_reranker):
    query_top1 = defaultdict(dict)
    query_meta = {}
    for reranker_name, results in results_by_reranker.items():
        for r in results:
            query_top1[r.query_id][reranker_name] = r.hit_ids[0] if r.hit_ids else None
            if r.query_id not in query_meta:
                query_meta[r.query_id] = {
                    "query_text": r.query_text,
                    "expected_id": r.expected_id,
                    "course": r.course,
                }
    records = []
    for query_id, top1_map in query_top1.items():
        unique_top1 = set(v for v in top1_map.values() if v)
        if len(unique_top1) > 1:
            meta = query_meta[query_id]
            records.append({
                "query_id": query_id,
                "query_text": meta["query_text"],
                "expected_id": meta["expected_id"],
                "course": meta["course"],
                "top1_by_reranker": top1_map,
                "num_unique_top1": len(unique_top1),
            })
    return sorted(records, key=lambda x: x["num_unique_top1"], reverse=True)


def _build_failure_score_distribution(results, reranker_name):
    failure_scores = [r.hit_scores[0] for r in results if r.expected_id not in r.hit_ids and r.hit_scores]
    success_scores = [r.hit_scores[0] for r in results if r.expected_id in r.hit_ids and r.hit_scores]
    return {
        "reranker": reranker_name,
        "failure_top1_score_avg": round(mean(failure_scores), 4) if failure_scores else None,
        "success_top1_score_avg": round(mean(success_scores), 4) if success_scores else None,
        "failure_count": len(failure_scores),
        "success_count": len(success_scores),
    }


def _write(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s (%d records)", path.name, len(data) if isinstance(data, list) else 1)


def save_failure_analysis(
    results_by_reranker: Dict[str, List[QueryResult]],
    metric_summaries: List[MetricSummary],
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    analysis_dir = output_dir / "failure_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    all_failures, all_successes, all_topic_rates, all_contamination, score_distributions = [], [], [], [], []

    for reranker_name, results in results_by_reranker.items():
        all_failures.extend(_build_query_failure_records(results, reranker_name))
        all_successes.extend(_build_query_success_records(results, reranker_name))
        all_topic_rates.extend(_build_topic_failure_rates(results, reranker_name))
        all_contamination.extend(_build_course_contamination(results, reranker_name))
        score_distributions.append(_build_failure_score_distribution(results, reranker_name))

    hard_queries = _build_hard_queries(results_by_reranker)
    disagreements = _build_reranker_disagreements(results_by_reranker)

    _write(analysis_dir / "per_query_failures.json", all_failures)
    _write(analysis_dir / "per_query_successes.json", all_successes)
    _write(analysis_dir / "topic_failure_rates.json", all_topic_rates)
    _write(analysis_dir / "course_contamination.json", all_contamination)
    _write(analysis_dir / "hard_queries.json", hard_queries)
    _write(analysis_dir / "reranker_disagreements.json", disagreements)
    _write(analysis_dir / "score_distributions.json", score_distributions)

    logger.info(
        "📊 Failure analysis saved to %s | failures=%d hard_queries=%d disagreements=%d",
        analysis_dir, len(all_failures), len(hard_queries), len(disagreements),
    )
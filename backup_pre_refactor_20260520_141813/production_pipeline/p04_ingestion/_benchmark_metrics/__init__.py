"""
_benchmark_metrics package
==========================
Metric computation for retrieval benchmark evaluation.

Public API
----------
    evaluate_config(...) -> list[QueryResult]
    aggregate_metrics(...) -> MetricSummary
    aggregate_metrics_by_topic(...) -> list[MetricSummary]
"""

from .evaluation import evaluate_config
from .aggregation import aggregate_metrics, aggregate_metrics_by_topic

__all__ = [
    "evaluate_config",
    "aggregate_metrics",
    "aggregate_metrics_by_topic",
]
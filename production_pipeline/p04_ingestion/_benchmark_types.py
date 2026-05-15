"""
_benchmark_types.py
===================
Shared data classes for the retrieval benchmark pipeline.

Single source of truth for query results and aggregated metrics.
All classes are frozen to prevent accidental mutation during aggregation.

Usage:
    from _benchmark_types import QueryResult, MetricSummary
"""
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class QueryResult:
    """Lean, immutable record of a single retrieval query execution."""
    query_id: str
    query_text: str
    expected_id: str
    course: str
    topic: int
    subtopic: Optional[int]
    intent: str
    hit_ids: tuple[str, ...]
    hit_scores: tuple[float, ...]
    latency_ms: float
    code_integrity_ref: float
    code_integrity_retrieved: Optional[float] = None

    def to_dict(self) -> dict:
        """JSON-safe serialization."""
        return asdict(self)


@dataclass(frozen=True)
class MetricSummary:
    """Aggregated metrics for a specific config/model/topic/intent slice."""
    config_name: str
    model_name: str
    topic: Optional[int] = None
    subtopic: Optional[int] = None
    intent: Optional[str] = None
    num_queries: int = 0
    hit_rate_1: float = 0.0
    hit_rate_3: float = 0.0
    hit_rate_5: float = 0.0
    hit_rate_10: float = 0.0
    mrr: float = 0.0
    ndcg_10: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    avg_code_integrity_ref: float = 0.0
    avg_code_integrity_retrieved: Optional[float] = None

    def to_dict(self) -> dict:
        """JSON-safe serialization."""
        return asdict(self)
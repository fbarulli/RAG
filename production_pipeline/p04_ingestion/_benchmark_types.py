"""
_benchmark_types.py
===================
Shared data classes for the retrieval benchmark pipeline.
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class TestQuery:
    id: str
    question: str
    expected_id: str
    course: str
    section: str
    answer: str
    strategy: str
    original_question: str
    topic: Optional[int] = None
    subtopic: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TestQuery":
        valid_fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    query_text: str
    expected_id: str
    course: str
    topic: int
    subtopic: Optional[int]
    hit_ids: tuple[str, ...]
    hit_scores: tuple[float, ...]
    latency_ms: float
    code_integrity_ref: float
    code_integrity_retrieved: Optional[float] = None
    query_type: str = "unknown"


@dataclass(frozen=True)
class MetricSummary:
    """Aggregated metrics including new diagnostics."""
    config_name: str
    model_name: str
    topic: Optional[int] = None
    subtopic: Optional[int] = None
    num_queries: int = 0

    # Core retrieval metrics
    hit_rate_1: float = 0.0
    hit_rate_3: float = 0.0
    hit_rate_5: float = 0.0
    hit_rate_10: float = 0.0
    mrr: float = 0.0
    ndcg_10: float = 0.0

    # Latency
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0

    # Code integrity
    avg_code_integrity_ref: float = 0.0
    avg_code_integrity_retrieved: Optional[float] = None

    # === NEW METRICS ===
    cross_course_contamination: float = 0.0      
    rank_std: float = 0.0                        
    failure_count: int = 0                       
    avg_failure_similarity: Optional[float] = None  
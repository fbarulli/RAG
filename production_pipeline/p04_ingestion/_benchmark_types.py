"""
_benchmark_types.py
===================
Shared data classes for the retrieval benchmark pipeline.
Intent field removed; stratification relies on topic/subtopic assignments.
"""
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class TestQuery:
    """
    A single test query for benchmark evaluation.
    
    Used by:
    - p05_evaluation: Query generation (output)
    - p04_ingestion: Benchmark loading (input)
    """
    id: str  # Unique query ID (UUID)
    question: str  # The generated/synthetic query text
    expected_id: str  # The document ID that should be retrieved
    course: str
    section: str
    answer: str  # Reference answer for code_integrity metric
    strategy: str  # Which prompt strategy generated this query
    original_question: str  # The source FAQ question
    topic: Optional[int] = None  # Populated by topic modeling post-hoc
    subtopic: Optional[int] = None
    
    def to_dict(self) -> dict:
        """JSON-safe serialization for test.jsonl output."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "TestQuery":
        """Parse from JSON dict, ignoring unknown fields for forward compatibility."""
        valid_fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


@dataclass(frozen=True)
class QueryResult:
    """Lean, immutable record of a single retrieval query execution."""
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

    def to_dict(self) -> dict:
        """JSON-safe serialization."""
        return asdict(self)


@dataclass(frozen=True)
class MetricSummary:
    """Aggregated metrics for a specific config/model/topic slice."""
    config_name: str
    model_name: str
    topic: Optional[int] = None
    subtopic: Optional[int] = None
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
"""
rag_pipeline/p04_ingestion/_benchmark_types.py
===================
Shared data classes for the retrieval benchmark pipeline.

No I/O, no logic — pure data definitions.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, computed_field


class SearchResult(BaseModel, frozen=True):
    """Intermediate result from a retriever before conversion to QueryResult."""
    hit_ids: tuple[str, ...]
    hit_scores: tuple[float, ...]
    hit_courses: tuple[str, ...]
    top_answer: Optional[str]
    latency_ms: float
    hit_answers: tuple[str, ...]
    hit_questions: tuple[str, ...] = ()
    reranker_latency_ms: float = 0.0


class QueryResult(BaseModel, frozen=True):
    """Per-query retrieval result."""
    query_id: str
    query_text: str
    expected_id: str
    course: str
    topic: Optional[int]
    subtopic: Optional[int]
    hit_ids: tuple[str, ...]
    hit_scores: tuple[float, ...]
    latency_ms: float
    code_integrity_ref: float
    code_integrity_retrieved: Optional[float] = None
    query_type: str = 'unknown'
    hit_courses: Optional[tuple[str, ...]] = None
    reranker_latency_ms: Optional[float] = None
    ner_primary_entity: Optional[str] = None
    ner_entities: tuple[str, ...] = ()

    @computed_field
    @property
    def rank(self) -> Optional[int]:
        return (self.hit_ids.index(self.expected_id) + 1) if self.expected_id in self.hit_ids else None

    @computed_field
    @property
    def hit_at_1(self) -> bool:
        return self.expected_id in self.hit_ids[:1]

    @computed_field
    @property
    def hit_at_3(self) -> bool:
        return self.expected_id in self.hit_ids[:3]

    @computed_field
    @property
    def hit_at_5(self) -> bool:
        return self.expected_id in self.hit_ids[:5]


class MetricSummary(BaseModel, frozen=True):
    """Aggregated metrics for one (config, model) pair, optionally stratified by topic."""
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
    ndcg_1:  float = 0.0
    ndcg_5:  float = 0.0
    ndcg_10: float = 0.0
    map_score: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    avg_code_integrity_ref: float = 0.0
    avg_code_integrity_retrieved: Optional[float] = None
    cross_course_contamination: float = 0.0
    rank_std: float = 0.0
    failure_count: int = 0
    avg_failure_similarity: Optional[float] = None

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_benchmark_row(cls, row: dict) -> "MetricSummary":
        """Construct from a benchmark_results.json row or equivalent dict."""
        num_queries = row.get("num_queries") or 1
        failure_count = row.get("failure_count", 0)
        return cls(
            config_name=row.get("config_name", ""),
            model_name=row.get("model_name", ""),
            topic=row.get("topic"),
            subtopic=row.get("subtopic"),
            num_queries=num_queries,
            hit_rate_1=row.get("hit_rate_1", row.get("h1", 0.0)) or 0.0,
            hit_rate_3=row.get("hit_rate_3", row.get("h3", 0.0)) or 0.0,
            hit_rate_5=row.get("hit_rate_5", row.get("h5", 0.0)) or 0.0,
            hit_rate_10=row.get("hit_rate_10", row.get("h10", 0.0)) or 0.0,
            mrr=row.get("mrr", 0.0) or 0.0,
            ndcg_1=row.get("ndcg_1", 0.0) or 0.0,
            ndcg_5=row.get("ndcg_5", 0.0) or 0.0,
            ndcg_10=row.get("ndcg_10", 0.0) or 0.0,
            map_score=row.get("map_score", 0.0) or 0.0,
            latency_p50=row.get("latency_p50", 0.0) or 0.0,
            latency_p95=row.get("latency_p95", 0.0) or 0.0,
            latency_p99=row.get("latency_p99", 0.0) or 0.0,
            avg_code_integrity_ref=row.get("avg_code_integrity_ref", 0.0) or 0.0,
            avg_code_integrity_retrieved=row.get("avg_code_integrity_retrieved"),
            cross_course_contamination=row.get("cross_course_contamination", row.get("cross_course", 0.0)) or 0.0,
            rank_std=row.get("rank_std", 0.0) or 0.0,
            failure_count=failure_count,
            avg_failure_similarity=row.get("avg_failure_similarity"),
        )
"""
SQLModel table definitions — single source of truth for the results DB.

Tables:
    CorpusDoc   — the FAQ corpus (questions + answers)
    Run         — one row per experiment+config execution
    RunMetrics  — aggregate metrics for a Run (1:1)
    QueryResult — one row per query per Run

Relationships are handled explicitly in store.py via SQL joins,
not ORM back-references, to avoid SQLModel forward-reference issues.
"""
from __future__ import annotations
from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field
import json


class CorpusDoc(SQLModel, table=True):
    __tablename__ = "corpus"

    es_id:              str           = Field(primary_key=True)
    question:           str
    answer:             str
    course:             str
    section:            Optional[str] = None
    ner_category:       Optional[str] = None
    ner_primary_entity: Optional[str] = None
    topic:              Optional[int] = None
    subtopic:           Optional[int] = None


class Run(SQLModel, table=True):
    __tablename__ = "runs"

    run_id:      str           = Field(primary_key=True)  # f"{experiment}__{config}"
    experiment:  str
    patch:       str
    config:      str
    model:       str
    git_commit:  str           = "unknown"
    timestamp:   datetime      = Field(default_factory=datetime.utcnow)
    corpus_size: Optional[int] = None


class RunMetrics(SQLModel, table=True):
    __tablename__ = "run_metrics"

    run_id:                       str            = Field(foreign_key="runs.run_id", primary_key=True)
    num_queries:                  Optional[int]   = None
    hit_rate_1:                   Optional[float] = None
    hit_rate_3:                   Optional[float] = None
    hit_rate_5:                   Optional[float] = None
    hit_rate_10:                  Optional[float] = None
    mrr:                          Optional[float] = None
    ndcg_1:                       Optional[float] = None
    ndcg_5:                       Optional[float] = None
    ndcg_10:                      Optional[float] = None
    map_score:                    Optional[float] = None
    latency_p50:                  Optional[float] = None
    latency_p95:                  Optional[float] = None
    latency_p99:                  Optional[float] = None
    avg_code_integrity_ref:       Optional[float] = None
    avg_code_integrity_retrieved: Optional[float] = None
    cross_course_contamination:   Optional[float] = None
    rank_std:                     Optional[float] = None
    failure_count:                Optional[int]   = None
    avg_failure_similarity:       Optional[float] = None


class QueryResult(SQLModel, table=True):
    __tablename__ = "query_results"

    id:          Optional[int]   = Field(default=None, primary_key=True)
    run_id:      str             = Field(foreign_key="runs.run_id")
    query_id:    str
    query_text:  str
    expected_id: Optional[str]   = Field(default=None, foreign_key="corpus.es_id")
    course:      Optional[str]   = None
    topic:       Optional[int]   = None
    subtopic:    Optional[int]   = None
    query_type:  Optional[str]   = None
    hit_ids:     Optional[str]   = None  # JSON-encoded list
    hit_scores:  Optional[str]   = None  # JSON-encoded list
    latency_ms:  Optional[float] = None
    hit_at_1:    bool            = False
    hit_at_5:    bool            = False
    rank:        Optional[int]   = None

    @property
    def hit_ids_list(self) -> list[str]:
        return json.loads(self.hit_ids) if self.hit_ids else []

    @property
    def hit_scores_list(self) -> list[float]:
        return json.loads(self.hit_scores) if self.hit_scores else []

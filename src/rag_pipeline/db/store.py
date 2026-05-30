"""
Conversion layer: core/models → db/models.

Single entry point:
    from rag_pipeline.db.store import save_experiment_result
    save_experiment_result(result, jsonl_path)
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from sqlmodel import select
from rag_pipeline.db.engine import get_session, init_db
from rag_pipeline.db.models import Run, RunMetrics, QueryResult

if TYPE_CHECKING:
    from rag_pipeline.core.models.ablation import ExperimentResult
    from rag_pipeline.ingestion.benchmark_types import MetricSummary


def _run_id(experiment: str, config: str) -> str:
    return f"{experiment}__{config}"


def _upsert_run(session, result: "ExperimentResult", config: str) -> Run:
    run_id = _run_id(result.name, config)
    existing = session.get(Run, run_id)
    if existing:
        session.delete(existing)
        session.commit()
    run = Run(
        run_id=run_id,
        experiment=result.name,
        patch=result.patch,
        config=config,
        model=result.model,
        git_commit=result.git_commit,
        timestamp=datetime.fromisoformat(result.timestamp) if result.timestamp else datetime.now(timezone.utc),
        corpus_size=result.corpus_size,
    )
    session.add(run)
    return run


def _upsert_metrics(session, run_id: str, m: "MetricSummary") -> None:
    existing = session.get(RunMetrics, run_id)
    if existing:
        session.delete(existing)
        session.commit()
    fields = m.model_dump()
    # drop fields not in RunMetrics
    fields.pop("config_name", None)
    fields.pop("model_name", None)
    fields.pop("topic", None)
    fields.pop("subtopic", None)
    session.add(RunMetrics(run_id=run_id, **fields))


def _load_query_results(jsonl_path: Path, run_id: str) -> list[QueryResult]:
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            hit_ids = r.get("hit_ids", [])
            expected = r.get("expected_id", "")
            rank = next((i + 1 for i, h in enumerate(hit_ids) if h == expected), None)
            rows.append(QueryResult(
                run_id=run_id,
                query_id=r.get("query_id", ""),
                query_text=r.get("query_text", ""),
                expected_id=expected or None,
                course=r.get("course"),
                topic=r.get("topic"),
                subtopic=r.get("subtopic"),
                query_type=r.get("query_type"),
                hit_ids=json.dumps(hit_ids),
                hit_scores=json.dumps(r.get("hit_scores", [])),
                latency_ms=r.get("latency_ms"),
                hit_at_1=bool(hit_ids and hit_ids[0] == expected),
                hit_at_5=expected in hit_ids[:5],
                rank=rank,
            ))
    return rows


def _delete_query_results(session, run_id: str) -> None:
    rows = session.exec(select(QueryResult).where(QueryResult.run_id == run_id)).all()
    for r in rows:
        session.delete(r)


def save_experiment_result(
    result: "ExperimentResult",
    ablation_results_dir: Path,
) -> None:
    """Persist an ExperimentResult and its per-query JSONLs to the DB."""
    init_db()
    with get_session() as session:
        for config, metrics in result.metrics.items():
            run_id = _run_id(result.name, config)
            run = _upsert_run(session, result, config)
            session.commit()

            _upsert_metrics(session, run_id, metrics)
            session.commit()

            jsonl = ablation_results_dir / f"{result.name}__{config}_query_results.jsonl"
            if jsonl.exists():
                _delete_query_results(session, run_id)
                session.commit()
                for qr in _load_query_results(jsonl, run_id):
                    session.add(qr)
                session.commit()

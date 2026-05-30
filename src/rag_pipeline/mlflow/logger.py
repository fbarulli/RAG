"""
MLflow logging for benchmark runs.

Usage:
    from rag_pipeline.ingestion.mlflow_logger import log_benchmark_run
    log_benchmark_run(cfg_name, summary, results, model_entry, config)
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import mlflow
from rag_pipeline.core.paths import Paths

if TYPE_CHECKING:
    from rag_pipeline.ingestion.benchmark_types import MetricSummary, QueryResult

EXPERIMENT_NAME = "rag-retrieval"


def _get_or_create_experiment() -> str:
    mlflow.set_tracking_uri(f"sqlite:///{Paths.mlflow_db()}")
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        return mlflow.create_experiment(EXPERIMENT_NAME)
    return exp.experiment_id


def log_benchmark_run(
    cfg_name: str,
    summary: "MetricSummary",
    results: "list[QueryResult]",
    model_entry: dict,
    tags: dict | None = None,
) -> None:
    exp_id = _get_or_create_experiment()
    with mlflow.start_run(experiment_id=exp_id, run_name=f"{model_entry['name']}__{cfg_name}"):
        # Tags
        mlflow.set_tag("config", cfg_name)
        mlflow.set_tag("model", model_entry["name"])
        mlflow.set_tag("collection", model_entry.get("collection", ""))
        for k, v in (tags or {}).items():
            mlflow.set_tag(k, v)

        # Aggregate metrics
        mlflow.log_metrics({
            "h1":              summary.hit_rate_1,
            "h3":              summary.hit_rate_3,
            "h5":              summary.hit_rate_5,
            "h10":             summary.hit_rate_10,
            "mrr":             summary.mrr,
            "ndcg_10":         summary.ndcg_10,
            "latency_p50":     summary.latency_p50,
            "latency_p95":     summary.latency_p95,
            "failure_rate":    summary.failure_count / summary.num_queries if summary.num_queries else 0.0,
            "cross_course":    summary.cross_course_contamination,
            "rank_std":        summary.rank_std,
        })

        # Per-query artifact
        import json, tempfile, os
        rows = []
        for r in results:
            rows.append({
                "query_id":            r.query_id,
                "query_text":          r.query_text,
                "expected_id":         r.expected_id,
                "course":              r.course,
                "topic":               r.topic,
                "subtopic":            r.subtopic,
                "query_type":          r.query_type,
                "ner_primary_entity":  r.ner_primary_entity,
                "ner_entities":        list(r.ner_entities),
                "rank":                r.rank,
                "hit_at_1":            r.hit_at_1,
                "hit_at_3":            r.hit_at_3,
                "hit_at_5":            r.hit_at_5,
                "hit_ids":             list(r.hit_ids),
                "hit_scores":          list(r.hit_scores),
                "latency_ms":          r.latency_ms,
            })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
            tmp = f.name
        mlflow.log_artifact(tmp, artifact_path="per_query")
        os.unlink(tmp)

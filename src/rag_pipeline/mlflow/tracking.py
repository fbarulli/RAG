"""
Unified MLflow tracking for all RAG-a-muffin experiments.

Usage:
    from rag_pipeline.mlflow.tracking import log_benchmark_run, log_ablation_run
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import mlflow
from rag_pipeline.core.paths import Paths

if TYPE_CHECKING:
    from rag_pipeline.ingestion.benchmark_types import MetricSummary, QueryResult
    from rag_pipeline.core.models.ablation import ExperimentResult

EXPERIMENT_BENCHMARK = "rag-retrieval"
EXPERIMENT_ABLATION  = "rag-ablation"

mlflow.set_tracking_uri(f"sqlite:///{Paths.mlflow_db()}")


def _get_or_create_experiment(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    return exp.experiment_id if exp else mlflow.create_experiment(name)


def _dedup(exp_id: str, run_name: str) -> None:
    client = mlflow.tracking.MlflowClient()
    existing = client.search_runs(
        exp_id, filter_string=f"tags.`mlflow.runName` = '{run_name}'"
    )
    for old in existing:
        client.delete_run(old.info.run_id)


def log_benchmark_run(
    cfg_name: str,
    cfg_dict: dict,
    summary: "MetricSummary",
    results: "list[QueryResult]",
    model_entry: dict,
    encode_mode: str,
    tags: dict | None = None,
) -> None:
    exp_id   = _get_or_create_experiment(EXPERIMENT_BENCHMARK)
    run_name = f"{model_entry['name']}__{cfg_name}"
    _dedup(exp_id, run_name)

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
        # --- identity tags ---
        mlflow.set_tag("model",       model_entry["name"])
        mlflow.set_tag("config",      cfg_name)
        mlflow.set_tag("collection",  model_entry.get("collection", ""))
        mlflow.set_tag("encode_mode", encode_mode)
        mlflow.set_tag("tier",        model_entry.get("tier", ""))
        for k, v in (tags or {}).items():
            mlflow.set_tag(k, v)

        # --- retrieval config as params (full audit trail) ---
        for k, v in cfg_dict.items():
            mlflow.log_param(k, v)
        mlflow.log_param("model_dims", model_entry.get("dims", ""))

        # --- aggregate metrics ---
        failure_rate = (
            summary.failure_count / summary.num_queries
            if summary.num_queries else 0.0
        )
        mlflow.log_metrics({
            "h1":           summary.hit_rate_1,
            "h3":           summary.hit_rate_3,
            "h5":           summary.hit_rate_5,
            "h10":          summary.hit_rate_10,
            "mrr":          summary.mrr,
            "ndcg_1":       summary.ndcg_1,
            "ndcg_5":       summary.ndcg_5,
            "ndcg_10":      summary.ndcg_10,
            "map_score":    summary.map_score,
            "latency_p50":  summary.latency_p50,
            "latency_p95":  summary.latency_p95,
            "latency_p99":  summary.latency_p99,
            "failure_rate": failure_rate,
            "cross_course": summary.cross_course_contamination,
            "rank_std":     summary.rank_std,
        })

        # --- per-query artifact ---
        import json, tempfile, os
        rows = [
            {
                "query_id":           r.query_id,
                "query_text":         r.query_text,
                "expected_id":        r.expected_id,
                "course":             r.course,
                "topic":              r.topic,
                "subtopic":           r.subtopic,
                "query_type":         r.query_type,
                "ner_primary_entity": r.ner_primary_entity,
                "ner_entities":       list(r.ner_entities),
                "rank":               r.rank,
                "hit_at_1":           r.hit_at_1,
                "hit_at_3":           r.hit_at_3,
                "hit_at_5":           r.hit_at_5,
                "hit_ids":            list(r.hit_ids),
                "hit_scores":         list(r.hit_scores),
                "latency_ms":         r.latency_ms,
            }
            for r in results
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
            tmp = f.name
        mlflow.log_artifact(tmp, artifact_path="per_query")
        os.unlink(tmp)


def log_ablation_run(
    result: "ExperimentResult",
    tags: dict | None = None,
) -> None:
    exp_id = _get_or_create_experiment(EXPERIMENT_ABLATION)

    for cfg_name, summary in result.metrics.items():
        run_name = f"ablation__{result.name}__{cfg_name}"
        _dedup(exp_id, run_name)

        with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
            mlflow.set_tag("experiment",  result.name)
            mlflow.set_tag("patch",       result.patch)
            mlflow.set_tag("config",      cfg_name)
            mlflow.set_tag("model",       result.model)
            mlflow.set_tag("git_commit",  result.git_commit)
            mlflow.set_tag("encode_mode", "qa")  # ablation always uses production encode_mode
            if result.corpus_size is not None:
                mlflow.set_tag("corpus_size", str(result.corpus_size))
            for k, v in (tags or {}).items():
                mlflow.set_tag(k, v)

            metrics_to_log = {
                k: v for k, v in summary.model_dump().items()
                if isinstance(v, (int, float))
            }
            if metrics_to_log:
                mlflow.log_metrics(metrics_to_log)

            jsonl = (
                Paths.ablation_results_dir()
                / f"{result.name}__{cfg_name}_query_results.jsonl"
            )
            if jsonl.exists():
                mlflow.log_artifact(str(jsonl), artifact_path="per_query")

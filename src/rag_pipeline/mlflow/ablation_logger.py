"""
MLflow logging for ablation runs.

Usage:
    from rag_pipeline.mlflow.ablation_logger import log_ablation_run
    log_ablation_run(result)
"""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
from rag_pipeline.core.paths import Paths

if TYPE_CHECKING:
    from rag_pipeline.core.models.ablation import ExperimentResult

EXPERIMENT_NAME = "rag-ablation"


def _get_or_create_experiment() -> str:
    mlflow.set_tracking_uri(f"sqlite:///{Paths.mlflow_db()}")
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        return mlflow.create_experiment(EXPERIMENT_NAME)
    return exp.experiment_id


def log_ablation_run(
    result: "ExperimentResult",
    tags: dict | None = None,
) -> None:
    exp_id = _get_or_create_experiment()
    for cfg, m in result.metrics.items():
        run_name = f"ablation__{result.name}__{cfg}"
        with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
            mlflow.set_tag("experiment", result.name)
            mlflow.set_tag("patch", result.patch)
            mlflow.set_tag("config", cfg)
            mlflow.set_tag("model", result.model)
            mlflow.set_tag("git_commit", result.git_commit)
            mlflow.set_tag("timestamp", result.timestamp)
            if result.corpus_size is not None:
                mlflow.set_tag("corpus_size", str(result.corpus_size))
            for k, v in (tags or {}).items():
                mlflow.set_tag(k, v)

            metrics_to_log = {k: v for k, v in m.model_dump().items() if isinstance(v, (int, float))}
            if metrics_to_log:
                mlflow.log_metrics(metrics_to_log)

            jsonl = Paths.ablation_results_dir() / f"{result.name}__{cfg}_query_results.jsonl"
            if jsonl.exists():
                mlflow.log_artifact(str(jsonl), artifact_path="per_query")

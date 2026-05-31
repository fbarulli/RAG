# rag_pipeline/ablation/experiment.py
"""
Run a named ablation experiment.

Usage:
    uv run python -m rag_pipeline.ablation run --name baseline
    uv run python -m rag_pipeline.ablation run --name no_category --null-category
    uv run python -m rag_pipeline.ablation run --name no_entity --null-entity
    uv run python -m rag_pipeline.ablation run --name no_cluster --skip-cluster
    uv run python -m rag_pipeline.ablation run --name no_rules --skip-rules
    uv run python -m rag_pipeline.ablation run --name no_ner --skip-ner
    uv run python -m rag_pipeline.ablation run --name no_topics --null-topics
"""
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel, Field

from rag_pipeline.core.models import Patch, ExperimentResult, GENERIC_ENTITIES, EncodeMode
from rag_pipeline.ingestion.benchmark_types import MetricSummary
from rag_pipeline.core.paths import Paths
from rag_pipeline.logging import get_logger
from rag_pipeline.mlflow.tracking import log_ablation_run
from rag_pipeline.db.store import save_experiment_result

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class Experiment(BaseModel):
    name: str
    patch: Patch
    configs: list[str] = Field(default_factory=lambda: [Paths.defaults()["production_config"]])
    model: str         = Field(default_factory=lambda: Paths.defaults()["production_model"])
    encode_mode: EncodeMode = Field(default_factory=lambda: EncodeMode(Paths.defaults().get("production_encode_mode", "question")))

    def run(self) -> ExperimentResult:
        logger.info("Running experiment: %s (patch=%s)", self.name, self.patch.label())

        assignments_path    = Paths.topic_assignments()
        entity_patterns_path = Paths.entity_patterns()
        with open(assignments_path, encoding="utf-8") as f:
            original_assignments = json.load(f)
        original_patterns    = entity_patterns_path.read_text(encoding="utf-8")

        try:
            if self.patch.needs_rerun:
                self._run_with_rerun(entity_patterns_path)
            else:
                self._run_payload_only(assignments_path, original_assignments)

            cfg_args = " ".join(self.configs)
            collection = Paths.collection_for_model(self.model, self.encode_mode)
            _run(
                f'uv run python -m rag_pipeline.ingestion.benchmark '
                f'--model "{self.model}" --configs {cfg_args} --collection "{collection}"'
            )
            metrics, result_files = _collect_results(self.name, self.configs, self.model)

        finally:
            # always restore original state
            with open(assignments_path, "w", encoding="utf-8") as f:
                json.dump(original_assignments, f, indent=2)
            entity_patterns_path.write_text(original_patterns, encoding="utf-8")
            logger.info("Restored original state")

        git_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=Paths.base(),
        )
        git_commit = git_result.stdout.strip() if git_result.returncode == 0 else "unknown"

        try:
            import json as _json
            corpus_size = sum(
                1 for _ in open(Paths.clean_jsonl(), encoding="utf-8")
            )
        except Exception:
            corpus_size = None

        result = ExperimentResult(
            name=self.name,
            patch=self.patch.label(),
            configs=self.configs,
            model=self.model,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            result_files=result_files,
            git_commit=git_commit,
            corpus_size=corpus_size,
        )
        meta_path = Paths.ablation_results_dir() / f"{self.name}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)
        logger.info("Saved → %s", meta_path)

        if self.name == "baseline":
            baseline_path = Paths.experiments_dir() / "baseline.json"
            with open(baseline_path, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, indent=2)
            logger.info("Baseline snapshot → %s", baseline_path)

        log_ablation_run(result)
        save_experiment_result(result, Paths.ablation_results_dir())
        return result

    def _run_payload_only(self, assignments_path: Path, original: dict) -> None:
        patched = json.loads(json.dumps(original))
        patched["results"][self.model]["assignments"] = self.patch.apply_to_assignments(
            patched["results"][self.model]["assignments"]
        )
        with open(assignments_path, "w", encoding="utf-8") as f:
            json.dump(patched, f, indent=2)
        _run(f'uv run python -m rag_pipeline.ingestion.ingest_models --models "{self.model}" --encode-mode "{self.encode_mode.value}" --no-skip-existing')

    def _run_with_rerun(self, entity_patterns_path: Path) -> None:
        if self.patch.empty_entity_patterns:
            entity_patterns_path.write_text(
                json.dumps({"TOOL": [], "ERROR": [], "CONCEPT": [], "LANGUAGE": [], "ADMIN": []}),
                encoding="utf-8",
            )
        env  = {**os.environ, **self.patch.env()}
        slug = self.model.replace("/", "_").replace("-", "_")
        out  = Paths.topics_experiments_dir() / f"topic_assignments_{slug}.json"
        cluster_flag = "--skip-cluster" if self.patch.skip_cluster else ""
        rules_flag   = "--skip-rules"   if self.patch.skip_rules   else ""
        _run(
            f'uv run python -m rag_pipeline.eda.topics.core.topic_modeling '
            f'--embedding-model "{self.model}" --output "{out}" '
            f'{cluster_flag} {rules_flag}'.strip()
        )
        _run(
            f'uv run python -m rag_pipeline.eda.topics.core.topic_merge --only "{out}"',
            env=env,
        )
        _run(f'uv run python -m rag_pipeline.ingestion.ingest_models --models "{self.model}" --encode-mode "{self.encode_mode.value}" --no-skip-existing')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_from_git(path: Path) -> dict:
    rel    = path.relative_to(Paths.base())
    result = subprocess.run(
        ["git", "show", f"HEAD:{rel}"],
        capture_output=True, text=True, cwd=Paths.base(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"git show failed: {result.stderr}")
    return json.loads(result.stdout)


def _run(cmd: str, env: dict = None) -> None:
    result = subprocess.run(cmd, shell=True, cwd=Paths.base(), env=env or os.environ)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def _collect_results(name: str, configs: list[str], model: str) -> tuple[dict, list[str]]:
    bench_dir    = Paths.reranker_results_dir()
    metrics      = {}
    summary_path = bench_dir / "benchmark_results.json"

    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            if row.get("model_name") == model and row.get("config_name") in configs:
                metrics[row["config_name"]] = MetricSummary.from_benchmark_row(row)

    result_files = []
    results_dir  = Paths.ablation_results_dir()
    for cfg in configs:
        src = bench_dir / f"{cfg}_query_results.jsonl"
        if src.exists():
            dst = results_dir / f"{name}__{cfg}_query_results.jsonl"
            shutil.copy2(src, dst)
            result_files.append(str(dst))

    return metrics, result_files
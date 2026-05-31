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
from rag_pipeline.core.io import atomic_json_write

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
        import logging
        log_path = Paths.ablation_results_dir() / f"{self.name}_run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logging.getLogger("rag_pipeline").addHandler(fh)
        try:
            return self._run_inner()
        finally:
            logging.getLogger("rag_pipeline").removeHandler(fh)
            fh.close()

    def _run_inner(self) -> ExperimentResult:
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
                f'--model "{self.model}" --configs {cfg_args} --collection "{collection}"',
                run_id=self.name
            )
            metrics, result_files = _collect_results(self.name, self.configs, self.model)

        finally:
            # always restore original state
            atomic_json_write(assignments_path, original_assignments, indent=2)
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
        atomic_json_write(meta_path, result.model_dump(), indent=2)
        logger.info("Saved → %s", meta_path)

        if self.name == "baseline":
            baseline_path = Paths.experiments_dir() / "baseline.json"
            atomic_json_write(baseline_path, result.model_dump(), indent=2)
            logger.info("Baseline snapshot → %s", baseline_path)

        log_ablation_run(result)
        save_experiment_result(result, Paths.ablation_results_dir())
        return result

    def _run_payload_only(self, assignments_path: Path, original: dict) -> None:
        patched = json.loads(json.dumps(original))
        patched["results"][self.model]["assignments"] = self.patch.apply_to_assignments(
            patched["results"][self.model]["assignments"]
        )
        atomic_json_write(assignments_path, patched, indent=2)
        # verify patch was applied
        _verify_assignments(assignments_path, self.model, self.patch)
        _run(f'uv run python -m rag_pipeline.ingestion.ingest_models --models "{self.model}" --encode-mode "{self.encode_mode.value}" --no-skip-existing', run_id=self.name)
        # verify ingest reflected patch
        _verify_collection(self.model, self.encode_mode, self.patch)

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
            f'{cluster_flag} {rules_flag}'.strip(),
            run_id=self.name
        )
        _run(
            f'uv run python -m rag_pipeline.eda.topics.core.topic_merge --only "{out}"',
            env=env,
            run_id=self.name
        )
        _run(f'uv run python -m rag_pipeline.ingestion.ingest_models --models "{self.model}" --encode-mode "{self.encode_mode.value}" --no-skip-existing', run_id=self.name)


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


def _verify_assignments(path: Path, model: str, patch: 'Patch') -> None:
    import json
    data = json.load(open(path))
    assignments = data['results'][model]['assignments']
    has_primary = sum(1 for a in assignments if a.get('ner_primary_entity'))
    has_entities = sum(1 for a in assignments if a.get('ner_entities'))
    logger.info("Assignments after patch: has_primary=%d has_entities=%d total=%d", has_primary, has_entities, len(assignments))
    if patch.null_entity and (has_primary > 0 or has_entities > 0):
        raise RuntimeError(f"Patch null_entity failed: has_primary={has_primary} has_entities={has_entities}")

def _verify_collection(model: str, encode_mode, patch: 'Patch') -> None:
    from qdrant_client import QdrantClient
    from rag_pipeline.core.paths import Paths
    collection = Paths.collection_for_model(model, encode_mode)
    client = QdrantClient(host='localhost', port=6333)
    result = client.scroll(collection, limit=100, with_payload=True, with_vectors=False)
    points = result[0]
    has_primary = sum(1 for p in points if p.payload.get('ner_primary_entity'))
    has_entities = sum(1 for p in points if p.payload.get('ner_entities'))
    logger.info("Collection after ingest (sample 100): has_primary=%d has_entities=%d", has_primary, has_entities)
    if patch.null_entity and (has_primary > 0 or has_entities > 0):
        raise RuntimeError(f"Collection still has entity data after null_entity patch: has_primary={has_primary} has_entities={has_entities}")

def _run(cmd: str, env: dict = None, run_id: str = "unknown") -> None:
    log_path = Paths.ablation_results_dir() / f"{run_id}_subprocess.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, shell=True, cwd=Paths.base(), env=env or os.environ, capture_output=True, text=True)
    with open(log_path, "a") as f:
        f.write(f"\n--- CMD: {cmd}\n")
        f.write(f"--- RC: {result.returncode}\n")
        if result.stdout: f.write(result.stdout)
        if result.stderr: f.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (rc={result.returncode}): {cmd}\nSTDERR: {result.stderr[-2000:]}")


def _collect_results(name: str, configs: list[str], model: str) -> tuple[dict, list[str]]:
    bench_dir = Paths.reranker_results_dir()
    metrics = {}

    # primary: read from benchmark_results.json written by benchmark.py
    summary_path = bench_dir / "benchmark_results.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            if row.get("model_name") == model and row.get("config_name") in configs:
                metrics[row["config_name"]] = MetricSummary.from_benchmark_row(row)

    # fallback: read from DB if benchmark_results.json missing
    if not metrics:
        import sqlite3
        try:
            con = sqlite3.connect(str(Paths.results_db()))
            con.row_factory = sqlite3.Row
            for cfg in configs:
                run_id = f"{name}__{cfg}"
                row = con.execute(
                    "SELECT r.config, r.model, rm.* FROM runs r JOIN run_metrics rm ON r.run_id=rm.run_id WHERE r.run_id=?",
                    (run_id,)
                ).fetchone()
                if row:
                    d = dict(row)
                    d["config_name"] = d.pop("config", cfg)
                    d["model_name"]  = d.pop("model", model)
                    metrics[cfg] = MetricSummary.from_benchmark_row(d)
            con.close()
        except Exception as e:
            logger.error("DB read failed: %s", e)

    if not metrics:
        logger.warning("No metrics found for %s configs=%s", name, configs)

    result_files = []
    results_dir = Paths.ablation_results_dir()
    for cfg in configs:
        src = bench_dir / f"{cfg}_query_results.jsonl"
        if src.exists():
            dst = results_dir / f"{name}__{cfg}_query_results.jsonl"
            shutil.copy2(src, dst)
            result_files.append(str(dst))

    return metrics, result_files
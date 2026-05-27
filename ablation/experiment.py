
# ablation/experiment.py
"""
Run a named ablation experiment.

Usage:
    uv run python -m ablation run --name baseline
    uv run python -m ablation run --name no_category --null-category
    uv run python -m ablation run --name no_entity --null-entity
    uv run python -m ablation run --name no_cluster --skip-cluster
    uv run python -m ablation run --name no_rules --skip-rules
    uv run python -m ablation run --name no_ner --skip-ner
    uv run python -m ablation run --name no_topics --null-topics
"""
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))

from src.rag_pipeline.core.paths import Paths
from src.rag_pipeline.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Patch:
    # payload patches (no re-run needed)
    null_entity: bool = False
    null_category: bool = False
    null_topics: bool = False
    # ner patches (require topic modeling re-run)
    skip_ner: bool = False            # null entity + category before assignments built
    empty_entity_patterns: bool = False  # wipe entity_patterns.json before topic modeling
    # reclassify patches (env var flags, require topic modeling re-run)
    skip_cluster: bool = False        # ABLATION_SKIP_CLUSTER
    skip_rules: bool = False          # ABLATION_SKIP_RULES

    @property
    def needs_rerun(self) -> bool:
        return self.skip_ner or self.empty_entity_patterns or self.skip_cluster or self.skip_rules

    def apply_to_assignments(self, assignments: list[dict]) -> list[dict]:
        for a in assignments:
            if self.null_entity or self.skip_ner:
                a["ner_primary_entity"] = None
            if self.null_category or self.skip_ner:
                a["ner_category"] = "OTHER"
            if self.null_topics:
                a["topic"] = -1
        return assignments

    def env(self) -> dict:
        e = {}
        if self.skip_cluster:
            e["ABLATION_SKIP_CLUSTER"] = "1"
        if self.skip_rules:
            e["ABLATION_SKIP_RULES"] = "1"
        return e

    def label(self) -> str:
        parts = []
        if self.null_entity:          parts.append("no_entity")
        if self.null_category:        parts.append("no_category")
        if self.null_topics:          parts.append("no_topics")
        if self.skip_ner:             parts.append("skip_ner")
        if self.empty_entity_patterns: parts.append("empty_patterns")
        if self.skip_cluster:         parts.append("skip_cluster")
        if self.skip_rules:           parts.append("skip_rules")
        return "+".join(parts) if parts else "baseline"


@dataclass
class ExperimentResult:
    name: str
    patch: str
    configs: list[str]
    model: str
    timestamp: str
    metrics: dict
    result_files: list[str]


@dataclass
class Experiment:
    name: str
    patch: Patch
    configs: list[str] = field(default_factory=lambda: ["entity_boosted", "vector_default"])
    model: str = "BAAI/bge-base-en-v1.5"

    def run(self) -> ExperimentResult:
        logger.info("Running experiment: %s (patch=%s)", self.name, self.patch.label())

        assignments_path = Paths.topic_assignments()
        entity_patterns_path = Paths.entity_patterns()
        original_assignments = _load_from_git(assignments_path)
        original_patterns = entity_patterns_path.read_text(encoding="utf-8")

        try:
            if self.patch.needs_rerun:
                self._run_with_rerun(entity_patterns_path)
            else:
                self._run_payload_only(assignments_path, original_assignments)

            cfg_args = " ".join(self.configs)
            _run(f'uv run python -m rag_pipeline.ingestion.p03_benchmark '
                 f'--model "{self.model}" --configs {cfg_args}')
            metrics, result_files = _collect_results(self.name, self.configs, self.model)

        finally:
            # always restore
            with open(assignments_path, "w", encoding="utf-8") as f:
                json.dump(original_assignments, f, indent=2)
            entity_patterns_path.write_text(original_patterns, encoding="utf-8")
            logger.info("Restored original state")

        result = ExperimentResult(
            name=self.name,
            patch=self.patch.label(),
            configs=self.configs,
            model=self.model,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            result_files=result_files,
        )
        meta_path = RESULTS_DIR / f"{self.name}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)
        logger.info("Saved → %s", meta_path)
        return result

    def _run_payload_only(self, assignments_path: Path, original: dict) -> None:
        patched = json.loads(json.dumps(original))
        patched["results"][self.model]["assignments"] = self.patch.apply_to_assignments(
            patched["results"][self.model]["assignments"]
        )
        with open(assignments_path, "w", encoding="utf-8") as f:
            json.dump(patched, f, indent=2)
        _run(f'uv run python -m src.rag_pipeline.ingestion.p00_ingest_qdrant --model "{self.model}"')

    def _run_with_rerun(self, entity_patterns_path: Path) -> None:
        if self.patch.empty_entity_patterns:
            entity_patterns_path.write_text(
                json.dumps({"TOOL": [], "ERROR": [], "CONCEPT": [], "LANGUAGE": [], "ADMIN": []}),
                encoding="utf-8"
            )
        env = {**os.environ, **self.patch.env()}
        slug = self.model.replace("/", "_").replace("-", "_")
        out = ROOT / "src/rag_pipeline/eda/topics/experiments" / f"topic_assignments_{slug}.json"
        _run(
            f'uv run python -m rag_pipeline.eda.topics.core.topic_modeling '
            f'--embedding-model "{self.model}" --output "{out}"',
            env=env
        )
        _run('uv run python -c "from src.rag_pipeline.eda.topics.core.topic_merge import TopicMerger; TopicMerger().merge()"', env=env)
        _run(f'uv run python -m src.rag_pipeline.ingestion.p00_ingest_qdrant --model "{self.model}"')


def _load_from_git(path: Path) -> dict:
    rel = path.relative_to(ROOT)
    result = subprocess.run(["git", "show", f"HEAD:{rel}"],
                            capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"git show failed: {result.stderr}")
    return json.loads(result.stdout)


def _run(cmd: str, env: dict = None) -> None:
    result = subprocess.run(cmd, shell=True, cwd=ROOT, env=env or os.environ)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def _collect_results(name: str, configs: list[str], model: str) -> tuple[dict, list[str]]:
    bench_dir = ROOT / "experiments" / "reranker_benchmarks"
    metrics = {}
    summary_path = bench_dir / "benchmark_results.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            if row.get("model_name") == model and row.get("config_name") in configs:
                metrics[row["config_name"]] = {
                    "h1":  row.get("hit_rate_1"),
                    "h3":  row.get("hit_rate_3"),
                    "h5":  row.get("hit_rate_5"),
                    "h10": row.get("hit_rate_10"),
                    "mrr": row.get("mrr"),
                    "p50_ms": row.get("latency_p50"),
                }
    result_files = []
    for cfg in configs:
        src = bench_dir / f"{cfg}_query_results.jsonl"
        if src.exists():
            dst = RESULTS_DIR / f"{name}__{cfg}_query_results.jsonl"
            shutil.copy2(src, dst)
            result_files.append(str(dst))
    return metrics, result_files

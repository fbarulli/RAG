# src/rag_pipeline/eda/topics/core/topic_merge.py
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from rag_pipeline.core.paths import Paths

logger = logging.getLogger(__name__)


def _find_model_files(exp_dir: Path) -> List[Path]:
    return sorted(
        f for f in exp_dir.glob("topic_assignments_*.json")
        if f.name != "topic_assignments_all.json"
    )


def _load_model_file(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_merged(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


class TopicMerger:
    """Merges per-model assignment files into a single output."""

    def merge(self, force: bool = False) -> Dict[str, Any]:
        exp_dir = Paths.topics_experiments_dir()
        output_path = Paths.topic_assignments()

        files = _find_model_files(exp_dir)
        logger.info("Found %d model assignment files", len(files))

        merged: Dict[str, Any] = {"metadata": {"models_merged": []}, "results": {}}

        for f in files:
            try:
                data = _load_model_file(f)
            except Exception:
                logger.error("Failed to load %s", f, exc_info=True)
                continue
            model = data.get("metadata", {}).get("model", f.stem)
            merged["metadata"]["models_merged"].append(model)
            merged["results"][model] = data

        _write_merged(merged, output_path)
        logger.info("Merged %d models → %s", len(files), output_path)
        return merged
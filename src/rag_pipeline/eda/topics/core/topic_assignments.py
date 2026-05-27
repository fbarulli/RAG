# src/rag_pipeline/eda/topics/core/topic_assignments.py
import json
import logging
from typing import Any, Dict, List

from src.rag_pipeline.core.paths import Paths

logger = logging.getLogger(__name__)


def _load_json(path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _iter_assignments(data: Dict[str, Any]):
    for model_data in data.get("results", {}).values():
        yield from model_data.get("assignments", [])


class TopicAssignments:
    """Loads and queries the merged topic assignments file."""

    def load_merged(self) -> Dict[str, Any]:
        path = Paths.topic_assignments()
        if not path.exists():
            logger.warning("Merged assignments file not found: %s", path)
            return {}
        try:
            data = _load_json(path)
            logger.info("Loaded merged assignments from %s", path)
            return data
        except Exception:
            logger.error("Failed to load merged assignments", exc_info=True)
            return {}

    def get_sample(self, n: int = 3) -> List[Dict[str, Any]]:
        data = self.load_merged()
        results: List[Dict[str, Any]] = []
        for assignment in _iter_assignments(data):
            results.append(assignment)
            if len(results) >= n:
                break
        return results
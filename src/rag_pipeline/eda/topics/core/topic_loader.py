# src/rag_pipeline/eda/topics/core/topic_loader.py
import json
import logging
from typing import Any, Dict, List

from rag_pipeline.eda.core.paths import Paths

logger = logging.getLogger(__name__)


def _model_assignment_path(model_name: str):
    safe = model_name.replace("/", "_").replace("-", "_")
    return Paths.topics_experiments_dir() / f"topic_assignments_{safe}.json"


class TopicLoader:
    """Loads raw data and per-model topic assignments."""

    def load_clean_data(self) -> List[Dict[str, Any]]:
        path = Paths.input_file("eda")
        if not path.exists():
            logger.error("Clean data not found: %s", path)
            return []
        try:
            with open(path, encoding="utf-8") as f:
                data = [json.loads(line) for line in f if line.strip()]
            logger.info("Loaded %d documents from %s", len(data), path)
            return data
        except Exception:
            logger.error("Failed to load clean data", exc_info=True)
            return []

    def load_previous_assignments(self, model_name: str) -> Dict[str, Any]:
        path = _model_assignment_path(model_name)
        if not path.exists():
            logger.warning("No previous assignments for %s at %s", model_name, path)
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.error("Failed to load assignments for %s", model_name, exc_info=True)
            return {}
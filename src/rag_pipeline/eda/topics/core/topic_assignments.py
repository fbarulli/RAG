import sys
import logging
import json
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parents[5]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag_pipeline.eda.topics.config import TopicsConfig
from src.rag_pipeline.core.paths import Paths


class TopicAssignments:
    """Handles loading, saving, and enriching topic assignments."""
    
    def __init__(self):
        self.config = TopicsConfig
        logger.info("TopicAssignments handler initialized")
    
    def load_merged(self) -> Dict[str, Any]:
        """Load the final merged assignments."""
        path = self.config.get_output_path("topic_assignments_all.json")
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        logger.warning("Merged assignments file not found")
        return {}
    
    def get_sample(self, n: int = 3) -> List[Dict]:
        """Return sample assignments for inspection."""
        data = self.load_merged()
        assignments = []
        for model_data in data.get("results", {}).values():
            assignments.extend(model_data.get("assignments", []))
            if len(assignments) >= n:
                break
        return assignments[:n]

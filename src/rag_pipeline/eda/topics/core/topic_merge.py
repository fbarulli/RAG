import sys
import logging
import json
from pathlib import Path
from collections import Counter
from typing import Dict, Any

# Setup logging + project root
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parents[5]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag_pipeline.eda.topics.config import TopicsConfig
from src.rag_pipeline.core.paths import Paths


class TopicMerger:
    """Merges per-model assignments and applies final rules."""
    
    def __init__(self):
        self.config = TopicsConfig
        logger.info("TopicMerger initialized")
    
    def merge(self, force: bool = False) -> Dict[str, Any]:
        """Merge all model results into one file."""
        logger.info("Starting merge process...")
        
        exp_dir = self.config.EXPERIMENTS_DIR
        output_path = self.config.get_output_path("topic_assignments_all.json")
        
        # TODO: Load rules and reclassify (to be implemented next)
        merged = {
            "metadata": {"models_merged": []},
            "results": {}
        }
        
        # Find all per-model files
        files = sorted(
            f for f in exp_dir.glob("topic_assignments_*.json")
            if f.name != "topic_assignments_all.json"
        )
        
        logger.info(f"Found {len(files)} model assignment files")
        
        for f in files:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            model = data.get('metadata', {}).get('model', f.stem)
            merged["metadata"]["models_merged"].append(model)
            merged["results"][model] = data
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding="utf-8") as fh:
            json.dump(merged, fh, indent=2)
        
        logger.info(f"Merged {len(files)} models → {output_path}")
        return merged

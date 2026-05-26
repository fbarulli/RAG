import sys
import logging
import json
from pathlib import Path
from typing import List, Dict, Any

# Setup logging + project root
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parents[5]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag_pipeline.eda.topics.config import TopicsConfig
from src.rag_pipeline.core.paths import Paths


class TopicLoader:
    """Loads raw data and previous topic assignments."""
    
    def __init__(self):
        self.config = TopicsConfig
        logger.info("TopicLoader initialized")
    
    def load_clean_data(self) -> List[Dict[str, Any]]:
        """Load the main clean dataset using central Paths."""
        logger.info("Loading clean data from central Paths...")
        try:
            data_path = Paths.input_file("eda")
            logger.info(f"Loading from: {data_path}")
            
            if not data_path.exists():
                logger.error(f"File not found: {data_path}")
                return []
            
            with open(data_path, encoding="utf-8") as f:
                data = [json.loads(line) for line in f if line.strip()]
            
            logger.info(f"Successfully loaded {len(data)} documents")
            return data
        except Exception as e:
            logger.error("Failed to load clean data", exc_info=True)
            return []
    
    def load_previous_assignments(self, model_name: str) -> Dict[str, Any]:
        """Load existing topic assignments for a model."""
        logger.info(f"Loading previous assignments for {model_name}")
        path = self.config.EXPERIMENTS_DIR / f"topic_assignments_{model_name.replace('/', '_').replace('-', '_')}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        logger.warning(f"No previous assignments found for {model_name}")
        return {}

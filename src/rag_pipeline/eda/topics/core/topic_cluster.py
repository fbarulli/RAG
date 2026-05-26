import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag_pipeline.eda.topics.config import TopicsConfig


class TopicCluster:
    """Handles embedding + clustering logic."""
    
    def __init__(self, model_name: str):
        self.config = TopicsConfig
        self.model_name = model_name
        logger.info(f"TopicCluster initialized with model: {model_name}")
    
    def run_clustering(self, documents: List[Dict[str, Any]], force_recluster: bool = False) -> List[Dict[str, Any]]:
        """Run clustering on documents."""
        logger.info(f"Running clustering on {len(documents)} documents using {self.model_name}")
        # TODO: implement actual clustering later
        return documents

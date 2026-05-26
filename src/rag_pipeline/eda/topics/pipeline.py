import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag_pipeline.eda.topics.config import TopicsConfig
from src.rag_pipeline.eda.topics.core.topic_loader import TopicLoader
from src.rag_pipeline.eda.topics.core.topic_cluster import TopicCluster
from src.rag_pipeline.eda.topics.core.topic_merge import TopicMerger


class TopicsPipeline:
    """Main orchestrator for the topics pipeline."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.config = TopicsConfig
        self.model_name = model_name or self.config.DEFAULT_MODEL
        logger.info(f"TopicsPipeline initialized with model: {self.model_name}")

        self.loader = TopicLoader()
        self.clusterer = None
        self.merger = TopicMerger()

    def run_full_pipeline(self, force_recluster: bool = False) -> Dict[str, Any]:
        logger.info("Starting full topics pipeline")
        try:
            data = self.loader.load_clean_data()
            
            if not self.clusterer:
                self.clusterer = TopicCluster(self.model_name)
            clustered = self.clusterer.run_clustering(data, force_recluster)
            
            # Merge step
            merge_result = self.merger.merge()
            
            return {
                "status": "success",
                "model": self.model_name,
                "documents_processed": len(clustered),
                "models_merged": len(merge_result.get("metadata", {}).get("models_merged", []))
            }
        except Exception as e:
            logger.error("Pipeline failed", exc_info=True)
            raise

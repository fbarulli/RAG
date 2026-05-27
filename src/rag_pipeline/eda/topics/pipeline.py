# src/rag_pipeline/eda/topics/pipeline.py
import logging
from typing import Any, Dict, Optional

from src.rag_pipeline.eda.topics.config import TopicsConfig
from src.rag_pipeline.eda.topics.core.topic_loader import TopicLoader
from src.rag_pipeline.eda.topics.core.topic_cluster import TopicCluster
from src.rag_pipeline.eda.topics.core.topic_merge import TopicMerger

logger = logging.getLogger(__name__)


class TopicsPipeline:
    """Orchestrates load → cluster → merge."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or TopicsConfig.DEFAULT_MODEL
        self.loader = TopicLoader()
        self.merger = TopicMerger()
        self._clusterer: Optional[TopicCluster] = None
        logger.info("TopicsPipeline initialized | model=%s", self.model_name)

    def _get_clusterer(self) -> TopicCluster:
        if self._clusterer is None:
            self._clusterer = TopicCluster(self.model_name)
        return self._clusterer

    def run_full_pipeline(self, force_recluster: bool = False) -> Dict[str, Any]:
        logger.info("Starting topics pipeline | model=%s", self.model_name)
        try:
            data = self.loader.load_clean_data()
            clustered = self._get_clusterer().run_clustering(data, force_recluster)
            merge_result = self.merger.merge()
            summary = {
                "status": "success",
                "model": self.model_name,
                "documents_processed": len(clustered),
                "models_merged": len(merge_result.get("metadata", {}).get("models_merged", [])),
            }
            logger.info("Pipeline complete | %s", summary)
            return summary
        except Exception:
            logger.error("Pipeline failed", exc_info=True)
            raise
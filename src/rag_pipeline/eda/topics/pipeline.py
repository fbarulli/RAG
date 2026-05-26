from typing import Dict, Any, Optional
from pathlib import Path

from .config import TopicsConfig
from .core.topic_loader import TopicLoader
from .core.topic_cluster import TopicCluster
from .core.topic_merge import TopicMerger
# from .classification.rules import RuleClassifier  # to be enabled later


class TopicsPipeline:
    """Main orchestrator for the topics pipeline."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.config = TopicsConfig
        self.model_name = model_name or self.config.DEFAULT_MODEL
        
        # Lazy initialization
        self.loader: Optional[TopicLoader] = None
        self.clusterer: Optional[TopicCluster] = None
        self.merger: Optional[TopicMerger] = None

    def run_full_pipeline(self, force_recluster: bool = False) -> Dict[str, Any]:
        """Run the complete topic assignment pipeline."""
        print(f"Starting topics pipeline with model: {self.model_name}")
        # TODO: implement step by step
        return {"status": "skeleton", "model": self.model_name}

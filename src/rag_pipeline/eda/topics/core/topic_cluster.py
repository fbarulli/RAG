# src/rag_pipeline/eda/topics/core/topic_cluster.py
import logging
from typing import Any

from src.rag_pipeline.logging import get_logger

logger = get_logger(__name__)


class TopicCluster:
    """Fits BERTopic on a list of questions and returns raw model + assignments."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.info("TopicCluster initialized | model=%s", model_name)

    def run_clustering_raw(
        self,
        questions: list[str],
        min_topic_size: int,
        min_samples: int,
        stopwords: list[str],
    ) -> tuple[Any, list[int], list[float]]:
        """
        Fit BERTopic on questions.
        Returns (topic_model, topics, probs).
        """
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from umap import UMAP
        from hdbscan import HDBSCAN

        logger.info(
            "Fitting BERTopic | model=%s docs=%d min_topic_size=%d",
            self.model_name, len(questions), min_topic_size,
        )

        embedding_model = SentenceTransformer(self.model_name)
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            min_samples=min_samples,
            prediction_data=True,
        )
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            language="english",
            calculate_probabilities=True,
            verbose=False,

        )
        topics, probs = topic_model.fit_transform(questions)
        logger.info(
            "Clustering complete | topics=%d outliers=%d",
            len(set(topics) - {-1}),
            sum(1 for t in topics if t == -1),
        )
        return topic_model, list(topics), list(probs)
"""
_topic_cluster.py
=================
Core clustering logic for topic modeling.
Wraps BERTopic and HDBSCAN with configurable, permissive defaults to minimize outliers.
"""
import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


from rag_pipeline.logging import get_logger

logger = get_logger(__name__)


def cluster_topics(
    questions: list[str],
    embedding_model_name: str,
    min_topic_size: int = 5,
    min_samples: int = 1,
    stopwords: list[str] | None = None,
) -> tuple[BERTopic, list[int], list[float], np.ndarray]:
    """
    Run BERTopic with explicit HDBSCAN configuration.
    
    Args:
        questions: List of FAQ question strings.
        embedding_model_name: SentenceTransformer model name/path.
        min_topic_size: HDBSCAN min_cluster_size parameter.
        min_samples: HDBSCAN min_samples parameter (lower = fewer outliers).
        
    Returns:
        Tuple of (topic_model, topic_labels, probabilities, embeddings)
    """
    logger.info(f"Loading embedding model: {embedding_model_name}")
    logger.info(f"[cluster_topics] stopwords received: {len(stopwords) if stopwords else 0}")

    embedder = SentenceTransformer(embedding_model_name,trust_remote_code=True)

    # Pre-compute embeddings for efficiency
    logger.info("Encoding questions...")
    embeddings = embedder.encode(questions, convert_to_numpy=True)

    # Configure HDBSCAN with permissive settings to reduce outliers
    # Note: metric parameter removed - HDBSCAN BallTree doesn't support 'cosine'
    # Default Euclidean works well with normalized sentence embeddings
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=min_samples,
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
    stop_words=stopwords or "english",
    ngram_range=(1, 2),
)


    topic_model = BERTopic(
        embedding_model=embedder,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        verbose=True,
    )

    logger.info(f"Clustering topics (min_cluster_size={min_topic_size}, min_samples={min_samples})...")
    topics, probs = topic_model.fit_transform(questions, embeddings=embeddings)

    return topic_model, topics, probs, embeddings
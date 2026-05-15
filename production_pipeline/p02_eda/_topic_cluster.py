"""
_topic_cluster.py
=================
Fit BERTopic model on FAQ questions with pre-computed embeddings.

Single responsibility: encode questions, fit topic model, return artifacts.
No document loading, no subtopic logic, no output serialization.

Functions:
    cluster_topics(questions, embedding_model_name, min_topic_size) -> tuple
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

from rag_pipeline.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def cluster_topics(
    questions: list[str],
    embedding_model_name: str,
    min_topic_size: int,
) -> tuple[BERTopic, list[int], list[float], np.ndarray]:
    """
    Fit BERTopic model on question corpus with pre-computed embeddings.
    
    Args:
        questions: List of question strings to cluster
        embedding_model_name: Name of sentence-transformers model to use
        min_topic_size: Minimum number of documents per topic for BERTopic
        
    Returns:
        topic_model: Fitted BERTopic instance for later use (e.g., get_topic_info)
        topics: List of topic IDs (int) per question, in input order
        probs: List of topic probabilities (float) per question, flattened
        embeddings: Pre-computed numpy array of shape (n_questions, embedding_dim)
                    for reuse in subtopic generation
    """
    logger.info(f"Loading embedding model: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)
    
    logger.info(f"Encoding {len(questions)} questions")
    embeddings = embedding_model.encode(
        questions,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    
    logger.info(f"Fitting BERTopic (min_topic_size={min_topic_size})")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        verbose=True,
    )
    
    # fit_transform returns (topics, probs) where probs may be 1D or 2D depending on version
    topics_raw, probs_raw = topic_model.fit_transform(questions, embeddings)
    
    # Normalize outputs to plain Python lists for serialization safety
    topics = np.array(topics_raw).flatten().tolist()
    probs = np.array(probs_raw).flatten().tolist()
    
    num_topics = len(set(t for t in topics if t != -1))
    logger.info(f"Topic modeling complete. Found {num_topics} topics + outliers.")
    
    return topic_model, topics, probs, embeddings


def get_topic_keywords(topic_model: BERTopic, topic_id: int, top_n: int = 10) -> list[str]:
    """
    Extract top keywords for a given topic from a fitted BERTopic model.
    
    Args:
        topic_model: Fitted BERTopic instance
        topic_id: Integer topic ID (e.g., 0, 1, 2, ... or -1 for outliers)
        top_n: Number of keywords to return
        
    Returns:
        List of keyword strings for the topic
    """
    keywords = topic_model.get_topic(topic_id)
    if keywords is None:
        return []
    return [word for word, _ in keywords[:top_n]]


def get_topic_summary(topic_model: BERTopic) -> list[dict]:
    """
    Extract summary info for all topics from a fitted BERTopic model.
    
    Args:
        topic_model: Fitted BERTopic instance
        
    Returns:
        List of dicts with topic metadata: topic ID, count, keywords, name
    """
    topic_info_df = topic_model.get_topic_info()
    summary = []
    
    for _, row in topic_info_df.iterrows():
        topic_num = int(row["Topic"])
        keywords = get_topic_keywords(topic_model, topic_num, top_n=10)
        summary.append({
            "topic": topic_num,
            "count": int(row["Count"]),
            "keywords": keywords,
            "name": row.get("Name", ""),
        })
    
    return summary
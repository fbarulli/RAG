"""
_topic_subtopics.py
===================
Generate subtopics for large parent topics using BERTopic.

Single responsibility: recursive clustering on topic subsets.
No document loading, no main model fitting, no output serialization.

Functions:
    build_subtopics(assignments, questions, embeddings, subtopic_threshold, subtopic_min_size) -> dict
"""
import numpy as np
from bertopic import BERTopic

from rag_pipeline.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_subtopics(
    assignments: list,
    questions: list[str],
    embeddings: np.ndarray,
    subtopic_threshold: int,
    subtopic_min_size: int,
) -> dict[str, dict]:
    """
    Generate subtopics for parent topics that exceed size threshold.
    
    Args:
        assignments: List of TopicAssignment dataclasses
        questions: List of question strings aligned with assignments
        embeddings: Pre-computed embeddings array aligned with questions
        subtopic_threshold: Generate subtopics only for topics larger than this
        subtopic_min_size: Minimum docs per subtopic in recursive clustering
        
    Returns:
        Dict mapping doc_id -> {subtopic: int, subtopic_keywords: list[str]}
    """
    # Count topic sizes
    topic_sizes = {}
    for a in assignments:
        t = a.topic
        topic_sizes[t] = topic_sizes.get(t, 0) + 1
    
    # Identify large topics needing subtopics
    large_topics = {t for t, size in topic_sizes.items() if t != -1 and size > subtopic_threshold}
    
    if not large_topics:
        logger.info("No topics exceed subtopic threshold; skipping subtopic generation")
        return {}
    
    logger.info(f"Generating subtopics for {len(large_topics)} large topics")
    
    # Log outlier count
    outlier_count = sum(1 for a in assignments if a.topic == -1)
    if outlier_count:
        logger.info(f"{outlier_count} outlier docs (topic=-1) skipped for subtopic generation")
    
    subtopic_cache = {}
    
    for i, parent_topic in enumerate(sorted(large_topics), 1):
        logger.info(f"Subtopic generation [{i}/{len(large_topics)}] topic {parent_topic}")
        
        # Get indices for this parent topic
        parent_indices = [idx for idx, a in enumerate(assignments) if a.topic == parent_topic]
        if not parent_indices:
            continue
        
        # Extract subset data
        parent_questions = [questions[idx] for idx in parent_indices]
        parent_embeddings = embeddings[parent_indices]
        parent_ids = [assignments[idx].id for idx in parent_indices]
        
        # Fit subtopic model with pre-computed embeddings
        sub_model = BERTopic(
            embedding_model=None,
            min_topic_size=subtopic_min_size,
            verbose=False,
        )
        sub_topics, _ = sub_model.fit_transform(parent_questions, parent_embeddings)
        
        # Build subtopic records
        for local_idx, global_idx in enumerate(parent_indices):
            doc_id = parent_ids[local_idx]
            sub_t = int(sub_topics[local_idx])
            keywords = [word for word, _ in sub_model.get_topic(sub_t)[:5]]
            subtopic_cache[doc_id] = {
                "subtopic": sub_t,
                "subtopic_keywords": keywords,
            }
    
    logger.info(f"Subtopic generation complete: {len(subtopic_cache)} docs assigned subtopics")
    return subtopic_cache
"""
_topic_assignments.py
=====================
Build and compile topic assignments for output.

Single responsibility: map docs to topics, attach subtopics, build output dict.
No document loading, no clustering, no I/O.

Functions:
    build_assignments(docs, topics, probs) -> list[TopicAssignment]
    attach_subtopics(assignments, subtopics) -> None
    build_output(assignments, topic_model, ...) -> dict
"""
from dataclasses import dataclass, asdict
from typing import Optional

from rag_pipeline.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TopicAssignment:
    """Structured record for document topic assignment."""
    id: str
    course: str
    section: str
    topic: int
    topic_probability: float
    question: str
    subtopic: Optional[int] = None
    subtopic_keywords: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_assignments(
    docs: list[dict],
    topics: list[int],
    probs: list[float],
) -> list[TopicAssignment]:
    """Map each document to its base topic assignment."""
    assignments = []
    for i, doc in enumerate(docs):
        assignments.append(TopicAssignment(
            id=doc["id"],
            course=doc["course"],
            section=doc.get("section", ""),
            topic=topics[i],
            topic_probability=probs[i],
            question=doc["question"],
        ))
    return assignments


def attach_subtopics(
    assignments: list[TopicAssignment],
    subtopics: dict[str, dict],
) -> None:
    """
    Attach subtopic info to assignments in place.
    
    Explicit mutation: this function modifies the assignment objects directly.
    Callers should be aware of this side effect.
    
    Args:
        assignments: List of TopicAssignment objects to update
        subtopics: Dict mapping doc_id -> {subtopic, subtopic_keywords}
    """
    for a in assignments:
        if a.id in subtopics:
            sub = subtopics[a.id]
            a.subtopic = sub["subtopic"]
            a.subtopic_keywords = sub["subtopic_keywords"]


def build_output(
    assignments: list[TopicAssignment],
    topic_model,
    embedding_model_name: str,
    min_topic_size: int,
    subtopic_threshold: int,
    total_docs: int,
) -> dict:
    """
    Compile final output structure for serialization.
    
    Args:
        assignments: List of TopicAssignment objects (with subtopics attached)
        topic_model: Fitted BERTopic instance for extracting topic info
        embedding_model_name: Name of embedding model used
        min_topic_size: BERTopic min_topic_size param
        subtopic_threshold: Threshold used for subtopic generation
        total_docs: Total number of input documents
        
    Returns:
        Dict with metadata, topics summary, and assignments list
    """
    # Extract topic summary from model
    topic_info_df = topic_model.get_topic_info()
    topic_summary = []
    for _, row in topic_info_df.iterrows():
        topic_num = int(row["Topic"])
        keywords = [word for word, _ in topic_model.get_topic(topic_num)[:10]]
        topic_summary.append({
            "topic": topic_num,
            "count": int(row["Count"]),
            "keywords": keywords,
            "name": row.get("Name", ""),
        })
    
    return {
        "metadata": {
            "model": embedding_model_name,
            "min_topic_size": min_topic_size,
            "subtopic_threshold": subtopic_threshold,
            "total_docs": total_docs,
            "num_topics": len([t for t in topic_summary if t["topic"] != -1]),
        },
        "topics": topic_summary,
        "assignments": [asdict(a) for a in assignments],
    }
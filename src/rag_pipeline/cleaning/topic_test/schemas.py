"""
topic_test/schemas.py
Re-exports topic-relevant models from rag_pipeline.core.models.
"""
from rag_pipeline.core.models import TopicAssignment, TopicAssignments
__all__ = ["TopicAssignment", "TopicAssignments"]

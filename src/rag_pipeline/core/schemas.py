"""Backwards-compatibility shim. Import from rag_pipeline.core.models instead."""
from rag_pipeline.core.models import (
    FAQDocument,
    ProviderConfig,
    MultiLLMResult,
    TopicAssignment,
    TopicAssignments,
)

__all__ = [
    "FAQDocument",
    "ProviderConfig",
    "MultiLLMResult",
    "TopicAssignment",
    "TopicAssignments",
]

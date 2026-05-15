# /home/admin/LLM/LLM/01/web/src/guardrails.py

import re
from typing import Tuple, Optional

# Define out-of-scope topics with patterns
FORBIDDEN_PATTERNS = [
    (r"politics|election|trump|biden|democrat|republican|voting", "politics"),
    (r"medical|diagnosis|surgery|hospital|doctor|medicine|healthcare|symptom", "medical advice"),
    (r"legal|lawyer|attorney|court|sue|lawsuit|legal advice", "legal advice"),
    (r"finance|stock|invest|crypto|bitcoin|trading|mortgage|loan", "finance"),
    (r"weather|forecast|temperature|rain|snow", "weather"),
    (r"sports|football|baseball|basketball|soccer|team|game", "sports"),
    (r"recipe|cooking|baking|food|dinner|lunch", "cooking"),
]

OUT_OF_SCOPE_RESPONSE = "I can only answer questions about Data Engineering, Machine Learning, and MLOps Zoomcamp courses. Please ask a course-related question."

def is_out_of_scope(query: str) -> Tuple[bool, Optional[str]]:
    """
    Check if query is outside course scope.
    Returns (is_out_of_scope, detected_category)
    """
    query_lower = query.lower()
    for pattern, category in FORBIDDEN_PATTERNS:
        if re.search(pattern, query_lower):
            return True, category
    return False, None

def guardrail_filter(query: str) -> Tuple[bool, Optional[str]]:
    """
    Filter query through guardrails.
    Returns (is_allowed, response_message)
    """
    is_out, category = is_out_of_scope(query)
    if is_out:
        return False, OUT_OF_SCOPE_RESPONSE
    return True, None

def get_guardrail_response(category: str) -> str:
    """Get specific response for a category."""
    return OUT_OF_SCOPE_RESPONSE
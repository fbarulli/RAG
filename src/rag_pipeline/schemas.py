"""src/rag_pipeline/schemas.py
Canonical schema for FAQ documents. Ensures consistency across parsing, validation, EDA, and ingestion.
"""
from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass(frozen=True)
class FAQDocument:
    """Immutable, typed representation of a FAQ document."""
    id: str
    question: str
    answer: str
    course: str
    section: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to plain dict for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "FAQDocument":
        """Create instance from a dict (ignores extra keys)."""
        valid_keys = {"id", "question", "answer", "course", "section"}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})

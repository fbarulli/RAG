"""FAQ document schema."""
from __future__ import annotations
from typing import Optional
import json
from pydantic import BaseModel, ConfigDict, field_validator


class FAQDocument(BaseModel):
    """Immutable, validated FAQ document."""
    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    answer: str
    course: str
    section: Optional[str] = None

    @field_validator("id", "question", "answer", "course")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be empty or whitespace-only")
        return v

    def to_dict(self) -> dict:
        return self.model_dump()

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> FAQDocument:
        valid_keys = {"id", "question", "answer", "course", "section"}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})

"""src/rag_pipeline/schemas.py
Canonical schemas for FAQ documents and LLM results.
"""
from dataclasses import dataclass, asdict
from pydantic import BaseModel, ConfigDict, field_validator
from typing import Optional
from pathlib import Path
import json


class FAQDocument(BaseModel):
    """Immutable, validated FAQ document. Raises ValidationError on bad data."""
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
    def from_dict(cls, data: dict) -> "FAQDocument":
        valid_keys = {"id", "question", "answer", "course", "section"}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


@dataclass(frozen=True)
class ProviderConfig:
    """Static configuration for a single LLM provider."""
    name: str
    model: str
    env_key: str
    


@dataclass(frozen=True)
class MultiLLMResult:
    """Result of a multi-provider LLM call."""
    content: str
    latency_ms: float
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    is_second_opinion: bool = False
    fallback_used: bool = False

@dataclass(frozen=True)
class TopicAssignment:
    """Represents a single question with its assigned topic and NER category."""
    question: str
    topic: int
    topic_name: str
    ner_category: str
    model: str


@dataclass
class TopicAssignments:
    """Container for all topic modeling results."""
    models: list[str]
    results: dict[str, dict]   # model -> {metadata, assignments}
    
    def save(self, path: Optional[Path] = None):
        """Save to standard location using Paths"""
        from rag_pipeline.core.paths import Paths
        if path is None:
            path = Paths.topic_assignments()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                "models": self.models,
                "results": self.results
            }, f, indent=2)
    
    @classmethod
    def load(cls) -> "TopicAssignments":
        from rag_pipeline.core.paths import Paths
        path = Paths.topic_assignments()
        if not path.exists():
            raise FileNotFoundError(f"Topic assignments not found at {path}")
        with open(path) as f:
            data = json.load(f)
        return cls(models=data["models"], results=data["results"])
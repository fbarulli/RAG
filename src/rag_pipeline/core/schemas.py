"""src/rag_pipeline/schemas.py
Canonical schemas for FAQ documents and LLM results.
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
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

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
    rate_limit_wait: int = 60


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
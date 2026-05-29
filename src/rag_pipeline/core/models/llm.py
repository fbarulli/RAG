"""LLM provider and result schemas."""
from __future__ import annotations
from pydantic import BaseModel


class ProviderConfig(BaseModel, frozen=True):
    """Static configuration for a single LLM provider."""
    name: str
    model: str
    env_key: str


class MultiLLMResult(BaseModel, frozen=True):
    """Result of a multi-provider LLM call."""
    content: str
    latency_ms: float
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    is_second_opinion: bool = False
    fallback_used: bool = False

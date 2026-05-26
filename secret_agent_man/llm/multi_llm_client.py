"""src/rag_pipeline/core/multi_llm_client.py
Public API: call_with_fallback and call_second_opinion via litellm.Router.
"""
from __future__ import annotations

from typing import Optional

from secret_agent_man.llm.schemas import ProviderConfig, MultiLLMResult
from secret_agent_man.llm.llm_config import build_router, DEFAULT_CASCADE
from secret_agent_man.llm.llm_provider import call_router
from secret_agent_man.logging import get_logger

logger = get_logger(__name__)

# module-level router — built once, reused across all calls
_router = None


def _get_router(cascade: Optional[list[ProviderConfig]] = None):
    global _router
    if _router is None or cascade is not None:
        _router = build_router(cascade)
    return _router


def call_with_fallback(
    prompt: str,
    max_tokens: int,
    temperature: float = 0.3,
    system: Optional[str] = None,
    cascade: Optional[list[ProviderConfig]] = None,
) -> MultiLLMResult:
    router = _get_router(cascade)
    result = call_router(router, prompt, max_tokens, temperature, system)
    logger.info(
        f"call_with_fallback: answered by '{result.provider}' "
        f"({result.latency_ms:.0f}ms)"
    )
    return result


def call_second_opinion(
    prompt: str,
    max_tokens: int,
    exclude_provider: str,
    temperature: float = 0.3,
    system: Optional[str] = None,
    cascade: Optional[list[ProviderConfig]] = None,
) -> MultiLLMResult:
    # Build a one-off router excluding the primary provider
    providers = [p for p in (cascade or DEFAULT_CASCADE) if p.name != exclude_provider]
    router = build_router(providers)
    result = call_router(router, prompt, max_tokens, temperature, system)
    logger.info(
        f"call_second_opinion: answered by '{result.provider}' "
        f"(excluded='{exclude_provider}', {result.latency_ms:.0f}ms)"
    )
    return MultiLLMResult(**{**result.__dict__, "is_second_opinion": True})
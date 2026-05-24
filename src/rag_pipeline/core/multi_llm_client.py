"""src/rag_pipeline/core/multi_llm_client.py
Multi-provider LLM cascade: fallback and second-opinion calls.
"""
from __future__ import annotations

import traceback
from typing import Optional

from rag_pipeline.core.schemas import ProviderConfig, MultiLLMResult
from rag_pipeline.core.llm_config import DEFAULT_CASCADE, init_env
from rag_pipeline.core.llm_provider import call_provider, is_available, is_rate_limit_error
from rag_pipeline.core.cooldown import set_cooldown
from rag_pipeline.logging import get_logger

logger = get_logger(__name__)


def call_with_fallback(
    prompt: str,
    max_tokens: int,
    temperature: float = 0.3,
    system: Optional[str] = None,
    cascade: Optional[list[ProviderConfig]] = None,
) -> MultiLLMResult:
    init_env()
    providers = cascade or DEFAULT_CASCADE
    first_available: Optional[str] = None

    for provider in providers:
        if not is_available(provider):
            continue
        if first_available is None:
            first_available = provider.name
        try:
            result = call_provider(provider, prompt, max_tokens, temperature, system)
            fallback = provider.name != first_available
            logger.info(
                f"call_with_fallback: answered by '{provider.name}' "
                f"(fallback={fallback}, {result.latency_ms:.0f}ms)"
            )
            return MultiLLMResult(**{**result.__dict__, "fallback_used": fallback})
        except Exception as exc:
            if is_rate_limit_error(exc):
                set_cooldown(provider)
                logger.warning(f"Provider '{provider.name}' rate-limited — trying next. Error: {exc}")
            else:
                logger.error(f"Provider '{provider.name}' failed:\n{traceback.format_exc()}")

    raise RuntimeError("All LLM providers exhausted or unconfigured.")


def call_second_opinion(
    prompt: str,
    max_tokens: int,
    exclude_provider: str,
    temperature: float = 0.3,
    system: Optional[str] = None,
    cascade: Optional[list[ProviderConfig]] = None,
) -> MultiLLMResult:
    init_env()
    providers = cascade or DEFAULT_CASCADE

    for provider in providers:
        if provider.name == exclude_provider:
            continue
        if not is_available(provider):
            continue
        try:
            result = call_provider(provider, prompt, max_tokens, temperature, system)
            logger.info(
                f"call_second_opinion: answered by '{provider.name}' "
                f"(excluded='{exclude_provider}', {result.latency_ms:.0f}ms)"
            )
            return MultiLLMResult(**{**result.__dict__, "is_second_opinion": True})
        except Exception as exc:
            if is_rate_limit_error(exc):
                set_cooldown(provider)
                logger.warning(f"Second-opinion provider '{provider.name}' rate-limited.")
            else:
                logger.error(f"Second-opinion provider '{provider.name}' failed:\n{traceback.format_exc()}")

    raise RuntimeError(
        f"No alternate provider available for second opinion (excluded='{exclude_provider}')."
    )
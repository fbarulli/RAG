"""src/rag_pipeline/core/llm_provider.py
Single-provider call logic and availability checks.
"""
from __future__ import annotations

import os
import time
from typing import Optional

from rag_pipeline.core.schemas import ProviderConfig, MultiLLMResult
from rag_pipeline.core.cooldown import is_cooled_down
from rag_pipeline.logging import get_logger

logger = get_logger(__name__)


def is_rate_limit_error(exc: Exception) -> bool:
    try:
        from litellm.exceptions import RateLimitError, ServiceUnavailableError
        if isinstance(exc, (RateLimitError, ServiceUnavailableError)):
            return True
    except ImportError:
        pass
    msg = str(exc)
    return any(code in msg for code in ["429", "502", "503", "504", "RateLimitError", "rate limit"])


def is_available(provider: ProviderConfig) -> bool:
    """Return True if provider has a key configured and is not in cooldown."""
    if not os.getenv(provider.env_key):
        logger.debug(f"Provider '{provider.name}' skipped — {provider.env_key} not set.")
        return False
    return not is_cooled_down(provider)


def call_provider(
    provider: ProviderConfig,
    prompt: str,
    max_tokens: int,
    temperature: float,
    system: Optional[str],
) -> MultiLLMResult:
    """
    Single attempt against one provider. Raises on any error.
    Caller handles cooldown and fallback.
    """
    from litellm import completion

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    t0 = time.monotonic()
    resp = completion(
        model=provider.model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = (time.monotonic() - t0) * 1000

    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)

    return MultiLLMResult(
        content=content.strip(),
        latency_ms=elapsed,
        model=provider.model,
        provider=provider.name,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )
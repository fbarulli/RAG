"""src/rag_pipeline/core/llm_provider.py
Single call through the litellm Router.
"""
from __future__ import annotations

import time
from typing import Optional, TYPE_CHECKING

from secret_agent_man.llm.schemas import MultiLLMResult
from secret_agent_man.logging import get_logger

if TYPE_CHECKING:
    from litellm import Router

logger = get_logger(__name__)


def call_router(
    router: "Router",
    prompt: str,
    max_tokens: int,
    temperature: float,
    system: Optional[str],
) -> MultiLLMResult:
    """Send a prompt through the router. Raises on total failure."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    t0 = time.monotonic()
    resp = router.completion(
        model="cascade",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = (time.monotonic() - t0) * 1000

    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    provider = getattr(resp, "_hidden_params", {}).get("custom_llm_provider", "unknown")
    model = getattr(resp, "model", "unknown")

    return MultiLLMResult(
        content=content.strip(),
        latency_ms=elapsed,
        model=model,
        provider=provider,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )
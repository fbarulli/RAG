"""
src/rag_pipeline/core/multi_llm_client.py
==========================================
Multi-provider LLM client with cascade fallback and second-opinion support.

Providers (in default cascade order):
    nvidia      nvidia_nim/meta/llama-3.1-70b-instruct
    groq        groq/llama-3.1-70b-versatile
    openrouter  openrouter/meta-llama/llama-3.1-70b-instruct
    huggingface huggingface/meta-llama/Llama-3.1-70B-Instruct

Behaviour:
    call_with_fallback()   — tries providers in order; skips cooled-down or
                             unconfigured providers; raises only when all exhausted.
    call_second_opinion()  — picks the next available provider after the one
                             that answered first; intended for agent reconciliation.

Cooldown:
    A provider that returns a rate-limit or 5xx error is placed in a per-process
    cooldown (module-level dict). It is skipped — not retried — until the cooldown
    expires. This avoids burning retries against a provider that is already down.

Usage:
    from rag_pipeline.core.multi_llm_client import call_with_fallback, call_second_opinion

    result = call_with_fallback("explain gradient descent", max_tokens=300)
    print(result.provider, result.content)

    opinion = call_second_opinion(
        "explain gradient descent",
        max_tokens=300,
        exclude_provider=result.provider,
    )
"""
from __future__ import annotations

import os
import time
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rag_pipeline.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_RETRIES = 3
_DEFAULT_RATE_LIMIT_WAIT = 60  # seconds


# ---------------------------------------------------------------------------
# .env loader — walks up from this file, same pattern as gem_client.py
# ---------------------------------------------------------------------------
def _find_dotenv() -> Optional[Path]:
    current = Path(__file__).resolve().parent
    while True:
        candidate = current / ".env"
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _init() -> None:
    """Load all provider API keys from nearest .env. Safe to call multiple times."""
    from dotenv import load_dotenv
    env_path = _find_dotenv()
    if env_path:
        load_dotenv(env_path)
        logger.debug(f"Loaded .env from {env_path}")
    else:
        logger.warning("No .env found; API keys must already be in the environment.")

    # litellm reads GROQ_API_KEY, OPENROUTER_API_KEY, HUGGINGFACE_API_KEY directly.
    # NVIDIA NIM needs its own env var name.
    nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    if nvidia_key:
        os.environ["NVIDIA_NIM_API_KEY"] = nvidia_key


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProviderConfig:
    """Static configuration for a single LLM provider."""
    name: str            # short identifier used in logs and exclude_provider
    model: str           # full litellm model string
    env_key: str         # env var whose presence signals the key is configured
    rate_limit_wait: int = _DEFAULT_RATE_LIMIT_WAIT


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


# ---------------------------------------------------------------------------
# Default cascade — order determines priority
# ---------------------------------------------------------------------------
DEFAULT_CASCADE: list[ProviderConfig] = [
    ProviderConfig(
        name="nvidia",
        model="nvidia_nim/meta/llama-3.1-70b-instruct",
        env_key="NVIDIA_API_KEY",
        rate_limit_wait=60,
    ),
    ProviderConfig(
        name="groq",
        model="groq/llama-3.1-70b-versatile",
        env_key="GROQ_API_KEY",
        rate_limit_wait=60,
    ),
    ProviderConfig(
        name="openrouter",
        model="openrouter/meta-llama/llama-3.1-70b-instruct",
        env_key="OPENROUTER_API_KEY",
        rate_limit_wait=30,
    ),
    ProviderConfig(
        name="huggingface",
        model="huggingface/meta-llama/Llama-3.1-70B-Instruct",
        env_key="HUGGINGFACE_API_KEY",
        rate_limit_wait=60,
    ),
]

# module-level cooldown registry: provider name -> monotonic timestamp until ready
_cooldown_until: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _is_rate_limit(exc: Exception) -> bool:
    try:
        from litellm.exceptions import RateLimitError, ServiceUnavailableError
        if isinstance(exc, (RateLimitError, ServiceUnavailableError)):
            return True
    except ImportError:
        pass
    msg = str(exc)
    return any(code in msg for code in ["429", "502", "503", "504", "RateLimitError", "rate limit"])


def _is_available(provider: ProviderConfig) -> bool:
    """Return True if the provider has a key configured and is not in cooldown."""
    if not os.getenv(provider.env_key):
        logger.debug(f"Provider '{provider.name}' skipped — {provider.env_key} not set.")
        return False
    cooldown_exp = _cooldown_until.get(provider.name, 0.0)
    if time.monotonic() < cooldown_exp:
        remaining = cooldown_exp - time.monotonic()
        logger.debug(f"Provider '{provider.name}' in cooldown for {remaining:.0f}s more.")
        return False
    return True


def _set_cooldown(provider: ProviderConfig) -> None:
    _cooldown_until[provider.name] = time.monotonic() + provider.rate_limit_wait
    logger.warning(
        f"Provider '{provider.name}' placed in cooldown for {provider.rate_limit_wait}s."
    )


def _call_provider(
    provider: ProviderConfig,
    prompt: str,
    max_tokens: int,
    temperature: float,
    system: Optional[str],
) -> MultiLLMResult:
    """
    Single attempt against one provider. Raises on any error.
    Caller is responsible for cooldown management.
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def call_with_fallback(
    prompt: str,
    max_tokens: int,
    temperature: float = 0.3,
    system: Optional[str] = None,
    cascade: Optional[list[ProviderConfig]] = None,
) -> MultiLLMResult:
    """
    Call LLM with automatic provider fallback.

    Iterates through cascade in order. Skips providers that are unconfigured
    or in cooldown. On rate-limit or 5xx error the provider is cooled down and
    the next one is tried immediately. Raises RuntimeError if all exhausted.

    Args:
        prompt:      User prompt.
        max_tokens:  Required — no silent truncation.
        temperature: Sampling temperature.
        system:      Optional system message.
        cascade:     Override provider order. Defaults to DEFAULT_CASCADE.

    Returns:
        MultiLLMResult with fallback_used=True if primary provider was skipped.
    """
    _init()
    providers = cascade or DEFAULT_CASCADE
    first_available: Optional[str] = None

    for provider in providers:
        if not _is_available(provider):
            continue

        if first_available is None:
            first_available = provider.name

        try:
            result = _call_provider(provider, prompt, max_tokens, temperature, system)
            fallback = provider.name != first_available
            logger.info(
                f"call_with_fallback: answered by '{provider.name}' "
                f"(fallback={fallback}, {result.latency_ms:.0f}ms)"
            )
            # Return a new frozen instance with fallback flag set correctly
            return MultiLLMResult(
                content=result.content,
                latency_ms=result.latency_ms,
                model=result.model,
                provider=result.provider,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                fallback_used=fallback,
            )
        except Exception as exc:
            if _is_rate_limit(exc):
                _set_cooldown(provider)
                logger.warning(
                    f"Provider '{provider.name}' rate-limited — trying next. "
                    f"Error: {exc}"
                )
            else:
                logger.error(
                    f"Provider '{provider.name}' failed with non-rate-limit error:\n"
                    f"{traceback.format_exc()}"
                )
            continue

    raise RuntimeError(
        "All LLM providers exhausted or unconfigured. "
        "Check API keys and cooldown state."
    )


def call_second_opinion(
    prompt: str,
    max_tokens: int,
    exclude_provider: str,
    temperature: float = 0.3,
    system: Optional[str] = None,
    cascade: Optional[list[ProviderConfig]] = None,
) -> MultiLLMResult:
    """
    Get a second opinion from a different provider than the one that answered first.

    Picks the next available provider in the cascade that is not exclude_provider.
    Does not cascade further — raises if no alternate provider is available.

    Args:
        prompt:            The same prompt sent to the first provider.
        max_tokens:        Required.
        exclude_provider:  Name of the provider to skip (e.g. result.provider).
        temperature:       Sampling temperature.
        system:            Optional system message.
        cascade:           Override provider order.

    Returns:
        MultiLLMResult with is_second_opinion=True.

    Example (agent reconciliation):
        first  = call_with_fallback(prompt, max_tokens=300)
        second = call_second_opinion(prompt, max_tokens=300,
                                     exclude_provider=first.provider)
        # agent compares first.content vs second.content
    """
    _init()
    providers = cascade or DEFAULT_CASCADE

    for provider in providers:
        if provider.name == exclude_provider:
            continue
        if not _is_available(provider):
            continue

        try:
            result = _call_provider(provider, prompt, max_tokens, temperature, system)
            logger.info(
                f"call_second_opinion: answered by '{provider.name}' "
                f"(excluded='{exclude_provider}', {result.latency_ms:.0f}ms)"
            )
            return MultiLLMResult(
                content=result.content,
                latency_ms=result.latency_ms,
                model=result.model,
                provider=result.provider,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                is_second_opinion=True,
            )
        except Exception as exc:
            if _is_rate_limit(exc):
                _set_cooldown(provider)
                logger.warning(
                    f"Second-opinion provider '{provider.name}' rate-limited. "
                    f"Trying next."
                )
            else:
                logger.error(
                    f"Second-opinion provider '{provider.name}' failed:\n"
                    f"{traceback.format_exc()}"
                )
            continue

    raise RuntimeError(
        f"No alternate provider available for second opinion "
        f"(excluded='{exclude_provider}'). Check keys and cooldown state."
    )
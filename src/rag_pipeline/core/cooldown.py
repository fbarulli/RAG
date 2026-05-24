"""src/rag_pipeline/core/cooldown.py
Per-process provider cooldown registry.
"""
from __future__ import annotations

import time

from rag_pipeline.core.schemas import ProviderConfig
from rag_pipeline.logging import get_logger

logger = get_logger(__name__)

# provider name -> monotonic timestamp when cooldown expires
_cooldown_until: dict[str, float] = {}


def is_cooled_down(provider: ProviderConfig) -> bool:
    """Return True if provider is still in cooldown."""
    expiry = _cooldown_until.get(provider.name, 0.0)
    if time.monotonic() < expiry:
        remaining = expiry - time.monotonic()
        logger.debug(f"Provider '{provider.name}' in cooldown for {remaining:.0f}s more.")
        return True
    return False


def set_cooldown(provider: ProviderConfig) -> None:
    """Place provider in cooldown for its configured duration."""
    _cooldown_until[provider.name] = time.monotonic() + provider.rate_limit_wait
    logger.warning(
        f"Provider '{provider.name}' placed in cooldown for {provider.rate_limit_wait}s."
    )


def clear_cooldown(provider_name: str) -> None:
    """Manually clear cooldown — useful in tests."""
    _cooldown_until.pop(provider_name, None)
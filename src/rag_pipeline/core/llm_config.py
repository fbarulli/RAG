"""src/rag_pipeline/core/llm_config.py
Provider configuration and environment initialisation.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from rag_pipeline.core.schemas import ProviderConfig
from rag_pipeline.logging import get_logger

logger = get_logger(__name__)


def find_dotenv() -> Optional[Path]:
    current = Path(__file__).resolve().parent
    while True:
        candidate = current / ".env"
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


def init_env() -> None:
    """Load provider API keys from nearest .env. Safe to call multiple times."""
    from dotenv import load_dotenv
    env_path = find_dotenv()
    if env_path:
        load_dotenv(env_path)
        logger.debug(f"Loaded .env from {env_path}")
    else:
        logger.warning("No .env found; API keys must already be in the environment.")

    nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    if nvidia_key:
        os.environ["NVIDIA_NIM_API_KEY"] = nvidia_key


DEFAULT_CASCADE: list[ProviderConfig] = [
    ProviderConfig(
        name="nvidia",
        model="nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct",
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
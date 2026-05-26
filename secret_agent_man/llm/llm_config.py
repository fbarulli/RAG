"""src/rag_pipeline/core/llm_config.py
Provider configuration and litellm.Router initialisation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from secret_agent_man.llm.schemas import ProviderConfig
from secret_agent_man.logging import get_logger

logger = get_logger(__name__)

_PROVIDERS_JSON = Path(__file__).parent / "providers.json"


def load_cascade(path: Path = _PROVIDERS_JSON) -> list[ProviderConfig]:
    with open(path) as f:
        return [ProviderConfig(**entry) for entry in json.load(f)]


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


def build_router(cascade: Optional[list[ProviderConfig]] = None) -> "litellm.Router":
    from litellm import Router

    init_env()
    providers = cascade or load_cascade()

    model_list = [
        {
            "model_name": "cascade",        # single logical name for all providers
            "litellm_params": {
                "model": p.model,
                "rpm": p.rpm,
                "tpm": p.tpm,
            },
        }
        for p in providers
        if os.getenv(p.env_key)             # skip unconfigured providers
    ]

    if not model_list:
        raise RuntimeError("No providers configured — check your API keys.")

    logger.info(f"Router: {len(model_list)} providers: {[p.name for p in providers if os.getenv(p.env_key)]}")

    return Router(
        model_list=model_list,
        routing_strategy="least-busy",      # picks least loaded provider
        fallbacks=[{"cascade": ["cascade"]}],
        retry_after=5,
        num_retries=3,
    )


DEFAULT_CASCADE: list[ProviderConfig] = load_cascade()
"""
src/rag_pipeline/service/cascading_model.py
============================================
smolagents-compatible model wrapper that routes agent LLM calls through
the multi-provider cascade client (NVIDIA → Groq → OpenRouter → HuggingFace).

Supports:
  - Automatic provider fallback on rate-limit / 5xx
  - Second opinion via call_second_opinion() for agent reconciliation

Usage:
    from rag_pipeline.service.cascading_model import CascadingModel
    from smolagents import CodeAgent

    model = CascadingModel()
    agent = CodeAgent(tools=[], model=model, add_base_tools=True)
    agent.run("your task here")
"""
from __future__ import annotations

from typing import Any, Optional

from rag_pipeline.core.llm_config import DEFAULT_CASCADE
from rag_pipeline.schemas import ProviderConfig, MultiLLMResult
from smolagents.models import ChatMessage, MessageRole, TokenUsage

from rag_pipeline.core.multi_llm_client import (
    call_with_fallback,
    call_second_opinion,
    MultiLLMResult,
    DEFAULT_CASCADE,
    ProviderConfig,
)
from rag_pipeline.logging import get_logger

logger = get_logger(__name__)


class CascadingModel(LiteLLMModel):
    """
    smolagents model that routes every generate() call through the
    multi-provider cascade. Falls back across providers automatically.

    Overrides generate() — the correct override point in smolagents 1.25.0.
    Returns ChatMessage as required by the smolagents contract.
    """

    def __init__(
        self,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        cascade: Optional[list[ProviderConfig]] = None,
    ) -> None:
        # model_id is required by LiteLLMModel.__init__ but we never use it
        # for actual calls — our cascade handles dispatch.
        super().__init__(model_id="cascade/fallback")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._cascade = cascade or DEFAULT_CASCADE

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """
        Route agent LLM call through cascade. Returns ChatMessage.

        smolagents passes messages as list[ChatMessage] or list[dict].
        We normalise to role/content dicts, extract the system message,
        and format the rest as a single prompt string.
        """
        # Normalise to dicts
        normalised: list[dict] = []
        for m in messages:
            if isinstance(m, dict):
                normalised.append(m)
            else:
                normalised.append({"role": str(m.role.value if hasattr(m.role, 'value') else m.role), "content": m.content or ""})

        system: Optional[str] = next(
            (m["content"] for m in normalised if m.get("role") == "system"), None
        )

        # Format non-system turns into a single prompt
        turns: list[str] = []
        for m in normalised:
            if m.get("role") == "system":
                continue
            label = "Assistant" if m.get("role") == "assistant" else "User"
            turns.append(f"{label}: {m.get('content', '')}")
        prompt = "\n".join(turns)

        result: MultiLLMResult = call_with_fallback(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            cascade=self._cascade,
        )

        logger.info(
            f"CascadingModel: provider='{result.provider}' "
            f"fallback={result.fallback_used} "
            f"tokens={result.prompt_tokens}+{result.completion_tokens} "
            f"latency={result.latency_ms:.0f}ms"
        )

        content = result.content

        # Honour stop sequences if the provider didn't handle them
        if stop_sequences:
            for seq in stop_sequences:
                if seq in content:
                    content = content[: content.index(seq)]

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=None,
            raw=result,
            token_usage=TokenUsage(
                input_tokens=result.prompt_tokens,
                output_tokens=result.completion_tokens,
            ),
        )

    def second_opinion(
        self,
        messages: list[ChatMessage | dict],
        exclude_provider: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> ChatMessage:
        """
        Get a second opinion from a different provider.

        Intended for agent reconciliation loops:
            first  = model.generate(messages)
            second = model.second_opinion(messages, exclude_provider=first.raw.provider)
            # agent compares first.content vs second.content

        Args:
            messages:         Same messages list passed to generate().
            exclude_provider: Provider name to skip (e.g. first.raw.provider).
            max_tokens:       Defaults to self.max_tokens.
            temperature:      Defaults to self.temperature.

        Returns:
            ChatMessage with raw.is_second_opinion=True.
        """
        normalised: list[dict] = []
        for m in messages:
            if isinstance(m, dict):
                normalised.append(m)
            else:
                normalised.append({"role": str(m.role.value if hasattr(m.role, 'value') else m.role), "content": m.content or ""})

        system: Optional[str] = next(
            (m["content"] for m in normalised if m.get("role") == "system"), None
        )
        turns = [
            f"{'Assistant' if m.get('role') == 'assistant' else 'User'}: {m.get('content', '')}"
            for m in normalised if m.get("role") != "system"
        ]
        prompt = "\n".join(turns)

        result: MultiLLMResult = call_second_opinion(
            prompt=prompt,
            max_tokens=max_tokens or self.max_tokens,
            exclude_provider=exclude_provider,
            temperature=temperature or self.temperature,
            system=system,
            cascade=self._cascade,
        )

        logger.info(
            f"CascadingModel.second_opinion: provider='{result.provider}' "
            f"latency={result.latency_ms:.0f}ms"
        )

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=result.content,
            tool_calls=None,
            raw=result,
            token_usage=TokenUsage(
                input_tokens=result.prompt_tokens,
                output_tokens=result.completion_tokens,
            ),
        )
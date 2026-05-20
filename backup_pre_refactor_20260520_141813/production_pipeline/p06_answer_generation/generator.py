"""
production_pipeline/p06_answer_generation/generator.py
LLM Answer Generation Module.

This module handles the generation of answers from retrieved context using 
Language Models (LLMs) in the RAG pipeline.

Classes:
    GeneratedAnswer: Dataclass capturing LLM outputs, metadata, and performance metrics.
    AnswerGenerator: Core engine to construct prompts and manage LLM execution.

Methods:
    AnswerGenerator.from_config(config): Instantiates the generator using a GenerationConfig object.
    AnswerGenerator.generate(query_id, query, context_doc_ids, context_text, prompt_style): Formats the context and query into a template, invokes the LLM, and returns execution metrics.
"""


from typing import Optional
from dataclasses import dataclass
from rag_pipeline.logging import get_logger
from rag_pipeline.paths import Paths
from rag_pipeline.llm_client import call_llm, LLMResult
from .config import GenerationConfig, PromptStyle, get_prompt, PROMPT_CONFIGS

logger = get_logger(__name__)


@dataclass
class GeneratedAnswer:
    """Result of answer generation."""
    query_id: str
    query: str
    context_doc_ids: list[str]
    context_text: str
    generated_answer: str
    prompt_style: str
    temperature: float
    max_tokens: int
    llm_model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    success: bool
    error: Optional[str] = None


class AnswerGenerator:
    """Generate answers using LLM with retrieved context."""

    def __init__(self, llm_model: str, max_tokens: int, temperature: float):
        self.llm_model = llm_model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @classmethod
    def from_config(cls, config: GenerationConfig) -> "AnswerGenerator":
        """Construct from a GenerationConfig."""
        return cls(
            llm_model=config.llm_model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

    def generate(
        self,
        query_id: str,
        query: str,
        context_doc_ids: list[str],
        context_text: str,
        prompt_style: PromptStyle,
    ) -> GeneratedAnswer:
        """
        Generate an answer using LLM.

        Args:
            query_id: Unique identifier for the query
            query: User's question
            context_doc_ids: List of retrieved document IDs
            context_text: Combined context from retrieved documents
            prompt_style: Which prompt template to use

        Returns:
            GeneratedAnswer dataclass with results.
        """
        prompt_cfg = PROMPT_CONFIGS[prompt_style]
        prompt = prompt_cfg.format(context=context_text, query=query)

        try:
            result: LLMResult = call_llm(
                prompt=prompt,
                max_tokens=prompt_cfg.max_tokens,
                model=self.llm_model,
                temperature=prompt_cfg.temperature,
            )

            return GeneratedAnswer(
                query_id=query_id,
                query=query,
                context_doc_ids=context_doc_ids,
                context_text=context_text,
                generated_answer=result.content,
                prompt_style=prompt_style,
                temperature=prompt_cfg.temperature,
                max_tokens=prompt_cfg.max_tokens,
                llm_model=self.llm_model,
                latency_ms=result.latency_ms,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                success=True,
                error=None,
            )

        except Exception as e:
            logger.error(f"Generation failed for {query_id}: {e}")
            return GeneratedAnswer(
                query_id=query_id,
                query=query,
                context_doc_ids=context_doc_ids,
                context_text=context_text,
                generated_answer="",
                prompt_style=prompt_style,
                temperature=prompt_cfg.temperature,
                max_tokens=prompt_cfg.max_tokens,
                llm_model=self.llm_model,
                latency_ms=0,
                prompt_tokens=0,
                completion_tokens=0,
                success=False,
                error=str(e),
            )
"""Evaluate generated answers against reference answers using LLM judge."""
import json
import traceback
from typing import Optional
from dataclasses import dataclass
from rag_pipeline.logging import get_logger
from rag_pipeline.core.paths import Paths
from rag_pipeline.core.llm_client import call_llm, LLMResult
logger = get_logger(__name__)
JUDGE_LLM = Paths.get('judge_llm', 'nvidia_nim/meta/llama-3.1-70b-instruct')
JUDGE_MAX_TOKENS = 150
JUDGE_TEMPERATURE = 0.0
JUDGE_PROMPT = 'You are evaluating a Q&A system for a course FAQ.\n\nUSER QUESTION: {query}\n\nREFERENCE ANSWER (ground truth):\n{reference}\n\nGENERATED ANSWER (to evaluate):\n{generated}\n\nRate the generated answer on two metrics (0.0 to 1.0):\n\n1. faithfulness: Does the generated answer address the user\'s question?\n   - 1.0 = Fully addresses the question\n   - 0.5 = Partially addresses the question\n   - 0.0 = Does not address the question at all\n\n2. factual_correctness: Does the generated answer match the reference answer\'s facts?\n   - 1.0 = All facts are correct and match reference\n   - 0.5 = Some facts match, some are incorrect/missing\n   - 0.0 = No facts match or contradicts reference\n\nOutput ONLY JSON: {{"faithfulness": 0.0, "factual_correctness": 0.0}}'

def _parse_scores(raw: str) -> dict[str, float]:
    """
    Parse faithfulness and factual_correctness from LLM judge response.
    Handles markdown fences and preamble text before the JSON object.
    """
    if '```' in raw:
        raw = raw.split('```')[1]
        if raw.startswith('json'):
            raw = raw[4:]
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    json_line = next((l for l in reversed(lines) if l.startswith('{')), raw)
    scores = json.loads(json_line)
    return {'faithfulness': float(scores.get('faithfulness', 0.0)), 'factual_correctness': float(scores.get('factual_correctness', 0.0))}

@dataclass
class EvaluationResult:
    """Result of answer evaluation."""
    query_id: str
    faithfulness: float
    factual_correctness: float
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    error: Optional[str] = None

class AnswerEvaluator:
    """Evaluate generated answers against reference using LLM judge."""

    def __init__(self, llm_model: str=JUDGE_LLM, max_tokens: int=JUDGE_MAX_TOKENS, temperature: float=JUDGE_TEMPERATURE):
        self.llm_model = llm_model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def evaluate(self, query_id: str, query: str, generated_answer: str, reference_answer: str) -> EvaluationResult:
        """
        Evaluate a generated answer against reference.

        Args:
            query_id: Unique identifier for the query
            query: Original user question
            generated_answer: The LLM-generated answer
            reference_answer: Ground truth answer from FAQ

        Returns:
            EvaluationResult with faithfulness and factual_correctness scores.
        """
        if not generated_answer:
            return EvaluationResult(query_id=query_id, faithfulness=0.0, factual_correctness=0.0, latency_ms=0, prompt_tokens=0, completion_tokens=0, error='No generated answer to evaluate')
        prompt = JUDGE_PROMPT.format(query=query, reference=reference_answer, generated=generated_answer)
        try:
            result: LLMResult = call_llm(prompt=prompt, max_tokens=self.max_tokens, model=self.llm_model, temperature=self.temperature)
            scores = _parse_scores(result.content.strip())
            return EvaluationResult(query_id=query_id, faithfulness=scores['faithfulness'], factual_correctness=scores['factual_correctness'], latency_ms=result.latency_ms, prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens, error=None)
        except Exception:
            logger.error(f'Evaluation failed for {query_id}:\n{traceback.format_exc()}')
            return EvaluationResult(query_id=query_id, faithfulness=0.0, factual_correctness=0.0, latency_ms=0, prompt_tokens=0, completion_tokens=0, error=traceback.format_exc())
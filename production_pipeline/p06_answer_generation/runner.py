"""Orchestrate answer generation and evaluation pipeline."""

import json
from typing import Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from rag_pipeline.logging import get_logger
from rag_pipeline.paths import Paths
from .config import GenerationConfig, PromptStyle, PROMPT_CONFIGS
from .retriever import ContextRetriever
from .generator import AnswerGenerator, GeneratedAnswer
from .evaluator import AnswerEvaluator, EvaluationResult

logger = get_logger(__name__)

TOP_K_VALUES = [1, 3, 5]


@dataclass
class PipelineResult:
    """Complete result for a single query."""
    query_id: str
    query: str
    expected_id: str
    retrieved_doc_ids: list[str]
    generation: GeneratedAnswer
    evaluation: EvaluationResult
    prompt_style: str
    top_k: int
    timestamp: str


class AnswerGenerationPipeline:
    """Orchestrate the full answer generation and evaluation pipeline."""

    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()

        self.retriever = ContextRetriever(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port,
            model_name=self.config.retrieval_model,
        )
        self.generator = AnswerGenerator.from_config(self.config)
        self.evaluator = AnswerEvaluator(llm_model=self.config.llm_model)

    def process_query(
        self,
        query_id: str,
        query: str,
        expected_id: str,
        reference_answer: str,
        prompt_style: PromptStyle,
        top_k: int,
        top_hit_ids: Optional[list[str]] = None,
    ) -> PipelineResult:
        """Process a single query through the full pipeline."""
        if top_hit_ids is None:
            logger.warning(f"No top_hit_ids provided for {query_id}, falling back to expected_id")
            retrieved_doc_ids = [expected_id]
        else:
            retrieved_doc_ids = top_hit_ids[:top_k]

        context_text = self.retriever.get_context(retrieved_doc_ids)

        generation = self.generator.generate(
            query_id=query_id,
            query=query,
            context_doc_ids=retrieved_doc_ids,
            context_text=context_text,
            prompt_style=prompt_style,
        )

        if generation.success:
            evaluation = self.evaluator.evaluate(
                query_id=query_id,
                query=query,
                generated_answer=generation.generated_answer,
                reference_answer=reference_answer,
            )
        else:
            evaluation = EvaluationResult(
                query_id=query_id,
                faithfulness=0.0,
                factual_correctness=0.0,
                latency_ms=0,
                prompt_tokens=0,
                completion_tokens=0,
                error=generation.error,
            )

        return PipelineResult(
            query_id=query_id,
            query=query,
            expected_id=expected_id,
            retrieved_doc_ids=retrieved_doc_ids,
            generation=generation,
            evaluation=evaluation,
            prompt_style=prompt_style,
            top_k=top_k,
            timestamp=datetime.now().isoformat(),
        )


def run_pipeline(
    config: Optional[GenerationConfig] = None,
    prompt_styles: Optional[list[PromptStyle]] = None,
    top_k_values: Optional[list[int]] = None,
    limit: Optional[int] = None,
) -> None:
    """
    Run the answer generation pipeline on benchmark results.

    Iterates over every combination of prompt_style × top_k, collecting
    generation metrics (tokens, latency) and judge scores (faithfulness,
    factual_correctness) for each combination.
    """
    config = config or GenerationConfig()
    prompt_styles = prompt_styles or list(PROMPT_CONFIGS.keys())
    top_k_values = top_k_values or TOP_K_VALUES

    benchmark_results_path = Paths.benchmark_query_results()
    test_set_path = Paths.test_jsonl()
    output_path = Paths.experiments_dir() / "generation_results.json"

    logger.info(f"Loading benchmark results from {benchmark_results_path}")
    with open(benchmark_results_path) as f:
        all_results = json.load(f)

    target_results = [
        r for r in all_results
        if r.get("model") == config.retrieval_model and r.get("config") == config.retrieval_config
    ]

    if limit:
        target_results = target_results[:limit]

    combinations = len(prompt_styles) * len(top_k_values)
    logger.info(
        f"Processing {len(target_results)} queries × "
        f"{len(prompt_styles)} styles × {len(top_k_values)} top_k values "
        f"= {len(target_results) * combinations} total runs"
    )

    ref_answers: dict[str, str] = {}
    with open(test_set_path) as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                ref_answers[doc["id"]] = doc.get("answer", "")

    pipeline = AnswerGenerationPipeline(config=config)
    all_pipeline_results: list[PipelineResult] = []

    total = len(target_results) * combinations
    run = 0

    for i, result in enumerate(target_results):
        query_id = result["query_id"]
        query = result["query_text"]
        expected_id = result["expected_id"]
        top_hit_ids = result.get("top_hit_ids") or (
            [result["top_hit_id"]] if result.get("top_hit_id") else []
        )
        reference_answer = ref_answers.get(query_id, "")
        if not reference_answer:
            logger.warning(f"No reference answer for {query_id}, evaluation scores unreliable")

        for style in prompt_styles:
            for top_k in top_k_values:
                run += 1
                logger.info(f"[{run}/{total}] {query_id} | style={style} top_k={top_k}")
                all_pipeline_results.append(pipeline.process_query(
                    query_id=query_id,
                    query=query,
                    expected_id=expected_id,
                    reference_answer=reference_answer,
                    prompt_style=style,
                    top_k=top_k,
                    top_hit_ids=top_hit_ids,
                ))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in all_pipeline_results], f, indent=2)

    logger.info(f"Saved {len(all_pipeline_results)} results to {output_path}")
    _print_summary(all_pipeline_results, prompt_styles, top_k_values)


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_summary(
    results: list[PipelineResult],
    styles: list[str],
    top_k_values: list[int],
) -> None:
    """Print summary table broken down by prompt_style × top_k."""
    col_w = 10

    # Header
    print("\n" + "=" * 110)
    print("ANSWER GENERATION SUMMARY")
    print("=" * 110)
    print(
        f"{'Style':<12} {'top_k':>5}  "
        f"{'Faithful':>{col_w}} {'Factual':>{col_w}} "
        f"{'Gen p50ms':>{col_w}} {'Judge p50ms':>{col_w}} "
        f"{'Prompt tok':>{col_w}} {'Comp tok':>{col_w}} "
        f"{'Total tok':>{col_w}}  {'N':>5}"
    )
    print("-" * 110)

    best_mrr = -1.0
    best_label = ""

    for style in styles:
        for top_k in top_k_values:
            bucket = [
                r for r in results
                if r.prompt_style == style and r.top_k == top_k
            ]
            if not bucket:
                continue

            faithful   = _avg([r.evaluation.faithfulness for r in bucket])
            factual    = _avg([r.evaluation.factual_correctness for r in bucket])
            gen_lat    = _sorted_p50([r.generation.latency_ms for r in bucket])
            judge_lat  = _sorted_p50([r.evaluation.latency_ms for r in bucket])
            prompt_tok = _avg([r.generation.prompt_tokens for r in bucket])
            comp_tok   = _avg([r.generation.completion_tokens for r in bucket])
            total_tok  = _avg([
                r.generation.prompt_tokens + r.generation.completion_tokens
                for r in bucket
            ])

            # Combined score for winner detection
            combined = (faithful + factual) / 2
            label = f"{style}/top_k={top_k}"
            if combined > best_mrr:
                best_mrr = combined
                best_label = label

            print(
                f"{style:<12} {top_k:>5}  "
                f"{faithful:>{col_w}.3f} {factual:>{col_w}.3f} "
                f"{gen_lat:>{col_w}.0f} {judge_lat:>{col_w}.0f} "
                f"{prompt_tok:>{col_w}.0f} {comp_tok:>{col_w}.0f} "
                f"{total_tok:>{col_w}.0f}  {len(bucket):>5}"
            )

        # Blank line between styles for readability
        print()

    print("=" * 110)
    print(f"🏆 Best combination: {best_label}  (avg judge score: {best_mrr:.3f})")
    print("=" * 110)


def _sorted_p50(values: list[float]) -> float:
    """Return the median (p50) of a list of floats."""
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Answer Generation Pipeline")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--retrieval-config", type=str, default=None)
    parser.add_argument("--prompt-style", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--styles", type=str, nargs="+", default=None)
    parser.add_argument(
        "--top-k-values", type=int, nargs="+", default=None,
        help="List of top_k values to evaluate, e.g. --top-k-values 1 3 5",
    )
    args = parser.parse_args()

    config = GenerationConfig(
        **({"retrieval_model": args.model} if args.model else {}),
        **({"retrieval_config": args.retrieval_config} if args.retrieval_config else {}),
        **({"prompt_style": args.prompt_style} if args.prompt_style else {}),
    )

    run_pipeline(
        config=config,
        prompt_styles=args.styles,
        top_k_values=args.top_k_values,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
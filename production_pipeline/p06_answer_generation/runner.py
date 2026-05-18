"""
production_pipeline/p06_answer_generation/runner.py

Orchestrate answer generation and evaluation pipeline.

Changes vs previous version:
- generate_single: accepts rerank=True to rerank retrieved IDs before
  context assembly. Reranking runs inside ContextRetriever so no new
  dependencies leak into runner.
- run_generations: passes rerank flag through; retrieves at max top_k
  then reranks per-combination (same cost as before at retrieval, one
  cross-encoder pass per query×top_k slice).
- run_pipeline: exposes rerank parameter, defaults True.
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from rag_pipeline.logging import get_logger
from rag_pipeline.paths import Paths
from .config import GenerationConfig, PromptStyle, PROMPT_CONFIGS
from .retriever import ContextRetriever
from .generator import AnswerGenerator
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
    generation: dict
    evaluation: dict
    prompt_style: str
    top_k: int
    timestamp: str
    reranked: bool = False


# ---------------------------------------------------------------------------
# Test set loader
# ---------------------------------------------------------------------------

def load_test_queries(test_set_path: Path, limit: Optional[int] = None) -> list[dict]:
    """Load test queries directly from test.jsonl."""
    queries = []
    with open(test_set_path) as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            queries.append({
                "query_id": doc["id"],
                "query_text": doc["question"],
                "expected_id": doc.get("expected_id") or doc.get("expected_doc_id") or doc["id"],
                "answer": doc.get("answer", ""),
            })

    if limit:
        queries = queries[:limit]

    logger.info(f"Loaded {len(queries)} test queries")
    return queries


# ---------------------------------------------------------------------------
# Retriever with embedding
# ---------------------------------------------------------------------------

class QueryRetriever:
    """Retrieve documents for a query using embedding model."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

        import json
        defaults_path = Paths.base() / "configs" / "defaults.json"
        with open(defaults_path) as f:
            defaults = json.load(f)

        qdrant_cfg = defaults.get("qdrant", {})
        host = qdrant_cfg.get("host", "localhost")
        port = qdrant_cfg.get("port", 6333)
        self.client = QdrantClient(host=host, port=port)
        short_name = model_name.split("/")[-1].replace("-", "_").replace(".", "_")
        self.collection = f"faqs_{short_name}"
        logger.info(f"Retriever using collection: {self.collection}")

    def retrieve(self, query: str, top_k: int) -> list[str]:
        """Retrieve top_k document IDs for a query."""
        vector = self.model.encode(query).tolist()
        results = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return [hit.payload.get("es_id", "") for hit in results.points]


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def generate_single(
    query_id: str,
    query: str,
    expected_id: str,
    reference_answer: str,
    retrieved_ids: list[str],
    prompt_style: str,
    top_k: int,
    generator: AnswerGenerator,
    evaluator: AnswerEvaluator,
    context_retriever: ContextRetriever,
    rerank: bool = True,
) -> PipelineResult:
    """
    Generate and evaluate a single answer.

    If rerank=True, retrieved_ids are reranked by the cross-encoder before
    context assembly. The cross-encoder scores (query, question+answer) pairs,
    which is more precise than embedding similarity for short FAQ docs.
    retrieved_ids should already be sliced to the desired top_k before calling
    this function, unless reranking — in which case pass the full candidate set
    and rerank() handles the slice.
    """
    if rerank and len(retrieved_ids) > 1:
        final_ids = context_retriever.rerank(query, retrieved_ids, top_k)
    else:
        final_ids = retrieved_ids[:top_k]

    context_text = context_retriever.get_context(final_ids)

    generation = generator.generate(
        query_id=query_id,
        query=query,
        context_doc_ids=final_ids,
        context_text=context_text,
        prompt_style=prompt_style,
    )

    if generation.success:
        evaluation = evaluator.evaluate(
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
        retrieved_doc_ids=final_ids,
        generation=asdict(generation),
        evaluation=asdict(evaluation),
        prompt_style=prompt_style,
        top_k=top_k,
        timestamp=datetime.now().isoformat(),
        reranked=rerank,
    )


def run_generations(
    test_queries: list[dict],
    retriever: QueryRetriever,
    context_retriever: ContextRetriever,
    generator: AnswerGenerator,
    evaluator: AnswerEvaluator,
    prompt_styles: list[str],
    top_k_values: list[int],
    rerank: bool = True,
) -> list[PipelineResult]:
    """
    Run generation for all query × style × top_k combinations.

    Retrieval strategy with reranking:
    - First-stage: retrieve RERANK_POOL candidates (wider net)
    - Per top_k combination: rerank the pool, slice to top_k
    - Cost: one cross-encoder pass per query×top_k (fast, ~10ms on CPU)

    Without reranking, behaviour is unchanged from before.
    """
    # Retrieve a wider pool when reranking so the cross-encoder has
    # more candidates to promote. 10 is enough for typical FAQ corpora.
    RERANK_POOL = 10
    retrieve_k = RERANK_POOL if rerank else max(top_k_values)

    results = []
    total = len(test_queries) * len(prompt_styles) * len(top_k_values)
    idx = 0

    for query_data in tqdm(test_queries, desc="Generating answers"):
        query_id = query_data["query_id"]
        query = query_data["query_text"]
        expected_id = query_data["expected_id"]
        reference_answer = query_data["answer"]

        # One retrieval call per query at pool size
        candidate_ids = retriever.retrieve(query, retrieve_k)

        for style in prompt_styles:
            for top_k in top_k_values:
                idx += 1
                logger.debug(f"[{idx}/{total}] {query_id} | style={style} top_k={top_k}")

                result = generate_single(
                    query_id=query_id,
                    query=query,
                    expected_id=expected_id,
                    reference_answer=reference_answer,
                    retrieved_ids=candidate_ids,   # full pool; rerank() slices
                    prompt_style=style,
                    top_k=top_k,
                    generator=generator,
                    evaluator=evaluator,
                    context_retriever=context_retriever,
                    rerank=rerank,
                )
                results.append(result)

    logger.info(f"Generated {len(results)} results")
    return results


def save_results(results: list[PipelineResult], output_path: Path) -> None:
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info(f"Saved {len(results)} results to {output_path}")


def print_summary(
    results: list[PipelineResult],
    styles: list[str],
    top_k_values: list[int],
) -> None:
    """Print summary table broken down by prompt_style × top_k."""
    col_w = 10

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

    best_score = -1.0
    best_label = ""

    for style in styles:
        for top_k in top_k_values:
            bucket = [r for r in results if r.prompt_style == style and r.top_k == top_k]
            if not bucket:
                continue

            faithful   = _avg([r.evaluation["faithfulness"] for r in bucket])
            factual    = _avg([r.evaluation["factual_correctness"] for r in bucket])
            gen_lat    = _p50([r.generation["latency_ms"] for r in bucket])
            judge_lat  = _p50([r.evaluation["latency_ms"] for r in bucket])
            prompt_tok = _avg([r.generation["prompt_tokens"] for r in bucket])
            comp_tok   = _avg([r.generation["completion_tokens"] for r in bucket])
            total_tok  = _avg([
                r.generation["prompt_tokens"] + r.generation["completion_tokens"]
                for r in bucket
            ])

            combined = (faithful + factual) / 2
            label = f"{style}/top_k={top_k}"
            if combined > best_score:
                best_score = combined
                best_label = label

            print(
                f"{style:<12} {top_k:>5}  "
                f"{faithful:>{col_w}.3f} {factual:>{col_w}.3f} "
                f"{gen_lat:>{col_w}.0f} {judge_lat:>{col_w}.0f} "
                f"{prompt_tok:>{col_w}.0f} {comp_tok:>{col_w}.0f} "
                f"{total_tok:>{col_w}.0f}  {len(bucket):>5}"
            )

        print()

    print("=" * 110)
    print(f"🏆 Best combination: {best_label}  (avg judge score: {best_score:.3f})")
    print("=" * 110)


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _p50(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    retrieval_model: Optional[str] = None,
    llm_model: Optional[str] = None,
    prompt_styles: Optional[list[str]] = None,
    top_k_values: Optional[list[int]] = None,
    limit: Optional[int] = None,
    rerank: bool = True,
) -> None:
    """Run answer generation pipeline directly from test set."""
    retrieval_model = retrieval_model or "BAAI/bge-base-en-v1.5"
    llm_model = llm_model or "nvidia_nim/meta/llama-3.1-70b-instruct"
    prompt_styles = prompt_styles or list(PROMPT_CONFIGS.keys())
    top_k_values = top_k_values or TOP_K_VALUES

    test_queries = load_test_queries(Paths.test_jsonl(), limit)

    retriever = QueryRetriever(model_name=retrieval_model)

    defaults_path = Paths.base() / "configs" / "defaults.json"
    with open(defaults_path) as f:
        defaults = json.load(f)

    qdrant_cfg = defaults.get("qdrant", {})
    context_retriever = ContextRetriever(
        host=qdrant_cfg.get("host", "localhost"),
        port=qdrant_cfg.get("port", 6333),
        model_name=retrieval_model,
    )

    config = GenerationConfig(llm_model=llm_model)
    generator = AnswerGenerator.from_config(config)
    evaluator = AnswerEvaluator(llm_model=llm_model)

    results = run_generations(
        test_queries=test_queries,
        retriever=retriever,
        context_retriever=context_retriever,
        generator=generator,
        evaluator=evaluator,
        prompt_styles=prompt_styles,
        top_k_values=top_k_values,
        rerank=rerank,
    )

    output_path = Paths.experiments_dir() / "generation_results.json"
    save_results(results, output_path)
    print_summary(results, prompt_styles, top_k_values)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    from configs.benchmark_cli import create_generation_parser
    parser = create_generation_parser()
    args = parser.parse_args()

    run_pipeline(
        retrieval_model=args.model or None,
        llm_model=args.llm_model or None,
        prompt_styles=args.styles,
        top_k_values=args.top_k_list,
        limit=args.limit,
        rerank=getattr(args, "rerank", True),
    )


if __name__ == "__main__":
    main()
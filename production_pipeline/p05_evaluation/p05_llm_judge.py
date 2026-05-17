"""
p05_llm_judge.py
================
LLM-as-judge evaluation for retrieval quality.

Reads benchmark_query_results.json (per-query retrieval results),
fetches the retrieved answer from Qdrant, and scores each result
using an LLM judge on two dimensions:

  faithfulness        : does the retrieved answer address the query?
  factual_correctness : does the retrieved answer match the reference answer?

Scores are 0.0–1.0. Results are saved incrementally to avoid loss on crash.

Run:
    uv run python -m production_pipeline.p05_evaluation.p05_llm_judge
    uv run python -m production_pipeline.p05_evaluation.p05_llm_judge --model "BAAI/bge-base-en-v1.5" --config "hybrid_rrf"
"""
import argparse
import json
import traceback
from pathlib import Path

from qdrant_client import QdrantClient

from rag_pipeline.llm_client import call_llm, LLMResult
from rag_pipeline.logging import get_logger
from rag_pipeline.paths import Paths

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_QUERY_RESULTS = Paths.experiments_dir() / "benchmark_query_results.json"
DEFAULT_OUTPUT = Paths.experiments_dir() / "judge_results.json"
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_CONFIG = "hybrid_rrf"
JUDGE_LLM = "nvidia_nim/meta/llama-3.1-70b-instruct"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
BATCH_SAVE_EVERY = 10


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """FAQ retrieval judge. Score the retrieved answer.

Q: {query}
REF: {reference}
GOT: {retrieved}

faithfulness: does GOT address Q? (1.0=yes, 0.5=partial, 0.0=no)
factual_correctness: does GOT match REF facts? (1.0=all, 0.5=some, 0.0=none)

Output ONLY: {{"faithfulness": <float>, "factual_correctness": <float>}}"""


# ---------------------------------------------------------------------------
# Qdrant answer fetcher
# ---------------------------------------------------------------------------

def build_answer_map(
    collection: str,
    client: QdrantClient,
) -> dict[str, str]:
    """Fetch all es_id -> answer mappings from Qdrant collection."""
    answer_map = {}
    offset = None

    while True:
        result = client.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = result
        for p in points:
            es_id = p.payload.get("es_id", "")
            answer = p.payload.get("answer", "")
            if es_id:
                answer_map[es_id] = answer

        if next_offset is None or len(points) == 0:
            break
        offset = next_offset

    logger.info(f"[build_answer_map] {len(answer_map)} answers loaded from {collection}")
    return answer_map


# ---------------------------------------------------------------------------
# Judge scorer
# ---------------------------------------------------------------------------

def score_result(
    query: str,
    reference: str,
    retrieved: str | None,
) -> dict[str, float]:
    """Call LLM judge and parse scores."""
    if not retrieved:
        return {"faithfulness": 0.0, "factual_correctness": 0.0, "latency_ms": 0.0}

    prompt = JUDGE_PROMPT.format(
        query=query,
        reference=reference,
        retrieved=retrieved,
    )

    try:
        result: LLMResult = call_llm(prompt=prompt, max_tokens=150, temperature=0.0)
        raw = result.content.strip()

        logger.info(
            f"[score_result] tokens=prompt:{result.prompt_tokens} "
            f"completion:{result.completion_tokens} "
            f"latency={result.latency_ms:.0f}ms"
        )
        logger.debug(f"[score_result] raw response: {raw}")

        # Strip markdown fences if present
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        # Extract last line containing JSON (prompt asks for explanation then JSON)
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        json_line = next(
            (l for l in reversed(lines) if l.startswith("{")),
            raw
        )

        scores = json.loads(json_line)
        return {
            "faithfulness": float(scores.get("faithfulness", 0.0)),
            "factual_correctness": float(scores.get("factual_correctness", 0.0)),
            "latency_ms": result.latency_ms,
        }

    except Exception:
        logger.warning(f"[score_result] Failed to parse judge response:\n{traceback.format_exc()}")
        return {"faithfulness": 0.0, "factual_correctness": 0.0, "latency_ms": 0.0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    model_name: str = DEFAULT_MODEL,
    config_name: str = DEFAULT_CONFIG,
    query_results_path: Path = DEFAULT_QUERY_RESULTS,
    output_path: Path = DEFAULT_OUTPUT,
    limit: int | None = None,
) -> None:
    logger.info(f"[p05_llm_judge] model={model_name} config={config_name}")

    if not query_results_path.exists():
        raise FileNotFoundError(f"Query results not found: {query_results_path}")

    all_results = json.load(open(query_results_path))
    target_results = [
        r for r in all_results
        if r.get("model") == model_name and r.get("config") == config_name
    ]

    if limit:
        target_results = target_results[:limit]
        logger.info(f"[p05_llm_judge] Limited to {limit} queries for test run")

    logger.info(f"[p05_llm_judge] {len(target_results)} queries to judge")

    if not target_results:
        raise ValueError(
            f"No results found for model={model_name} config={config_name}. "
            f"Available: {list(set((r['model'], r['config']) for r in all_results))}"
        )

    # Load reference answers from test set
    test_set_path = Paths.processed_dir() / "test.jsonl"
    ref_answers: dict[str, str] = {}
    for line in open(test_set_path, encoding="utf-8"):
        doc = json.loads(line)
        ref_answers[doc["id"]] = doc.get("answer", "")

    # Connect to Qdrant and build answer map
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    short_name = model_name.split("/")[-1].replace("-", "_").replace(".", "_")
    collection = f"faqs_{short_name}"
    answer_map = build_answer_map(collection, client)

    # Load existing results for resume support
    existing: dict[str, dict] = {}
    if output_path.exists():
        try:
            existing = {r["query_id"]: r for r in json.load(open(output_path))}
            logger.info(f"[p05_llm_judge] Resuming — {len(existing)} already scored")
        except Exception:
            logger.warning("[p05_llm_judge] Could not load existing results — starting fresh")

    judge_results = list(existing.values())
    scored = 0
    skipped = 0

    for i, r in enumerate(target_results):
        query_id = r["query_id"]

        if query_id in existing:
            skipped += 1
            continue

        query_text = r["query_text"]
        expected_id = r["expected_id"]
        top_hit_id = r.get("top_hit_id")

        ref_answer = ref_answers.get(query_id, "")
        retrieved_answer = answer_map.get(top_hit_id, "") if top_hit_id else ""

        scores = score_result(
            query=query_text,
            reference=ref_answer,
            retrieved=retrieved_answer,
        )

        judge_result = {
            "query_id": query_id,
            "query_text": query_text,
            "expected_id": expected_id,
            "top_hit_id": top_hit_id,
            "hit": r.get("hit", False),
            "query_type": r.get("query_type", "unknown"),
            "course": r.get("course", ""),
            "model": model_name,
            "config": config_name,
            **scores,
        }
        judge_results.append(judge_result)
        scored += 1

        logger.info(
            f"[{i+1}/{len(target_results)}] {query_id} | "
            f"hit={r.get('hit')} | "
            f"faith={scores['faithfulness']:.2f} "
            f"fc={scores['factual_correctness']:.2f} | "
            f"latency={scores.get('latency_ms', 0):.0f}ms | "
            f"{query_text[:60]}"
        )

        if scored % BATCH_SAVE_EVERY == 0:
            _save(judge_results, output_path)
            logger.info(f"[p05_llm_judge] Checkpoint saved ({scored} scored)")

    _save(judge_results, output_path)
    _print_summary(judge_results, model_name, config_name)
    logger.info(f"[p05_llm_judge] Done — {scored} scored, {skipped} skipped (resume)")


def _save(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def _print_summary(results: list[dict], model: str, config: str) -> None:
    if not results:
        return

    faith = [r["faithfulness"] for r in results if "faithfulness" in r]
    fc = [r["factual_correctness"] for r in results if "factual_correctness" in r]

    by_type: dict[str, list] = {}
    for r in results:
        qt = r.get("query_type", "unknown")
        by_type.setdefault(qt, []).append(r)

    non_chaos = [r for r in results if r.get("query_type") != "chaos_monkey"]
    chaos = [r for r in results if r.get("query_type") == "chaos_monkey"]

    composite_non_chaos = sum(
        (r["faithfulness"] + r["factual_correctness"]) / 2
        for r in non_chaos
    ) / len(non_chaos) if non_chaos else 0

    composite_chaos = sum(
        r["factual_correctness"] for r in chaos
    ) / len(chaos) if chaos else 0

    print(f"\n{'='*70}")
    print(f"LLM JUDGE RESULTS — {model} / {config}")
    print(f"{'='*70}")
    print(f"  Queries scored         : {len(results)}")
    print(f"  Mean faithfulness      : {sum(faith)/len(faith):.3f}")
    print(f"  Mean factual corr.     : {sum(fc)/len(fc):.3f}")
    print(f"  Composite (non-chaos)  : {composite_non_chaos:.3f}")
    print(f"  Composite (chaos fc)   : {composite_chaos:.3f}")
    print(f"\n  By query type:")
    for qt, items in sorted(by_type.items()):
        f_scores = [r["faithfulness"] for r in items]
        fc_scores = [r["factual_correctness"] for r in items]
        primary = "fc only" if qt == "chaos_monkey" else "faith+fc"
        print(
            f"    {qt:20s} n={len(items):3d} "
            f"faith={sum(f_scores)/len(f_scores):.3f} "
            f"fc={sum(fc_scores)/len(fc_scores):.3f} "
            f"[{primary}]"
        )
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-judge for retrieval evaluation")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--query-results", type=Path, default=DEFAULT_QUERY_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="Limit queries for test run")
    args = parser.parse_args()

    main(
        model_name=args.model,
        config_name=args.config,
        query_results_path=args.query_results,
        output_path=args.output,
        limit=args.limit,
    )
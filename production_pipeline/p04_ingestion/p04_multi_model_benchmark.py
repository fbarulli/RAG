"""
p04_multi_model_benchmark.py
============================
Runs retrieval benchmarks across multiple embedding models on a holdout test set.
Calls p03's building blocks directly rather than spawning subprocesses, so errors
surface with full tracebacks and no argument-interface mismatch is possible.

Output: experiments/benchmark_results.json
        experiments/benchmark_summary.txt
        experiments/benchmark_comparison.json
Run:    uv run python -m production_pipeline.p04_ingestion.p04_multi_model_benchmark
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rag_pipeline.logging import get_logger
from rag_pipeline.paths import Paths

from ._benchmark_loader import load_configs, load_test_set, load_topic_assignments
from ._benchmark_metrics import aggregate_metrics, evaluate_config
from ._benchmark_report import print_full_benchmark_report, save_benchmark_results
from ._benchmark_types import MetricSummary

logger = get_logger(__name__)

DEFAULT_TEST_SET = Paths.processed_dir() / "test.jsonl"
DEFAULT_TOPIC_ASSIGNMENTS = Paths.experiments_dir() / "topic_assignments.json"
DEFAULT_CONFIGS = Path("configs/retrieval_configs.json")
DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_TOP_K = 10
DEFAULT_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2",
    "nomic-ai/nomic-embed-text-v1.5",
    "intfloat/e5-small-v2",
]


# ---------------------------------------------------------------------------
# Comparison table (p04-specific; p03's report covers per-config detail)
# ---------------------------------------------------------------------------

def _best_summary_per_model(summaries: list[MetricSummary]) -> list[dict[str, Any]]:
    """
    Collapse per-config summaries to one row per model by picking the config
    with the highest MRR for each model.
    """
    best: dict[str, MetricSummary] = {}
    for s in summaries:
        if s.num_queries == 0:
            continue
        if s.model_name not in best or s.mrr > best[s.model_name].mrr:
            best[s.model_name] = s

    rows = []
    for s in best.values():
        rows.append({
            "model": s.model_name,
            "best_config": s.config_name,
            "hit_at_1": s.hit_rate_1,
            "hit_at_5": s.hit_rate_5,
            "hit_at_10": s.hit_rate_10,
            "mrr": s.mrr,
            "ndcg_at_10": s.ndcg_10,
            "latency_p50_ms": s.latency_p50,
            "latency_p95_ms": s.latency_p95,
            # New metrics
            "cross_course_contam": s.cross_course_contamination,
            "rank_std": s.rank_std,
            "failures": s.failure_count,
            "avg_fail_sim": s.avg_failure_similarity or 0.0,
        })

    rows.sort(key=lambda r: r["mrr"], reverse=True)
    return rows


def print_comparison_table(rows: list[dict]) -> None:
    """Print summary table with all new metrics."""
    if not rows:
        print("No results to display.")
        return

    print("\n" + "=" * 160)
    print("MULTI-MODEL BENCHMARK COMPARISON (Best config per model)")
    print("=" * 160)
    print(
        f"{'Model':<35} {'Config':<12} {'H@1':>6} {'H@5':>6} {'H@10':>6} {'MRR':>6} "
        f"{'NDCG@10':>8} {'P50(ms)':>7} {'P95(ms)':>7} {'CrossCourse':>11} "
        f"{'RankStd':>7} {'Fails':>5} {'FailSim':>7}"
    )
    print("-" * 160)

    for r in sorted(rows, key=lambda x: x['mrr'], reverse=True):
        print(
            f"{r['model']:<35} "
            f"{r['best_config']:<12} "
            f"{r['hit_at_1']:>5.1%} "
            f"{r['hit_at_5']:>5.1%} "
            f"{r['hit_at_10']:>5.1%} "
            f"{r['mrr']:>5.3f} "
            f"{r['ndcg_at_10']:>7.3f} "
            f"{r['latency_p50_ms']:>6.1f} "
            f"{r['latency_p95_ms']:>6.1f} "
            f"{r['cross_course_contam']:>10.2%} "
            f"{r['rank_std']:>6.2f} "
            f"{r['failures']:>4} "
            f"{r.get('avg_fail_sim', 0):>6.3f}"
        )
    print("=" * 160)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-model retrieval benchmark")
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--output-dir", type=Path, default=Paths.experiments_dir())
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="Models to benchmark (defaults to DEFAULT_MODELS)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Single retrieval config to test (defaults to all configs)",
    )
    parser.add_argument("--qdrant-host", type=str, default=DEFAULT_QDRANT_HOST)
    parser.add_argument("--qdrant-port", type=int, default=DEFAULT_QDRANT_PORT)
    parser.add_argument(
        "--no-detail", action="store_true",
        help="Skip printing the per-config detail report; show comparison table only",
    )
    args = parser.parse_args()

    models_to_run = args.models or DEFAULT_MODELS

    try:
        logger.info("Step 1/4: Loading test set, topics, and configs...")
        test_set = load_test_set(args.test_set)
        topic_map = load_topic_assignments(DEFAULT_TOPIC_ASSIGNMENTS)
        configs = load_configs(DEFAULT_CONFIGS)

        if args.config and args.config not in configs:
            logger.error(f"Config '{args.config}' not found. Available: {list(configs.keys())}")
            sys.exit(1)

        configs_to_test = [args.config] if args.config else list(configs.keys())

        logger.info("Step 2/4: Connecting to Qdrant...")
        client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)

        all_summaries: list[MetricSummary] = []

        logger.info(
            f"Step 3/4: Running evaluations "
            f"({len(models_to_run)} models x {len(configs_to_test)} configs)..."
        )
        for model_name in models_to_run:
            collection = f"faqs_{model_name.split('/')[-1].replace('-', '_')}"

            if not client.collection_exists(collection):
                logger.warning(f"Collection '{collection}' not found; skipping {model_name}")
                continue

            logger.info(f"Loading model: {model_name}")
            model = SentenceTransformer(model_name)

            for cfg_name in configs_to_test:
                config = configs[cfg_name]
                logger.info(f"  Evaluating config '{cfg_name}' on {collection}")

                results = evaluate_config(
                    client=client,
                    collection=collection,
                    model=model,
                    test_set=test_set,
                    topic_map=topic_map,
                    config=config,
                    top_k=DEFAULT_TOP_K,
                )
                summary = aggregate_metrics(results, cfg_name, model_name)
                all_summaries.append(summary)
                logger.info(f"    Hit@5: {summary.hit_rate_5:.1%}  MRR: {summary.mrr:.4f}")

        if not all_summaries:
            logger.error(
                "No results collected — check that Qdrant collections exist "
                "for the requested models"
            )
            sys.exit(1)

        logger.info("Step 4/4: Generating reports...")

        if not args.no_detail:
            print_full_benchmark_report(all_summaries)

        save_benchmark_results(all_summaries, args.output_dir)

        comparison_rows = _best_summary_per_model(all_summaries)
        print_comparison_table(comparison_rows)

        comparison_path = args.output_dir / "benchmark_comparison.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "models_tested": len(models_to_run),
                    "configs_tested": configs_to_test,
                    "best_model": comparison_rows[0]["model"] if comparison_rows else None,
                    "results": comparison_rows,
                },
                f,
                indent=2,
            )
        logger.info(f"Saved comparison summary: {comparison_path}")

    except Exception as exc:
        logger.exception(f"Benchmark failed: {exc}")
        sys.exit(1)



if __name__ == "__main__":
    main()


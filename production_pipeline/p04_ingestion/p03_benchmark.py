"""
p03_benchmark.py
================
Retrieval benchmark for FAQ RAG pipeline.
Tests embedding models + retrieval configs against held-out test set.
Metrics: Hit Rate@k, MRR, NDCG, latency, code integrity, topic-stratified.
Input:  test.jsonl, topic_assignments.json, retrieval_configs.json
Output: experiments/benchmark_results.json, benchmark_summary.txt
Run:    uv run python -m production_pipeline.p04_ingestion.p03_benchmark
"""
import argparse
from pathlib import Path
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rag_pipeline.paths import Paths
from rag_pipeline.logging import get_logger
from ._benchmark_loader import load_test_set, load_topic_assignments, load_configs
from ._benchmark_metrics import evaluate_config, aggregate_metrics
from ._benchmark_report import print_full_benchmark_report, save_benchmark_results

logger = get_logger(__name__)

DEFAULT_TEST_SET = Paths.processed_dir() / "test.jsonl"
DEFAULT_TOPIC_ASSIGNMENTS = Paths.experiments_dir() / "topic_assignments.json"
DEFAULT_CONFIGS = Path("configs/retrieval_configs.json")
DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_TOP_K = 10
DEFAULT_EMBEDDING_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-small-v2",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
]
OUTPUT_DIR = Paths.experiments_dir()

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark retrieval configurations")
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--topic-assignments", type=Path, default=DEFAULT_TOPIC_ASSIGNMENTS)
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--qdrant-host", type=str, default=DEFAULT_QDRANT_HOST)
    parser.add_argument("--qdrant-port", type=int, default=DEFAULT_QDRANT_PORT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser

def main():
    args = _build_parser().parse_args()

    try:
        logger.info("Step 1/6: Loading test set")
        test_set = load_test_set(args.test_set)

        logger.info("Step 2/6: Loading topic assignments")
        topic_map = load_topic_assignments(args.topic_assignments)

        logger.info("Step 3/6: Loading retrieval configs")
        configs = load_configs(args.configs)

        if args.config and args.config not in configs:
            raise ValueError(f"Config '{args.config}' not found. Available: {list(configs.keys())}")

        logger.info("Step 4/6: Connecting to Qdrant")
        client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)

        models_to_test = [args.model] if args.model else DEFAULT_EMBEDDING_MODELS
        configs_to_test = [args.config] if args.config else list(configs.keys())
        all_summaries = []

        logger.info("Step 5/6: Running benchmark evaluations")
        for model_name in models_to_test:
            logger.info(f"Loading model: {model_name}")
            model = SentenceTransformer(model_name)

            collection = args.collection or model_name.split("/")[-1].replace("-", "_")
            logger.info(f"Using collection: '{collection}'")

            if not client.collection_exists(collection):
                logger.warning(f"Collection '{collection}' not found; skipping {model_name}")
                continue

            for config_name in configs_to_test:
                config = configs[config_name]
                logger.info(f"Evaluating config: {config_name} on collection: {collection}")

                raw_results = evaluate_config(
                    client=client,
                    collection=collection,
                    model=model,
                    test_set=test_set,
                    topic_map=topic_map,
                    config=config,
                    top_k=args.top_k,
                )

                summary = aggregate_metrics(raw_results, config_name, model_name)
                all_summaries.append(summary)
                logger.info(f"  Hit@5: {summary.hit_rate_5:.1%} | MRR: {summary.mrr:.4f}")

        if not all_summaries:
            raise RuntimeError("No results collected — check collections exist and configs are valid")

        logger.info("Step 6/6: Generating reports")
        print_full_benchmark_report(all_summaries)
        save_benchmark_results(all_summaries, args.output_dir)
        logger.info("Benchmark complete.")

    except Exception as e:
        logger.exception(f"Benchmark failed: {e}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
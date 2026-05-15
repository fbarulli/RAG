"""
p03_benchmark.py
================
Orchestration script for the retrieval benchmark.
Run as: uv run python -m production_pipeline.p04_ingestion.p03_benchmark
"""
import argparse
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rag_pipeline.paths import Paths
from rag_pipeline.logging import get_logger

from ._benchmark_loader import load_test_set, load_topic_assignments, load_configs
from ._benchmark_metrics import evaluate_config, aggregate_metrics
from ._benchmark_report import print_full_benchmark_report, save_benchmark_results

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark retrieval configurations")
    parser.add_argument("--model", type=str, default=None, help="Single model to test")
    parser.add_argument("--config", type=str, default=None, help="Single config to test")
    parser.add_argument("--test-set", type=Path, default=None, help="Override default test set path")
    args = parser.parse_args()

    try:
        logger.info("Step 1/4: Loading test set, topics, and configs...")
        test_set = load_test_set(args.test_set if args.test_set else DEFAULT_TEST_SET)
        topic_map = load_topic_assignments(DEFAULT_TOPIC_ASSIGNMENTS)
        configs = load_configs(DEFAULT_CONFIGS)

        if args.config and args.config not in configs:
            raise ValueError(f"Config '{args.config}' not found. Available: {list(configs.keys())}")

        logger.info("Step 2/4: Connecting to Qdrant...")
        client = QdrantClient(host=DEFAULT_QDRANT_HOST, port=DEFAULT_QDRANT_PORT)

        models_to_test = [args.model] if args.model else DEFAULT_EMBEDDING_MODELS
        configs_to_test = [args.config] if args.config else list(configs.keys())
        all_summaries = []

        logger.info("Step 3/4: Running benchmark evaluations...")
        for model_name in models_to_test:
            logger.info(f"Loading model: {model_name}")
            model = SentenceTransformer(model_name)
            collection = f"faqs_{model_name.split('/')[-1].replace('-', '_')}"

            if not client.collection_exists(collection):
                logger.warning(f"Collection {collection} missing; skipping model {model_name}.")
                continue

            for cfg_name in configs_to_test:
                config = configs[cfg_name]
                logger.info(f"Evaluating: {cfg_name} on {collection}")
                
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
                logger.info(f"  -> Hit@5: {summary.hit_rate_5:.1%}, MRR: {summary.mrr:.4f}")

        if not all_summaries:
            raise RuntimeError("No results collected. Check models and collections.")

        logger.info("Step 4/4: Generating reports...")
        print_full_benchmark_report(all_summaries)
        save_benchmark_results(all_summaries, OUTPUT_DIR)
        logger.info("Benchmark complete.")

    except Exception as e:
        logger.exception(f"Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
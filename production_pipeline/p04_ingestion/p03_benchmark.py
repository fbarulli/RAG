"""
p03_benchmark.py
================
Orchestration script for the retrieval benchmark.
Run as: uv run python -m production_pipeline.p04_ingestion.p03_benchmark
"""
import argparse
import sys
import traceback
from pathlib import Path
import json

from elasticsearch import Elasticsearch
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
DEFAULT_TOPIC_ASSIGNMENTS = Path("production_pipeline/p02_eda/experiments/topic_assignments_all.json")
DEFAULT_CONFIGS = Path("configs/retrieval_configs.json")
DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_ES_HOST = "http://localhost:9200"
DEFAULT_ES_INDEX = "faqs"
DEFAULT_TOP_K = 10
DEFAULT_EMBEDDING_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-small-v2",
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2",
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
        test_set = load_test_set(
            args.test_set if args.test_set else DEFAULT_TEST_SET,
            clean_path=Paths.processed_dir() / "clean.jsonl",
        )
        logger.info(f"Loaded {len(test_set)} test queries")

        configs = load_configs(DEFAULT_CONFIGS)
        logger.info(f"Loaded {len(configs)} retrieval configs: {list(configs.keys())}")

        if args.config and args.config not in configs:
            raise ValueError(f"Config '{args.config}' not found. Available: {list(configs.keys())}")

        logger.info("Step 2/4: Connecting to Qdrant and Elasticsearch...")
        client = QdrantClient(host=DEFAULT_QDRANT_HOST, port=DEFAULT_QDRANT_PORT)
        logger.info(f"Qdrant connected at {DEFAULT_QDRANT_HOST}:{DEFAULT_QDRANT_PORT}")

        es = Elasticsearch(hosts=[DEFAULT_ES_HOST])
        if es.ping():
            logger.info(f"Elasticsearch connected at {DEFAULT_ES_HOST}")
        else:
            logger.warning("Elasticsearch not available — BM25 configs will be skipped")
            es = None

        models_to_test = [args.model] if args.model else DEFAULT_EMBEDDING_MODELS
        configs_to_test = [args.config] if args.config else list(configs.keys())
        all_summaries = []
        all_raw_results = []

        logger.info("Step 3/4: Running benchmark evaluations...")
        for model_name in models_to_test:
            try:
                logger.info(f"Loading model: {model_name}")
                if "nomic" in model_name.lower():
                    model = SentenceTransformer(model_name, trust_remote_code=True)
                else:
                    model = SentenceTransformer(model_name)

                collection = f"faqs_{model_name.split('/')[-1].replace('-', '_').replace('.', '_')}"
                logger.info(f"Target collection: {collection}")

                if not client.collection_exists(collection):
                    logger.warning(f"Collection {collection} missing; skipping model {model_name}.")
                    continue

                topic_map = load_topic_assignments(DEFAULT_TOPIC_ASSIGNMENTS, model_name)
                logger.info(f"Topic map loaded: {len(topic_map)} entries for {model_name}")

                for cfg_name in configs_to_test:
                    try:
                        config = configs[cfg_name]
                        search_type = config.get("search_type", "vector")

                        if search_type == "bm25" and es is None:
                            logger.warning(f"Skipping {cfg_name}: ES not available")
                            continue

                        logger.info(f"Evaluating: {cfg_name} ({search_type}) on {collection}")

                        results = evaluate_config(
                            client=client,
                            collection=collection,
                            model=model,
                            test_set=test_set,
                            topic_map=topic_map,
                            config=config,
                            top_k=DEFAULT_TOP_K,
                            es=es,
                            es_index=DEFAULT_ES_INDEX,
                        )

                        logger.info(f"evaluate_config returned {len(results)} results")

                        summary = aggregate_metrics(results, cfg_name, model_name)
                        all_summaries.append(summary)
                        all_raw_results.extend([
                    {
                        "query_id": r.query_id,
                        "query_text": r.query_text,
                        "expected_id": r.expected_id,
                        "course": r.course,
                        "topic": r.topic,
                        "query_type": r.query_type,
                        "hit": r.expected_id in r.hit_ids[:5],
                        "rank": r.hit_ids.index(r.expected_id) + 1 if r.expected_id in r.hit_ids else -1,
                        "top_hit_id": r.hit_ids[0] if r.hit_ids else None,
                        "config": cfg_name,
                        "model": model_name,
                    }
                    for r in results
                ])
                        logger.info(
                            f"  -> Hit@5: {summary.hit_rate_5:.1%}, "
                            f"MRR: {summary.mrr:.4f}, "
                            f"Queries: {summary.num_queries}"
                        )

                    except Exception:
                        logger.error(
                            f"Config {cfg_name} on {model_name} failed:\n{traceback.format_exc()}"
                        )
                        continue

            except Exception:
                logger.error(
                    f"Model {model_name} failed:\n{traceback.format_exc()}"
                )
                continue

        if not all_summaries:
            raise RuntimeError("No results collected. Check models and collections.")
        raw_path = OUTPUT_DIR / "benchmark_query_results.json"
        with open(raw_path, "w") as f:
            json.dump(all_raw_results, f, indent=2)
        logger.info(f"Saved {len(all_raw_results)} per-query results to {raw_path}")
        logger.info("Step 4/4: Generating reports...")
        print_full_benchmark_report(all_summaries)
        save_benchmark_results(all_summaries, OUTPUT_DIR)
        logger.info("Benchmark complete.")

    except Exception:
        logger.error(f"Benchmark failed:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
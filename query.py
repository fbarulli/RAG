"""
Quick & safe CLI to test reranking using existing modules only.
No modifications to core files yet.
"""

import sys
import argparse
from pathlib import Path
import logging

# Use src modules as requested
from rag_pipeline.paths import Paths
from rag_pipeline.logging import get_logger

logger = get_logger(__name__)

# Import existing p04 modules
from production_pipeline.p04_ingestion._benchmark_metrics.retrievers import run_vector_retrieval
from production_pipeline.p04_ingestion._benchmark_reranker import evaluate_with_reranker
from production_pipeline.p04_ingestion._benchmark_config import BenchmarkConfig


def main():
    parser = argparse.ArgumentParser(description="Test reranker standalone")
    parser.add_argument("--reranker", type=str, default=None,
                        help="Reranker name from rerankers.json (e.g. bge-reranker-base)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--query", type=str, 
                        default="What are the key differences between Python and JavaScript?")
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    config = BenchmarkConfig.from_args(argparse.Namespace())  # load defaults

    # Use first available embedding model if not specified
    model_entries = config.get_model_entries()
    model_entry = model_entries[0]
    collection = args.collection or model_entry["collection"]
    model_name = args.model or model_entry["name"]

    logger.info(f"Testing reranker: {args.reranker or 'default'}")
    logger.info(f"Collection: {collection} | Embedding model: {model_name}")

    try:
        # 1. Load embedding model (same as p03_benchmark.py)
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(model_name, trust_remote_code=model_entry.get("trust_remote_code", False))

        # 2. Get query vector
        query_vector = embedder.encode(args.query, convert_to_numpy=True).tolist()

        # 3. Run initial vector retrieval (existing function)
        client = config.qdrant_client

        initial_result = run_vector_retrieval(
            client=client,
            collection=collection,
            query_vector=query_vector,
            course_filter="",           # empty = no filter
            config={},
            top_k=args.top_k * 3        # fetch more candidates for reranking
        )

        logger.info(f"Initial retrieval: {len(initial_result.hit_ids)} documents")

        # 4. Convert to candidate format expected by reranker
        candidates = []
        for i, doc_id in enumerate(initial_result.hit_ids):
            candidates.append({
                "es_id": doc_id,
                "payload": {
                    "es_id": doc_id,
                    "question": "",   # will be filled if you fetch full payload
                    "answer": initial_result.hit_answers[i] if hasattr(initial_result, 'hit_answers') else ""
                }
            })

        # 5. Apply reranker using existing function
        reranked_ids, metrics = evaluate_with_reranker(
            query=args.query,
            retrieved_candidates=candidates,
            reranker_name=args.reranker,
            top_k=args.top_k
        )

        print("\n=== RERANKING RESULTS ===")
        print(f"Reranker used : {metrics.get('reranker_name', 'None')}")
        print(f"Rerank latency: {metrics.get('reranker_latency_ms', 0):.1f} ms")
        print(f"Top {len(reranked_ids)} IDs: {reranked_ids[:5]}...")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
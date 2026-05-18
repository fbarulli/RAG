#!/usr/bin/env python3
"""
Test different reranking strategies on a small stratified sample.
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

from rag_pipeline.paths import Paths
from rag_pipeline.logging import get_logger

logger = get_logger(__name__)

from ._benchmark_config import BenchmarkConfig
from ._benchmark_metrics.retrievers import run_entity_boosted_retrieval
from ._benchmark_reranker import evaluate_with_reranker


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RETRIEVAL_CANDIDATE_MULTIPLIER = 4

# Hardcoded per production config
PRODUCTION_CONFIG = "entity_boosted"
PRODUCTION_MODEL  = "BAAI/bge-base-en-v1.5"   
PRODUCTION_COLLECTION = "faqs_bge_base_en_v1_5"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mean_reciprocal_rank(results: List[Dict], reranker_name: str) -> float:
    rr_scores = []
    for r in results:
        relevant_id = r.get("expected_id")
        if relevant_id is None:
            continue
        top_ids = r.get(f"{reranker_name}_top_ids", [])
        try:
            rank = top_ids.index(relevant_id) + 1
            rr_scores.append(1.0 / rank)
        except ValueError:
            rr_scores.append(0.0)
    return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0


def recall_at_k(results: List[Dict], reranker_name: str, k: int) -> float:
    hits = sum(
        1 for r in results
        if r.get("expected_id") in r.get(f"{reranker_name}_top_ids", [])[:k]
    )
    return hits / len(results) if results else 0.0


# ---------------------------------------------------------------------------
# JSON serialisation (handles numpy scalars / arrays)
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark rerankers on stratified sample.")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--top-k",       type=int, default=10)
    parser.add_argument("--rerankers",   nargs="+", default=["bge-reranker-base"])
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    # Fix: pass actual parsed args, not an empty Namespace
    config = BenchmarkConfig.from_args(args)

    # Collection and model are pinned together — they must match in dimension.
    # faqs_bge_base_en_v1_5 = 768-dim, built with BAAI/bge-base-en-v1.5
    collection = PRODUCTION_COLLECTION
    logger.info(f"Using collection: '{collection}', model: '{PRODUCTION_MODEL}' (768-dim)")

    # 1. Load stratified sample
    try:
        from production_pipeline.p01_data_cleaning.p04_stratified_test_split import create_stratified_sample
        test_queries = create_stratified_sample(size=args.sample_size)
        logger.info(f"Loaded {len(test_queries)} stratified queries")
    except Exception as e:
        logger.warning(f"Stratified sampler failed: {e}. Using random sample.")
        test_queries = config.get_test_set()[:args.sample_size]

    if args.dry_run:
        logger.info(
            f"Dry-run OK — {len(test_queries)} queries, "
            f"collection: '{collection}', rerankers: {args.rerankers}"
        )
        sys.exit(0)

    # 2. Set up embedder
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(PRODUCTION_MODEL)

    # Dimension guard — catches model/collection mismatch immediately
    # rather than silently failing across every query
    test_vec = embedder.encode("dimension check", convert_to_numpy=True)
    actual_dim = len(test_vec)
    expected_dim = 768
    if actual_dim != expected_dim:
        logger.error(
            f"Dimension mismatch: '{PRODUCTION_MODEL}' produces {actual_dim}-dim vectors "
            f"but '{PRODUCTION_COLLECTION}' expects {expected_dim}-dim. Aborting."
        )
        sys.exit(1)
    logger.info(f"Embedder ready: {PRODUCTION_MODEL} ({actual_dim}-dim verified)")

    # Create Qdrant client once — make_qdrant_client() opens a connection,
    # so we reuse a single instance across all queries rather than reconnecting per query.
    qdrant_client = config.make_qdrant_client()

    # 3. Output setup
    output_dir = Paths.experiments_dir() / "reranker_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    ndj_path = output_dir / "results.ndjson"

    # Resume: skip queries already written to disk
    processed: set = set()
    if ndj_path.exists():
        with open(ndj_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        processed.add(json.loads(line)["query"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        if processed:
            logger.info(f"Resuming — {len(processed)} queries already processed, skipping.")

    # 4. Main loop
    results: List[Dict] = []

    with open(ndj_path, "a") as ndj_file:
        for i, item in enumerate(test_queries):
            query = item["query"]

            if query in processed:
                logger.debug(f"[{i+1}/{len(test_queries)}] Skipping (already processed).")
                continue

            logger.info(f"[{i+1}/{len(test_queries)}] {query[:90]}...")

            try:
                query_vector = embedder.encode(query, convert_to_numpy=True).tolist()

                # Retrieval — timed separately from reranking
                t0 = time.perf_counter()
                initial = run_entity_boosted_retrieval(
                    ner_category=item.get("ner_category"),
                    ner_primary_entity=item.get("ner_primary_entity"),
                    client=qdrant_client,
                    collection=collection,
                    query_vector=query_vector,
                    course_filter="",
                    config={},
                    top_k=args.top_k * RETRIEVAL_CANDIDATE_MULTIPLIER,
                )
                retrieval_latency_ms = round((time.perf_counter() - t0) * 1000, 1)

                # Build candidate list; guard against missing hit_answers attribute
                hit_answers = getattr(initial, "hit_answers", None) or [""] * len(initial.hit_ids)
                candidates = [
                    {
                        "es_id": doc_id,
                        "payload": {
                            "es_id": doc_id,
                            "question": query,
                            "answer": hit_answers[j],
                        },
                    }
                    for j, doc_id in enumerate(initial.hit_ids)
                ]

                result: Dict = {
                    "query":                query,
                    "topic":                item.get("course", "unknown"),
                    "expected_id":          item.get("expected_id"),
                    "retrieval_latency_ms": retrieval_latency_ms,
                }

                # Run all rerankers on the same candidates (retrieval cost paid once)
                for reranker_name in args.rerankers:
                    ids, metrics = evaluate_with_reranker(
                        query=query,
                        retrieved_candidates=candidates,
                        reranker_name=reranker_name,
                        top_k=args.top_k,
                    )
                    result[f"{reranker_name}_top_ids"]      = ids
                    result[f"{reranker_name}_latency_ms"]   = metrics.get("reranker_latency_ms", 0.0)

                results.append(result)
                ndj_file.write(json.dumps(result, cls=NumpyEncoder) + "\n")
                ndj_file.flush()

            except Exception as e:
                # exc_info=True so the traceback appears in logs — critical for debugging
                logger.error(f"[{i+1}] Failed on query '{query[:60]}': {e}", exc_info=True)
                continue

    if not results:
        logger.warning("No results collected — check collection name and Qdrant connection.")
        sys.exit(1)

    # 5. Summary
    print(f"\n=== Reranker Comparison Summary ({len(results)} queries) ===")
    for reranker_name in args.rerankers:
        latencies    = [r[f"{reranker_name}_latency_ms"] for r in results]
        avg_latency  = sum(latencies) / len(latencies) if latencies else 0.0
        mrr          = mean_reciprocal_rank(results, reranker_name)
        r_at_k       = recall_at_k(results, reranker_name, k=args.top_k)

        print(f"\n  {reranker_name}")
        print(f"    Avg rerank latency : {avg_latency:.1f} ms")
        print(f"    Avg retrieval lat  : {sum(r['retrieval_latency_ms'] for r in results) / len(results):.1f} ms")
        if any(r.get("expected_id") for r in results):
            print(f"    MRR                : {mrr:.4f}")
            print(f"    Recall@{args.top_k:<3}          : {r_at_k:.4f}")

    # Per-topic latency breakdown
    print("\n--- Latency by topic ---")
    for reranker_name in args.rerankers:
        print(f"\n  [{reranker_name}]")
        by_topic: Dict = defaultdict(list)
        for r in results:
            by_topic[r.get("topic", "unknown")].append(r[f"{reranker_name}_latency_ms"])
        for topic, lats in sorted(by_topic.items()):
            print(f"    {topic}: avg {sum(lats)/len(lats):.1f} ms ({len(lats)} queries)")

    # Write full JSON summary alongside ndjson
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"Results saved to {output_dir}")
    print(f"\nFull results  → {ndj_path}")
    print(f"JSON summary  → {summary_path}")


if __name__ == "__main__":
    main()
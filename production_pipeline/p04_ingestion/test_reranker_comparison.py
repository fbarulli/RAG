
"""
production_pipeline/p04_ingestion/test_reranker_comparison.py
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

# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark rerankers on stratified sample.")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--top-k",       type=int, default=10)
    parser.add_argument("--rerankers",   nargs="+", default=["bge-reranker-base"])
    parser.add_argument("--dry-run",     action="store_true")
    return parser.parse_args()


def _load_test_queries(config: BenchmarkConfig, sample_size: int) -> List[Dict]:
    try:
        from production_pipeline.p01_data_cleaning.p04_stratified_test_split import create_stratified_sample
        queries = create_stratified_sample(size=sample_size)
        logger.info(f"Loaded {len(queries)} stratified queries")
        return queries
    except Exception as e:
        logger.warning(f"Stratified sampler failed: {e}. Using random sample.")
        return config.get_test_set()[:sample_size]


def _build_embedder(expected_dim: int = 768):
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(PRODUCTION_MODEL)
    test_vec = embedder.encode("dimension check", convert_to_numpy=True)
    actual_dim = len(test_vec)
    if actual_dim != expected_dim:
        logger.error(
            f"Dimension mismatch: '{PRODUCTION_MODEL}' produces {actual_dim}-dim vectors "
            f"but '{PRODUCTION_COLLECTION}' expects {expected_dim}-dim. Aborting."
        )
        sys.exit(1)
    logger.info(f"Embedder ready: {PRODUCTION_MODEL} ({actual_dim}-dim verified)")
    return embedder


def _load_processed_queries(ndj_path: Path) -> set:
    """Return set of already-processed query strings for resume support."""
    processed = set()
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
    return processed


# ---------------------------------------------------------------------------
# Per-query processing
# ---------------------------------------------------------------------------

def _retrieve_candidates(
    embedder,
    qdrant_client,
    query: str,
    item: Dict,
    top_k: int,
) -> tuple:
    """Encode query, run entity-boosted retrieval, return (candidates, retrieval_latency_ms)."""
    query_vector = embedder.encode(query, convert_to_numpy=True).tolist()

    t0 = time.perf_counter()
    initial = run_entity_boosted_retrieval(
        ner_category=item.get("ner_category"),
        ner_primary_entity=item.get("ner_primary_entity"),
        client=qdrant_client,
        collection=PRODUCTION_COLLECTION,
        query_vector=query_vector,
        course_filter="",
        config={},
        top_k=top_k * RETRIEVAL_CANDIDATE_MULTIPLIER,
    )
    retrieval_latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    hit_answers = getattr(initial, "hit_answers", None) or [""] * len(initial.hit_ids)
    candidates = [
        {
            "es_id": doc_id,
            "payload": {
                "es_id":     doc_id,
                "question":  query,
                "answer":    hit_answers[j],
            },
        }
        for j, doc_id in enumerate(initial.hit_ids)
    ]
    return candidates, retrieval_latency_ms


def _run_rerankers(
    query: str,
    item: Dict,
    candidates: List[Dict],
    reranker_names: List[str],
    top_k: int,
    retrieval_latency_ms: float,
) -> Dict:
    """Run all rerankers on the same candidate set and return a result dict."""
    result: Dict = {
        "query":                query,
        "topic":                item.get("course", "unknown"),
        "expected_id":          item.get("expected_id"),
        "retrieval_latency_ms": retrieval_latency_ms,
    }
    for reranker_name in reranker_names:
        ids, metrics = evaluate_with_reranker(
            query=query,
            retrieved_candidates=candidates,
            reranker_name=reranker_name,
            top_k=top_k,
        )
        result[f"{reranker_name}_top_ids"]    = ids
        result[f"{reranker_name}_latency_ms"] = metrics.get("reranker_latency_ms", 0.0)
    return result


def _process_queries(
    test_queries: List[Dict],
    processed: set,
    embedder,
    qdrant_client,
    reranker_names: List[str],
    top_k: int,
    ndj_path: Path,
) -> List[Dict]:
    """Main per-query loop with resume support and per-item error isolation."""
    results: List[Dict] = []

    with open(ndj_path, "a") as ndj_file:
        for i, item in enumerate(test_queries):
            query = item["query"]

            if query in processed:
                logger.debug(f"[{i+1}/{len(test_queries)}] Skipping (already processed).")
                continue

            logger.info(f"[{i+1}/{len(test_queries)}] {query[:90]}...")

            try:
                candidates, retrieval_latency_ms = _retrieve_candidates(
                    embedder, qdrant_client, query, item, top_k,
                )
                result = _run_rerankers(
                    query, item, candidates, reranker_names, top_k, retrieval_latency_ms,
                )
                results.append(result)
                ndj_file.write(json.dumps(result, cls=NumpyEncoder) + "\n")
                ndj_file.flush()

            except Exception as e:
                logger.error(f"[{i+1}] Failed on query '{query[:60]}': {e}", exc_info=True)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_summary(results: List[Dict], reranker_names: List[str], top_k: int) -> None:
    print(f"\n=== Reranker Comparison Summary ({len(results)} queries) ===")
    avg_retrieval_lat = sum(r["retrieval_latency_ms"] for r in results) / len(results)

    for reranker_name in reranker_names:
        latencies   = [r[f"{reranker_name}_latency_ms"] for r in results]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        mrr         = mean_reciprocal_rank(results, reranker_name)
        r_at_k      = recall_at_k(results, reranker_name, k=top_k)

        print(f"\n  {reranker_name}")
        print(f"    Avg rerank latency : {avg_latency:.1f} ms")
        print(f"    Avg retrieval lat  : {avg_retrieval_lat:.1f} ms")
        if any(r.get("expected_id") for r in results):
            print(f"    MRR                : {mrr:.4f}")
            print(f"    Recall@{top_k:<3}          : {r_at_k:.4f}")

    print("\n--- Latency by topic ---")
    for reranker_name in reranker_names:
        print(f"\n  [{reranker_name}]")
        by_topic: Dict = defaultdict(list)
        for r in results:
            by_topic[r.get("topic", "unknown")].append(r[f"{reranker_name}_latency_ms"])
        for topic, lats in sorted(by_topic.items()):
            print(f"    {topic}: avg {sum(lats)/len(lats):.1f} ms ({len(lats)} queries)")


def _save_results(results: List[Dict], output_dir: Path, ndj_path: Path) -> None:
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Results saved to {output_dir}")
    print(f"\nFull results  → {ndj_path}")
    print(f"JSON summary  → {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args    = _parse_args()
    config  = BenchmarkConfig.from_defaults().merge_args(args)

    logger.info(f"Using collection: '{PRODUCTION_COLLECTION}', model: '{PRODUCTION_MODEL}' (768-dim)")

    test_queries = _load_test_queries(config, args.sample_size)

    if args.dry_run:
        logger.info(
            f"Dry-run OK — {len(test_queries)} queries, "
            f"collection: '{PRODUCTION_COLLECTION}', rerankers: {args.rerankers}"
        )
        sys.exit(0)

    embedder     = _build_embedder(expected_dim=768)
    qdrant_client = config.make_qdrant_client()

    output_dir = Paths.experiments_dir() / "reranker_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    ndj_path = output_dir / "results.ndjson"

    processed = _load_processed_queries(ndj_path)

    results = _process_queries(
        test_queries=test_queries,
        processed=processed,
        embedder=embedder,
        qdrant_client=qdrant_client,
        reranker_names=args.rerankers,
        top_k=args.top_k,
        ndj_path=ndj_path,
    )

    if not results:
        logger.warning("No results collected — check collection name and Qdrant connection.")
        sys.exit(1)

    _print_summary(results, args.rerankers, args.top_k)
    _save_results(results, output_dir, ndj_path)


if __name__ == "__main__":
    main()
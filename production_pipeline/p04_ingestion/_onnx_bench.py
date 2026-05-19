"""
production_pipeline/p04_ingestion/_onnx_bench.py

Main execution entry point for the ONNX Cross-Encoder matrix evaluation.
Completely driven by SRP function slices.
"""

import sys
import logging
import traceback
from typing import List, Dict, Any, Tuple
from qdrant_client import QdrantClient

from ._benchmark_config import BenchmarkConfig
from configs.benchmark_cli import create_benchmark_parser

# Clean imports from the new engine module (Zero duplicate code)
from ._onnx_bench_engine import (
    load_matrix_configs,
    extract_active_environment,
    compile_onnx_runtime_node
)

logger = logging.getLogger(__name__)


def setup_bi_encoder_context(config: BenchmarkConfig) -> Dict[str, Any]:
    """Matches active environment models to structural metadata stored in models.json."""
    model_entries = config.get_model_entries()
    if not model_entries:
        raise ValueError("No base embedding models mapped inside models.json")

    target_model, collection_name = extract_active_environment()
    
    # Locate matching configuration dictionary
    model_entry = next((m for m in model_entries if m["name"] == target_model), model_entries[0])
    model_entry["collection"] = collection_name
    return model_entry


def prepare_sliced_test_set(config: BenchmarkConfig, args: Any) -> List[Dict[str, Any]]:
    """Loads evaluation datasets and cleanly truncates data arrays to match requested sample sizes."""
    test_set = config.get_test_set()
    sample_size = getattr(args, "sample_size", 0)
    
    if sample_size > 0:
        test_set = test_set[:sample_size]
        
    print(f"[INFO] Running on {len(test_set)} queries (full dataset: {sample_size == 0})")
    return test_set


def verify_live_infrastructure(client: QdrantClient, collection_name: str) -> None:
    """Pre-flight safety check asserting database collections are ready for compute calls."""
    try:
        collection_info = client.get_collection(collection_name=collection_name)
        logger.info("Collection '%s' verified live with %d active points", 
                    collection_name, collection_info.points_count)
    except Exception as e:
        logger.error("Pre-flight cluster validation failed for '%s': %s", collection_name, e)
        sys.exit(1)


def parse_runtime_hyperparameters(args: Any) -> Dict[str, Any]:
    """Constructs uniform retrieval hyperparameter sets for the downstream loops."""
    return {
        "boost_question": 5.0,
        "boost_text": 5.0,
        "rrf_k": 60,
        "course_filter": getattr(args, 'course_filter', "machine-learning-zoomcamp"),
    }


def process_single_matrix_node(
    client: QdrantClient,
    test_set: List[Any],
    embedding_entry: Dict[str, Any],
    reranker_entry: Dict[str, Any],
    retrieval_config: Dict[str, Any]
) -> None:
    """Executes benchmark iterations for a singular, isolated cross-encoder model setup."""
    logger.info("🚀 Benchmarking Matrix Node: %s (%s)", reranker_entry["name"], reranker_entry["model"])
    
    # Generate isolated ONNX session graph
    encoder = compile_onnx_runtime_node(reranker_entry)
    
    # Assumes your global query execution block helper is imported or present
    from ._onnx_bench_utils import run_benchmark_loop  # Or wherever your active loop sits
    run_benchmark_loop(
        client=client,
        test_set=test_set,
        model_entry=embedding_entry,
        config=retrieval_config,
        encoder=encoder
    )


def execute_matrix_evaluation(
    client: QdrantClient,
    test_set: List[Any],
    embedding_entry: Dict[str, Any],
    reranker_entries: List[Dict[str, Any]],
    retrieval_config: Dict[str, Any],
    target_override: str = None
) -> None:
    """Orchestrates loop conditions over the matrix nodes, handling independent graph crashes gracefully."""
    for reranker_entry in reranker_entries:
        if target_override and reranker_entry["name"] != target_override:
            continue
            
        try:
            process_single_matrix_node(client, test_set, embedding_entry, reranker_entry, retrieval_config)
        except Exception as e:
            logger.error("Skipping node failure on cross-encoder '%s': %s", reranker_entry["name"], e)
            continue


def main() -> None:
    """Orchestration layout for the pipeline. Keeps execution steps completely synchronous."""
    try:
        # 1. Input Processing
        parser = create_benchmark_parser()
        args = parser.parse_args()
        
        # 2. Context Loading
        config = BenchmarkConfig.from_defaults().merge_args(args)
        test_set = prepare_sliced_test_set(config, args)
        embedding_entry = setup_bi_encoder_context(config)
        reranker_entries = load_matrix_configs()
        retrieval_config = parse_runtime_hyperparameters(args)
        
        # 3. Connection & Infrastructure Validation
        client = config.qdrant_client
        verify_live_infrastructure(client, embedding_entry["collection"])

        # 4. Evaluation Loop Strategy Execution
        target_override = getattr(args, 'reranker', None)
        execute_matrix_evaluation(client, test_set, embedding_entry, reranker_entries, retrieval_config, target_override)
        
        logger.info("✅ All Matrix Reranker Benchmarks complete!")
        
    except Exception as e:
        logger.error("ONNX Benchmark Pipeline critical crash: %s", e)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

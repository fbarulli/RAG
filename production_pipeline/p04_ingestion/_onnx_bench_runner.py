"""
production_pipeline/p04_ingestion/_onnx_bench_runner.py

The standalone entry point and CLI orchestrator for the ONNX Cross-Encoder matrix evaluation.
RESPONSIBILITY: Manages top-level application bootstrapping and reporting lifecycle hooks.
"""

import logging
import sys
import traceback
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from configs.benchmark_cli import create_benchmark_parser
from ._benchmark_config import BenchmarkConfig
from ._onnx_bench_config import load_matrix_configs, extract_active_environment
from ._onnx_bench_engine import verify_live_infrastructure
from ._onnx_bench import (
    prepare_sliced_test_set, 
    setup_bi_encoder_context, 
    parse_runtime_hyperparameters, 
    execute_matrix_evaluation
)
from ._benchmark_report import (
    print_full_benchmark_report,
    save_benchmark_results,
    save_performance_summary,
)

# Setup root pipeline logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def _initialize_bi_encoder(model_name: str, trust_remote: bool = False) -> SentenceTransformer:
    """RESPONSIBILITY: Instantiates the text-embedding bi-encoder model once at the root level."""
    logger.info("Initializing Baseline Embedding Model for dynamic search vectors: %s", model_name)
    return SentenceTransformer(model_name, trust_remote_code=trust_remote)


def _persist_final_reports(summaries: List[Any], output_dir: str) -> None:
    """RESPONSIBILITY: Passes metrics out to terminal layouts and saves structured reports to disk."""
    if not summaries:
        logger.warning("⚠️ Benchmarks terminated. Zero successful summaries generated.")
        return
        
    print_full_benchmark_report(summaries)
    save_benchmark_results(summaries, output_dir)
    save_performance_summary(summaries, output_dir)
    logger.info("✅ All Matrix Reranker Benchmarks complete!")


def main() -> None:
    """RESPONSIBILITY: The single execution orchestrator layout for top-level stages."""
    try:
        # 1. Input Processing & Core Configuration Context Lookup
        parser = create_benchmark_parser()
        args = parser.parse_args()
        
        config = BenchmarkConfig.from_defaults().merge_args(args)
        test_set = prepare_sliced_test_set(config, args)
        embedding_entry = setup_bi_encoder_context(config)
        reranker_entries = load_matrix_configs()
        retrieval_config = parse_runtime_hyperparameters(args)
        
        # 2. Connection, Client, and Cluster Infrastructure Setup Verification
        client = config.qdrant_client
        verify_live_infrastructure(client, embedding_entry["collection"])

        # 3. Dedicated Embedding Component Initialization Pass
        embedding_model = _initialize_bi_encoder(
            embedding_entry["name"], 
            trust_remote=embedding_entry.get("trust_remote_code", False)
        )

        # 4. Extract Dynamic Topic Mappings from the Config Instance
        # Ensures topic-level groupings match your ground-truth requirements
        try:
            topic_map = config.get_topic_map(embedding_entry["name"])
        except Exception:
            logger.warning("Could not extract topic maps via configuration object. Using None.")
            topic_map = None

        # 5. Invoke Core Matrix Evaluation Slices
        target_override = getattr(args, 'reranker', None)
        summaries = execute_matrix_evaluation(
            client=client, 
            test_set=test_set, 
            embedding_entry=embedding_entry, 
            reranker_entries=reranker_entries, 
            retrieval_config=retrieval_config, 
            embedding_model=embedding_model, 
            topic_map=topic_map,  # Pass topic_map context through cleanly
            target_override=target_override
        )
        
        # 6. Output Management Persistence Slices
        _persist_final_reports(summaries, config.output_dir)
        
    except Exception as e:
        logger.error("ONNX Benchmark Runner critical application crash: %s", e)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

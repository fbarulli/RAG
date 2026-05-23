"""
rag_pipeline/ingestion/_onnx_bench_runner.py

Standalone CLI orchestrator for the ONNX Cross-Encoder matrix evaluation.
RESPONSIBILITY: Manages top-level application bootstrapping and reporting lifecycle hooks.
"""
import logging
import sys
import traceback
from typing import Any, Dict, List, Optional
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from configs.benchmark_cli import create_benchmark_parser
from .benchmark_config import BenchmarkConfig
from .onnx_bench_config import load_matrix_configs
from .onnx_bench import prepare_sliced_test_set, setup_bi_encoder_context, parse_runtime_hyperparameters, execute_matrix_evaluation
from .benchmark_report import print_full_benchmark_report, save_benchmark_results, save_performance_summary
from .onnx_bench_failure_analysis import save_failure_analysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def _bootstrap_config(args: Any) -> tuple:
    """RESPONSIBILITY: Parses CLI args into config, test set, embedding entry, reranker entries, and retrieval config."""
    config = BenchmarkConfig.from_defaults().merge_args(args)
    test_set = prepare_sliced_test_set(config, args)
    embedding_entry = setup_bi_encoder_context(config)
    reranker_entries = load_matrix_configs()
    retrieval_config = parse_runtime_hyperparameters(args)
    return (config, test_set, embedding_entry, reranker_entries, retrieval_config)


def _connect_and_verify(config: BenchmarkConfig, collection: str) -> QdrantClient:
    """RESPONSIBILITY: Materialises the Qdrant client and smoke-tests live infrastructure."""
    client = config.qdrant_client
    collections = [c.name for c in client.get_collections().collections]
    if collection not in collections:
        raise RuntimeError(f"Collection '{collection}' not found in Qdrant. Available: {collections}")
    logger.info("✅ Qdrant connection verified — collection '%s' exists", collection)
    return client


def _initialize_bi_encoder(model_name: str, trust_remote: bool = False) -> SentenceTransformer:
    """RESPONSIBILITY: Instantiates the text-embedding bi-encoder model once at the root level."""
    logger.info('Initializing Baseline Embedding Model: %s', model_name)
    return SentenceTransformer(model_name, trust_remote_code=trust_remote)


def _resolve_topic_map(config: BenchmarkConfig, model_name: str) -> Dict[str, Any] | None:
    """RESPONSIBILITY: Extracts topic-level groupings from config, with graceful fallback to None."""
    try:
        return config.get_topic_map(model_name)
    except Exception:
        logger.warning('Could not extract topic maps. Using None.')
        return None


def _persist_final_reports(summaries: List[Any], output_dir: Optional[Path]) -> None:
    """RESPONSIBILITY: Passes metrics out to terminal layouts and saves structured reports to disk."""
    if output_dir is None:
        raise ValueError("output_dir is not set — cannot save reports")
    if not summaries:
        logger.warning('⚠️ Benchmarks terminated. Zero successful summaries generated.')
        return
    metric_summaries = [s for s, _, _ in summaries]
    results_by_reranker = {name: results for _, results, name in summaries}
    print_full_benchmark_report(metric_summaries)
    save_benchmark_results(metric_summaries, output_dir)
    save_performance_summary(metric_summaries, output_dir)
    save_failure_analysis(results_by_reranker, metric_summaries, output_dir)
    logger.info('✅ All Matrix Reranker Benchmarks complete!')


def main() -> None:
    """RESPONSIBILITY: The single execution orchestrator for top-level stages."""
    try:
        args = create_benchmark_parser().parse_args()
        config, test_set, embedding_entry, reranker_entries, retrieval_config = _bootstrap_config(args)
        client = _connect_and_verify(config, embedding_entry['collection'])
        embedding_model = _initialize_bi_encoder(embedding_entry['name'], trust_remote=embedding_entry.get('trust_remote_code', False))
        topic_map = _resolve_topic_map(config, embedding_entry['name'])
        summaries = execute_matrix_evaluation(
            client=client,
            test_set=test_set,
            embedding_entry=embedding_entry,
            reranker_entries=reranker_entries,
            retrieval_config=retrieval_config,
            embedding_model=embedding_model,
            topic_map=topic_map,
            target_override=getattr(args, 'reranker', None),
            cache_dir=config.cache_dir,
            reset=config.reset,
        )
        _persist_final_reports(summaries, config.output_dir)
    except Exception as e:
        logger.error('ONNX Benchmark Runner critical crash: %s', e)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

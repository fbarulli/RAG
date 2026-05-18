"""
p04_ingestion/p03_benchmark.py - Simplified using centralized config.
"""

import sys
import traceback
from pathlib import Path

from ._benchmark_metrics.evaluation import evaluate_config
from ._benchmark_metrics.aggregation import aggregate_metrics
from ._benchmark_report import print_full_benchmark_report, save_benchmark_results, save_performance_summary
from ._benchmark_config import BenchmarkConfig
from configs.benchmark_cli import create_benchmark_parser
from rag_pipeline.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = create_benchmark_parser()
    args = parser.parse_args()
    
    # Load config from defaults + CLI overrides
    config = BenchmarkConfig.from_args(args)

    cli_config = BenchmarkConfig.from_args(args)
    
    # Merge CLI overrides (non-None values take precedence)
    for key, value in cli_config.__dict__.items():
        if value is not None:
            setattr(config, key, value)
    
    try:
        logger.info("Loading test set...")
        test_set = config.get_test_set()
        
        logger.info("Loading configs...")
        configs = config.get_configs()
        
        logger.info("Loading model...")
        model_entry = config.get_model_entries()[0]
        model_name = model_entry["name"]
        
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, trust_remote_code=model_entry.get("trust_remote_code", False))
        
        topic_map = config.get_topic_map(model_name)
        
        logger.info(f"Evaluating {model_name}...")
        all_results = []
        
        for cfg_name, cfg in configs.items():
            logger.info(f"  Config: {cfg_name}")
            results = evaluate_config(
                client=config.qdrant_client,
                collection=model_entry["collection"],
                model=model,
                test_set=test_set,
                topic_map=topic_map,
                config=cfg,
                top_k=config.top_k,
                es=config.es_client,
                es_index=config.es_index,
                encode_batch_size=config.encode_batch_size,
            )
            summary = aggregate_metrics(results, cfg_name, model_name)
            all_results.append((cfg_name, summary, results))
            
            logger.info(f"    Hit@5={summary.hit_rate_5:.1%} MRR={summary.mrr:.4f}")
        
        # Generate reports
        summaries = [s for _, s, _ in all_results]
        print_full_benchmark_report(summaries)
        save_benchmark_results(summaries, config.output_dir)
        save_performance_summary(summaries, config.output_dir)
        
        logger.info("Benchmark complete!")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
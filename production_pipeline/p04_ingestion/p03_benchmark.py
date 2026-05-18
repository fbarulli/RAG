"""
p04_ingestion/p03_benchmark.py
"""

import sys
import traceback
from rag_pipeline.logging import get_logger

from ._benchmark_metrics.evaluation import evaluate_config
from ._benchmark_metrics.aggregation import aggregate_metrics
from ._benchmark_report import print_full_benchmark_report, save_benchmark_results, save_performance_summary
from ._benchmark_config import BenchmarkConfig
from ._benchmark_reranker import evaluate_with_reranker
from configs.benchmark_cli import create_benchmark_parser

logger = get_logger(__name__)


def main():
    parser = create_benchmark_parser()
    args = parser.parse_args()
    
    # Initialize the BenchmarkConfig instance safely from the parsed CLI args
    config = BenchmarkConfig.from_args(args)
    
    # Load and sample the evaluation dataset completely at the root context level
    test_set = config.get_test_set()
    if hasattr(args, "sample_size") and args.sample_size > 0:
        test_set = test_set[:args.sample_size]
        
    # Fixed the quote boundary syntax crash by alternating internal parameters with single quotes
    print(f"[INFO] Running on {len(test_set)} queries (full dataset: {getattr(args, 'sample_size', 0) == 0})")
    
    try:
        all_configs = config.get_configs()
            
        if args.config:
            if args.config in all_configs:
                configs = {args.config: all_configs[args.config]}
                logger.info(f"Running single config: {args.config}")
            else:
                raise KeyError(f"Config '{args.config}' not found.")
        else:
            configs = all_configs

        # Model handling - default to first if none provided
        model_entries = config.get_model_entries()
        if not model_entries:
            raise ValueError("No models found in models.json")

        model_entry = model_entries[0]   # default
        
        if args.model:
            for m in model_entries:
                if m["name"] == args.model:
                    model_entry = m
                    break

        model_name = model_entry["name"]
        
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, 
                                  trust_remote_code=model_entry.get("trust_remote_code", False))
        
        topic_map = config.get_topic_map(model_name)
        
        logger.info(f"Evaluating {model_name} | Config: {list(configs.keys())[0]}")

        all_results = []
        for cfg_name, cfg in configs.items():
            logger.info(f"  Running: {cfg_name}")
            logger.info(f"Config dict: {cfg}")
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
            logger.info(f"    Hit@5={summary.hit_rate_5:.1%}  MRR={summary.mrr:.4f}")

        # Reports
        summaries = [s for _, s, _ in all_results]
        print_full_benchmark_report(summaries)
        save_benchmark_results(summaries, config.output_dir)
        save_performance_summary(summaries, config.output_dir)
        
        logger.info("✅ Benchmark complete!")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

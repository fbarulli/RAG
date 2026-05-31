"""
p04_ingestion/benchmark.py
"""
import json
import sys
import traceback
from pathlib import Path

from rag_pipeline.logging import get_logger
from .benchmark_metrics_data.evaluation import evaluate_config
from .benchmark_metrics_data.aggregation import aggregate_metrics
from .benchmark_report import print_full_benchmark_report
from .benchmark_persistence import save_benchmark_results, save_performance_summary
from .benchmark_config import BenchmarkConfig
from configs.benchmark_cli import create_benchmark_parser
from rag_pipeline.mlflow.tracking import log_benchmark_run

logger = get_logger(__name__)


def _parse_config(args) -> tuple[BenchmarkConfig, dict]:
    """Build BenchmarkConfig and resolve the retrieval configs to run."""
    config = BenchmarkConfig.from_defaults().merge_args(args)
    all_configs = config.get_configs()
    if args.config:
        if args.config not in all_configs:
            raise KeyError(f"Config '{args.config}' not found. Available: {sorted(all_configs)}")
        logger.info(f'Running single config: {args.config}')
        return config, {args.config: all_configs[args.config]}
    keys = getattr(args, 'configs', None)
    if keys:
        # Accept both --configs a b c and --configs a,b,c
        flat = [k for entry in keys for k in entry.split(',')]
        missing = [k for k in flat if k not in all_configs]
        if missing:
            raise KeyError(f"Config(s) not found: {missing}. Available: {sorted(all_configs)}")
        logger.info(f'Running configs subset: {flat}')
        return config, {k: all_configs[k] for k in flat}
    return config, all_configs


def _resolve_model_entry(config: BenchmarkConfig, args) -> dict:
    """Return the model registry entry to use for this run."""
    model_entries = config.get_model_entries()
    if not model_entries:
        raise ValueError('No models found in models.json')
    entry = model_entries[0]
    if args.model:
        for m in model_entries:
            if m['name'] == args.model:
                entry = m
                break
    if getattr(args, 'collection', None):
        entry = dict(entry)
        entry['collection'] = args.collection
    return entry


def _load_embedding_model(model_entry: dict):
    """Instantiate the SentenceTransformer embedding model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        model_entry['name'],
        trust_remote_code=model_entry.get('trust_remote_code', False),
    )


def _load_test_set(config: BenchmarkConfig, args) -> list[dict]:
    """Load and optionally truncate the test set."""
    test_set = config.get_test_set()
    if hasattr(args, 'query_type') and args.query_type:
        test_set = [q for q in test_set if q.get('query_type') in args.query_type]
        print(f'[INFO] Filtered to query types {args.query_type}: {len(test_set)} queries')
    if hasattr(args, 'sample_size') and args.sample_size > 0:
        test_set = test_set[:args.sample_size]
    print(f"[INFO] Running on {len(test_set)} queries (full dataset: {getattr(args, 'sample_size', 0) == 0})")
    return test_set


def _save_query_results(results: list, cfg_name: str, output_dir: Path) -> None:
    """Serialize raw QueryResults to JSONL for downstream reranker benchmarking."""
    out = output_dir / f"{cfg_name}_query_results.jsonl"
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps({
                "query_id":                r.query_id,
                "query_text":              r.query_text,
                "expected_id":             r.expected_id,
                "course":                  r.course,
                "topic":                   r.topic,
                "subtopic":                r.subtopic,
                "query_type":              r.query_type,
                "hit_ids":                 list(r.hit_ids),
                "hit_scores":              list(r.hit_scores),
                "latency_ms":              r.latency_ms,
                "reranker_latency_ms":     r.reranker_latency_ms,
                "code_integrity_ref":      r.code_integrity_ref,
                "code_integrity_retrieved": r.code_integrity_retrieved,
            }) + "\n")
    logger.info(f"QueryResults saved → {out}")


def _run_config(cfg_name: str, cfg: dict, config: BenchmarkConfig,
                model, model_entry: dict, test_set: list, topic_map: dict) -> tuple:
    """Evaluate one retrieval config and return (cfg_name, summary, results)."""
    logger.info(f'  Running: {cfg_name}')
    logger.info(f'Config dict: {cfg}')
    logger.info(f'Collection: {model_entry["collection"]}')
    from rag_pipeline.core.paths import Paths
    collection = Paths.collection_for_model(model_entry['name'], config.encode_mode)
    results = evaluate_config(
        client=config.qdrant_client,
        collection=collection,
        model=model,
        test_set=test_set,
        topic_map=topic_map,
        config=cfg,
        top_k=config.top_k,
        es=config.es_client,
        es_index=config.es_index,
        encode_batch_size=config.encode_batch_size,
        cache_dir=config.cache_dir,
        model_name=model_entry['name'],
    )
    summary = aggregate_metrics(results, cfg_name, model_entry['name'])
    logger.info(f'    Hit@5={summary.hit_rate_5:.1%}  MRR={summary.mrr:.4f}')
    return cfg_name, summary, results


def main():
    parser = create_benchmark_parser()
    args = parser.parse_args()

    try:
        config, configs      = _parse_config(args)
        test_set             = _load_test_set(config, args)
        model_entry          = _resolve_model_entry(config, args)
        model                = _load_embedding_model(model_entry)
        topic_map            = config.get_topic_map(model_entry['name'])

        logger.info(f"Evaluating {model_entry['name']} | Configs: {list(configs.keys())}")

        summaries = []
        results_map = {}
        for cfg_name, cfg in configs.items():
            name, summary, results = _run_config(
                cfg_name, cfg, config, model, model_entry, test_set, topic_map
            )
            _save_query_results(results, name, config.output_dir)
            summaries.append(summary)
            results_map[name] = results
            log_benchmark_run(name, cfg, summary, results, model_entry, encode_mode=config.encode_mode.value, tags={"collection": model_entry["collection"]})

        print_full_benchmark_report(summaries, query_results_map=results_map)
        save_benchmark_results(summaries, config.output_dir)
        save_performance_summary(summaries, config.output_dir)
        logger.info('✅ Benchmark complete!')

    except Exception as e:
        logger.error(f'Benchmark failed: {e}')
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
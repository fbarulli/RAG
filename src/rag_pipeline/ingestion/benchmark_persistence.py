"""
benchmark_persistence.py
Handles the storage, merging, and historical tracking of benchmark results.
"""
import json
from pathlib import Path
from typing import Any, Optional
from rag_pipeline.logging import get_logger
from .benchmark_types import MetricSummary

logger = get_logger(__name__)

def _load_existing_summaries(results_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load existing results and index by (model, config) for upserting."""
    if not results_path.exists():
        return {}
    try:
        with results_path.open(encoding='utf-8') as f:
            rows = json.load(f)
        return {(row['model_name'], row['config_name']): row for row in rows}
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f'Could not load existing results from {results_path}: {e}. Starting fresh.')
        return {}

def _merge_summaries(existing: dict[tuple[str, str], dict[str, Any]], new_summaries: list[MetricSummary]) -> list[MetricSummary]:
    """Upsert new results into existing history."""
    merged = dict(existing)
    for s in new_summaries:
        merged[s.model_name, s.config_name] = s.model_dump()
    return [MetricSummary(**row) for row in merged.values()]

def save_benchmark_results(summaries: list[MetricSummary], output_dir: Path) -> None:
    """Save results to JSON using an upsert scheme to preserve history."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'benchmark_results.json'
    
    existing = _load_existing_summaries(results_path)
    merged = _merge_summaries(existing, summaries)
    
    with results_path.open('w', encoding='utf-8') as f:
        json.dump([s.model_dump() for s in merged], f, indent=2, ensure_ascii=False)
    logger.info(f'Saved benchmark results to {results_path}')

def save_performance_summary(summaries: list[MetricSummary], output_dir: Path) -> None:
    """Create the canonical ranking file (reranker_benchmark_performance.json)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [s.model_dump() for s in summaries if s.num_queries > 0]
    if not all_rows: return
    
    # Find best config per model
    best_per_model = {}
    for row in all_rows:
        m = row['model_name']
        if m not in best_per_model or row['mrr'] > best_per_model[m]['mrr']:
            best_per_model[m] = row
            
    winner = max(best_per_model.values(), key=lambda x: x['mrr'])
    
    payload = {
        'winner_model': winner['model_name'],
        'winner_config': winner['config_name'],
        'winner_mrr': winner['mrr'],
        'best_per_model': sorted(best_per_model.values(), key=lambda r: r['mrr'], reverse=True),
        'all_results': sorted(all_rows, key=lambda r: r['mrr'], reverse=True)
    }
    
    with (output_dir / 'reranker_benchmark_performance.json').open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


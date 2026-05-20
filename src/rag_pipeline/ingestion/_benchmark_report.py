"""
def print_full_benchmark_report(summaries: list[MetricSummary]) -> None:
    Print benchmark results in clean table format, including the overall best config per model and per-config comparison matrices.
    I/O: summaries (list[MetricSummary]) -> None

def save_benchmark_results(summaries: list[MetricSummary], output_dir: Path) -> None:
    Save per-run benchmark results to JSON and a human-readable text summary using an upsert scheme indexed by model and config.
    I/O: summaries (list[MetricSummary]), output_dir (Path) -> None
_benchmark_report.py
====================
Print reports and save results for the retrieval benchmark.

Single responsibility: human-readable output + JSON serialization.
No metric computation, no data loading, no retrieval logic.

Functions:
    print_full_benchmark_report(summaries)  -> None
    save_benchmark_results(summaries, output_dir) -> None
    save_performance_summary(summaries, output_dir) -> None
"""
from __future__ import annotations
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any
from rag_pipeline.core.logging import get_logger
from ._benchmark_types import MetricSummary
logger = get_logger(__name__)

def _format_summary(s: MetricSummary) -> list[str]:
    """Format a single MetricSummary into display lines."""
    lines = [f'Config: {s.config_name} | Model: {s.model_name}']
    if s.topic is not None:
        lines.append(f'  Topic: {s.topic} | Subtopic: {s.subtopic}')
    ret_integrity = f'{s.avg_code_integrity_retrieved:.1%}' if s.avg_code_integrity_retrieved is not None else 'N/A'
    lines += [f'  Queries: {s.num_queries}', f'  Hit Rate @1: {s.hit_rate_1:.1%} | @3: {s.hit_rate_3:.1%} | @5: {s.hit_rate_5:.1%} | @10: {s.hit_rate_10:.1%}', f'  MRR: {s.mrr:.4f} | NDCG@10: {s.ndcg_10:.4f}', f'  Latency: p50={s.latency_p50:.1f}ms | p95={s.latency_p95:.1f}ms | p99={s.latency_p99:.1f}ms', f'  Code Integrity (Ref): {s.avg_code_integrity_ref:.1%} | (Retrieved): {ret_integrity}', f'  Cross-Course Contamination: {s.cross_course_contamination:.2%}', f'  Rank Std Dev: {s.rank_std:.2f}   (lower is better)', f'  Failures (Hit@10=0): {s.failure_count}']
    if s.avg_failure_similarity is not None:
        lines.append(f'  Avg Failure Similarity: {s.avg_failure_similarity:.4f}')
    return lines

def _sort_summaries(summaries: list[MetricSummary]) -> list[MetricSummary]:
    """Sort summaries deterministically for consistent reporting."""
    return sorted(summaries, key=lambda s: (s.config_name, s.model_name, s.topic or -1, s.subtopic or -1))

def _winner_key(row: dict[str, Any]) -> tuple[float, float, float]:
    """Tie-breaking sort key: MRR → Hit@1 → NDCG@10."""
    return (row.get('mrr', 0.0), row.get('hit_rate_1', 0.0), row.get('ndcg_10', 0.0))

def _short_model_name(full_name: str, max_len: int=35) -> str:
    """Extract short model name from full path."""
    name = full_name.split('/')[-1]
    return name[:max_len] if len(name) > max_len else name

def _load_existing_summaries(results_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Load existing benchmark_results.json and index by (model_name, config_name).
    Returns an empty dict if the file doesn't exist or is malformed.
    """
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
    """
    Upsert new_summaries into existing by (model_name, config_name).
    New results overwrite same-key existing rows; all other rows are kept.
    """
    merged = dict(existing)
    for s in new_summaries:
        merged[s.model_name, s.config_name] = asdict(s)
    return _sort_summaries([MetricSummary(**row) for row in merged.values()])

def print_winner_table(summaries: list[MetricSummary]) -> None:
    """Print a compact winner table showing best config per model."""
    if not summaries:
        print('No results to display.')
        return
    best_per_model: dict[str, MetricSummary] = {}
    for s in summaries:
        if s.model_name not in best_per_model or s.mrr > best_per_model[s.model_name].mrr:
            best_per_model[s.model_name] = s
    print('\n' + '=' * 95)
    print('BEST CONFIG PER MODEL (by MRR)')
    print('=' * 95)
    print(f"{'Model':<35} {'Config':<22} {'H@1':>8} {'H@5':>8} {'MRR':>8} {'p50(ms)':>9}")
    print(f"{'-' * 35} {'-' * 22} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 9}")
    for s in sorted(best_per_model.values(), key=lambda x: x.mrr, reverse=True):
        model_short = _short_model_name(s.model_name, 34)
        print(f'{model_short:<35} {s.config_name:<22} {s.hit_rate_1:>7.1%} {s.hit_rate_5:>7.1%} {s.mrr:>7.4f} {s.latency_p50:>8.1f}')
    winner = max(best_per_model.values(), key=lambda x: x.mrr)
    print(f"\n{'=' * 95}")
    print(f'🏆 OVERALL WINNER: {_short_model_name(winner.model_name)} with {winner.config_name}')
    print(f'   MRR: {winner.mrr:.4f} | H@1: {winner.hit_rate_1:.1%} | H@5: {winner.hit_rate_5:.1%}')
    print('=' * 95)

def print_comparison_table(summaries: list[MetricSummary]) -> None:
    """Print a compact comparison table across all model/config combinations."""
    if not summaries:
        print('No results to display.')
        return
    by_config: dict[str, list[MetricSummary]] = defaultdict(list)
    for s in summaries:
        by_config[s.config_name].append(s)
    for config_name, config_summaries in by_config.items():
        print(f"\n{'=' * 100}")
        print(f'Config: {config_name}')
        print(f"{'=' * 100}")
        print(f"{'Model':<35} {'H@1':>8} {'H@5':>8} {'H@10':>9} {'MRR':>8} {'p50(ms)':>9} {'Fail':>6}")
        print(f"{'-' * 35} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 8} {'-' * 9} {'-' * 6}")
        for s in sorted(config_summaries, key=lambda x: x.mrr, reverse=True):
            model_short = _short_model_name(s.model_name, 34)
            print(f'{model_short:<35} {s.hit_rate_1:>7.1%} {s.hit_rate_5:>7.1%} {s.hit_rate_10:>8.1%} {s.mrr:>7.4f} {s.latency_p50:>8.1f} {s.failure_count:>6}')
        best = max(config_summaries, key=lambda x: x.mrr)
        print(f'\n🏆 Best for {config_name}: {_short_model_name(best.model_name)} (MRR={best.mrr:.4f})')

def print_full_benchmark_report(summaries: list[MetricSummary]) -> None:
    """
    Print benchmark results in clean table format.

    Shows:
        1. Winner table (best config per model)
        2. Per-config comparison tables
    """
    print('\n' + '=' * 100)
    print('RETRIEVAL BENCHMARK REPORT')
    print('=' * 100)
    print_winner_table(summaries)
    print_comparison_table(summaries)

def save_benchmark_results(summaries: list[MetricSummary], output_dir: Path) -> None:
    """
    Save per-run benchmark results to JSON and a human-readable text summary.

    Results are upserted by (model_name, config_name): new rows overwrite
    existing rows with the same key; all other existing rows are preserved.
    Running a single model updates only that model's rows.
    Running all models replaces all rows for every model in that run.

    Also calls ``save_performance_summary`` to update the canonical
    ``benchmark_performance.json`` rankings file.

    Files written
    -------------
    ``benchmark_results.json``     — full results for every model/config combination
    ``benchmark_summary.txt``      — human-readable equivalent
    ``benchmark_performance.json`` — ranked view with winner (via save_performance_summary)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'benchmark_results.json'
    existing = _load_existing_summaries(results_path)
    merged = _merge_summaries(existing, summaries)
    with results_path.open('w', encoding='utf-8') as f:
        json.dump([asdict(s) for s in merged], f, indent=2, ensure_ascii=False)
    logger.info(f'Saved benchmark results: {results_path}')
    summary_path = output_dir / 'benchmark_summary.txt'
    with summary_path.open('w', encoding='utf-8') as f:
        f.write('RETRIEVAL BENCHMARK SUMMARY\n')
        f.write('=' * 120 + '\n\n')
        for s in merged:
            if s.num_queries == 0:
                continue
            f.write('\n'.join(_format_summary(s)) + '\n\n')
    logger.info(f'Saved benchmark summary: {summary_path}')
    save_performance_summary(merged, output_dir)

def save_performance_summary(summaries: list[MetricSummary], output_dir: Path) -> None:
    """
    Save ``benchmark_performance.json`` — a consolidated rankings view.

    Contains all model/config combinations ranked by MRR, per-model best-config
    rows, and an overall winner (tie-broken by Hit@1 then NDCG@10).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = [asdict(s) for s in summaries if s.num_queries > 0]
    best_per_model: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        m = row['model_name']
        if m not in best_per_model or row['mrr'] > best_per_model[m]['mrr']:
            best_per_model[m] = row
    winner = max(best_per_model.values(), key=_winner_key, default=None)
    payload: dict[str, Any] = {'winner_model': winner['model_name'] if winner else None, 'winner_config': winner['config_name'] if winner else None, 'winner_mrr': winner['mrr'] if winner else None, 'best_per_model': sorted(best_per_model.values(), key=lambda r: r['mrr'], reverse=True), 'all_results': sorted(all_rows, key=lambda r: r['mrr'], reverse=True)}
    perf_path = output_dir / 'benchmark_performance.json'
    with perf_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f'Saved performance summary: {perf_path}')
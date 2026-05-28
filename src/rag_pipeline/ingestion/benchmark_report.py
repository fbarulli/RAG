"""
benchmark_report.py
Simplified reporting. Focuses on summary metrics.
Trace data remains available in .jsonl result files.
"""
from typing import Optional
from .benchmark_types import MetricSummary, QueryResult

def _short_model_name(full_name: str) -> str:
    return full_name.split('/')[-1]

def print_full_benchmark_report(summaries: list[MetricSummary], query_results_map: Optional[dict[str, list[QueryResult]]] = None) -> None:
    if not summaries:
        print("\nNo results.")
        return

    print('\n' + '=' * 110)
    print(f'{"RETRIEVAL BENCHMARK REPORT":^110}')
    print('=' * 110)

    # 1. Performance Table
    print(f"\n{'Model':<30} {'Config':<20} {'H@1':>8} {'H@5':>8} {'MRR':>8} {'p50(ms)':>10}")
    print(f"{'-' * 30} {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10}")
    
    sorted_summaries = sorted(summaries, key=lambda s: s.mrr, reverse=True)
    for s in sorted_summaries:
        print(f'{_short_model_name(s.model_name):<30} {s.config_name:<20} {s.hit_rate_1:>7.1%} {s.hit_rate_5:>7.1%} {s.mrr:>7.4f} {s.latency_p50:>9.1f}')

    # 2. Overall Winner
    winner = sorted_summaries[0]
    print(f"\n{'=' * 110}")
    print(f"OVERALL WINNER: {_short_model_name(winner.model_name)} | {winner.config_name} (MRR: {winner.mrr:.4f})")
    print('=' * 110)

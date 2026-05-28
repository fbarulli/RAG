"""
benchmark_report.py
Ultra-minimal reporting.
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

    # 3. Per-Query Type Trace
    if query_results_map:
        print(f"\n{'QUERY TYPE TRACE (Hit@1)':^110}")
        configs = list(query_results_map.keys())
        print(f"{'Query Type':<25} | {'Expected':<12} | " + " | ".join([f"{cfg:<15}" for cfg in configs]))
        print("-" * 110)
        
        # Use the first config as the reference list
        master_results = query_results_map[configs[0]]
        for res in master_results:
            row_hits = []
            for cfg in configs:
                # Find the result for this specific query ID in this config's list
                cfg_res = next((r for r in query_results_map[cfg] if r.query_id == res.query_id), None)
                if cfg_res and cfg_res.hit_ids and cfg_res.hit_ids[0] == res.expected_id:
                    row_hits.append("HIT")
                else:
                    row_hits.append("MISS")
            
            # res.query_type is 'original', 'grounded_analyst', etc.
            print(f"{res.query_type:<25} | {str(res.expected_id):<12} | " + " | ".join([f"{h:<15}" for h in row_hits]))

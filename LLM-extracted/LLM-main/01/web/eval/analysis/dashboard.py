"""
eval/analysis/dashboard.py
===========================
Unified dashboard for all evaluation results.
Handles both self-retrieval benchmarks and test_variations.py output.

Run:    uv run python eval/analysis/dashboard.py
"""
import sys, os, json, glob
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

import numpy as np
from collections import defaultdict

RESULTS_DIR = 'experiments/results'


def load_latest_variations():
    files = sorted(glob.glob(f'{RESULTS_DIR}/variations_*.json'))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def show_variations_dashboard():
    """Dashboard for test_variations.py output."""
    data = load_latest_variations()
    if not data:
        print("No variations results. Run: uv run python eval/benchmarks/test_variations.py")
        return
    
    configs = data['configs']
    
    print(f"\n{'='*100}")
    print(f"RETRIEVAL VARIATIONS DASHBOARD")
    print(f"Queries: {data['metadata']['total_queries']} | Model: {data['metadata']['embedding_model']}")
    print(f"{'='*100}")
    
    # Full metrics
    print(f"\n{'Retriever':<22} {'Hit':>7} {'R@1':>7} {'R@5':>7} {'MRR':>7} {'NDCG':>7} {'Fail':>5} {'P50ms':>7} {'P95ms':>7}")
    print(f"{'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*7} {'-'*7}")
    for c in configs:
        print(f"{c['name']:<22} {c['hit_rate']:>6.2%} {c['recall_at_k']['1']:>6.2%} "
              f"{c['recall_at_k']['5']:>6.2%} {c['mrr']:>7.4f} {c['ndcg@10']:>7.4f} "
              f"{c['failure_count']:>5} {c['p50_latency']:>6.1f} {c['p95_latency']:>6.1f}")
    
    # Per-strategy
    best = max(configs, key=lambda c: c['recall_at_k']['5'])
    if best.get('per_strategy'):
        print(f"\n{'='*60}")
        print(f"PER-STRATEGY ({best['name']})")
        for s, m in sorted(best['per_strategy'].items()):
            print(f"  {s:<25} R@5={m['recall_at_k']['5']:.1%} ({m['found']}/{m['total']})")
    
    # Per-course
    if best.get('per_course'):
        print(f"\n{'='*60}")
        print(f"PER-COURSE ({best['name']})")
        for course, m in sorted(best['per_course'].items()):
            print(f"  {course:<30} R@5={m['recall_at_k']['5']:.1%}")
    
    # Conclusions
    print(f"\n{'='*60}")
    print("CONCLUSIONS")
    best = max(configs, key=lambda c: c['recall_at_k']['5'])
    worst = min(configs, key=lambda c: c['recall_at_k']['5'])
    fastest = min(configs, key=lambda c: c['p50_latency'])
    
    print(f"  1. Best: {best['name']} (R@5={best['recall_at_k']['5']:.1%}, NDCG={best['ndcg@10']:.4f})")
    print(f"  2. Fastest: {fastest['name']} (P50={fastest['p50_latency']}ms)")
    
    # BM25 vs Vector
    bm25 = next((c for c in configs if c['name'] == 'bm25_balanced'), None)
    vec = next((c for c in configs if c['name'] == 'vector_cosine'), None)
    if bm25 and vec:
        print(f"  3. Vector vs BM25: +{vec['recall_at_k']['5'] - bm25['recall_at_k']['5']:.1%} R@5, "
              f"{vec['p50_latency']/bm25['p50_latency']:.1f}x slower")
    
    # Hybrid
    hybrid = next((c for c in configs if c['name'] == 'hybrid_70_30_vec'), None)
    if hybrid and vec:
        print(f"  4. Hybrid (70/30) vs Vector: +{hybrid['recall_at_k']['5'] - vec['recall_at_k']['5']:.1%} R@5")
    
    # Reranker
    rerank = next((c for c in configs if 'reranked' in c['name']), None)
    if rerank and vec:
        print(f"  5. Reranking: {rerank['recall_at_k']['5'] - vec['recall_at_k']['5']:+.1%} R@5, "
              f"{rerank['p50_latency']/vec['p50_latency']:.0f}x slower")

    # Hardest strategy
    if best.get('per_strategy'):
        strategies = best['per_strategy']
        hardest = min(strategies, key=lambda s: strategies[s]['recall_at_k']['5'])
        easiest = max(strategies, key=lambda s: strategies[s]['recall_at_k']['5'])
        print(f"  6. Hardest query type: {hardest} ({strategies[hardest]['recall_at_k']['5']:.1%})")
        print(f"  7. Easiest query type: {easiest} ({strategies[easiest]['recall_at_k']['5']:.1%})")


def show_dashboard():
    """Entry point for notebook and CLI."""
    show_variations_dashboard()

if __name__ == '__main__':
    show_variations_dashboard()

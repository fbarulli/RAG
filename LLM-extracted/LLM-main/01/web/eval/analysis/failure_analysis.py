"""
eval/analysis/failure_analysis.py
==================================
Deep-dive failure analysis for retrieval evaluation.
Called from notebook: from eval.analysis.failure_analysis import analyze_failures

Usage:
    analyze_failures()
"""
import sys, os, json, glob
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

from elasticsearch import Elasticsearch
from collections import Counter

RESULTS_DIR = 'experiments/results'


def load_data():
    files = sorted(glob.glob(f'{RESULTS_DIR}/variations_*.json'))
    if not files:
        return None, None
    with open(files[-1]) as f:
        return json.load(f), files[-1]


def analyze_failures():
    data, path = load_data()
    if not data:
        print("No variations data")
        return
    
    hybrid = next((c for c in data['configs'] if c['name'] == 'hybrid_70_30_vec'), None)
    bm25 = next((c for c in data['configs'] if c['name'] == 'bm25_balanced'), None)
    vec = next((c for c in data['configs'] if c['name'] == 'vector_cosine'), None)
    
    if not all([hybrid, bm25, vec]):
        print("Missing config data")
        return
    
    es = Elasticsearch('http://localhost:9200')
    
    print(f"\n{'='*70}")
    print(f"FAILURE DEEP-DIVE (data: {os.path.basename(path)})")
    print(f"{'='*70}")
    
    # ── 1. Failure Overview ─────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("FAILURE OVERVIEW")
    print(f"{'─'*50}")
    print(f"  {'Config':<25} {'Failures':>10} {'R@5':>10} {'TopSim':>10}")
    for c in [bm25, vec, hybrid]:
        print(f"  {c['name']:<25} {c['failure_count']:>10} {c['recall_at_k']['5']:>9.1%} {c['failure_avg_sim']:>10.4f}")
    
    # ── 2. Per-Strategy Breakdown ──────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("PER-STRATEGY R@5 COMPARISON")
    print(f"{'─'*50}")
    print(f"  {'Strategy':<25} {'BM25':>8} {'Vector':>8} {'Hybrid':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for s in hybrid['per_strategy']:
        b5 = bm25['per_strategy'][s]['recall_at_k']['5']
        v5 = vec['per_strategy'][s]['recall_at_k']['5']
        h5 = hybrid['per_strategy'][s]['recall_at_k']['5']
        print(f"  {s:<25} {b5:>7.2%} {v5:>7.2%} {h5:>7.2%}")
    
    # ── 3. Per-Course Breakdown ────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("PER-COURSE R@5 COMPARISON")
    print(f"{'─'*50}")
    print(f"  {'Course':<30} {'BM25':>8} {'Vector':>8} {'Hybrid':>8} {'Docs':>6}")
    for course in sorted(hybrid['per_course']):
        b5 = bm25['per_course'].get(course, {}).get('recall_at_k', {}).get('5', 0)
        v5 = vec['per_course'].get(course, {}).get('recall_at_k', {}).get('5', 0)
        h5 = hybrid['per_course'][course]['recall_at_k']['5']
        count = es.count(index='faqs_complete', body={'query': {'term': {'course': course}}})['count']
        print(f"  {course:<30} {b5:>7.2%} {v5:>7.2%} {h5:>7.2%} {count:>6}")
    
    # ── 4. Chaos Monkey Deep Dive ──────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("CHAOS MONKEY FAILURE ANALYSIS (high-temp wrong-angle queries)")
    print(f"{'─'*50}")
    
    if hybrid.get('failures_sample'):
        chaos_fails = [f for f in hybrid['failures_sample'] if f['strategy'] == 'chaos_monkey']
        print(f"  {len(chaos_fails)} chaos_monkey failures in sample")
        print(f"\n  Sample failures:")
        for f in chaos_fails[:5]:
            print(f"    Q: {f['query']}")
            print(f"    Top sim: {f.get('top_sim', 'N/A'):.4f}" if f.get('top_sim') else f"    Top sim: N/A")
            if f.get('returned'):
                print(f"    Returned: {f['returned'][0][:80] if f['returned'] else 'N/A'}")
            print()
    
    # ── 5. Hardest Queries ─────────────────────────────────────────────────
    print(f"{'─'*50}")
    print("QUERIES FAILING ACROSS ALL CONFIGS")
    print(f"{'─'*50}")
    if hybrid.get('failures_sample') and bm25.get('failures_sample') and vec.get('failures_sample'):
        h_fails = set(f['query'][:60] for f in hybrid['failures_sample'])
        b_fails = set(f['query'][:60] for f in bm25['failures_sample'])
        v_fails = set(f['query'][:60] for f in vec['failures_sample'])
        common = h_fails & b_fails & v_fails
        print(f"  {len(common)} queries fail in all 3 configs")
        for q in list(common)[:5]:
            print(f"    • {q}")
    
    # ── 6. Key Takeaways ───────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("KEY TAKEAWAYS")
    print(f"{'─'*50}")
    
    hardest_course = min(hybrid['per_course'], key=lambda c: hybrid['per_course'][c]['recall_at_k']['5'])
    easiest_course = max(hybrid['per_course'], key=lambda c: hybrid['per_course'][c]['recall_at_k']['5'])
    hardest_strat = min(hybrid['per_strategy'], key=lambda s: hybrid['per_strategy'][s]['recall_at_k']['5'])
    
    print(f"  1. chaos_monkey is the bottleneck — {hybrid['per_strategy']['chaos_monkey']['recall_at_k']['5']:.0%} R@5")
    print(f"     (vs {hybrid['per_strategy']['grounded_analyst']['recall_at_k']['5']:.0%} for grounded_analyst)")
    print(f"  2. {hardest_course} is hardest course ({hybrid['per_course'][hardest_course]['recall_at_k']['5']:.0%} R@5)")
    print(f"     with {es.count(index='faqs_complete', body={'query': {'term': {'course': hardest_course}}})['count']} docs — largest by far")
    print(f"  3. {easiest_course} is easiest ({hybrid['per_course'][easiest_course]['recall_at_k']['5']:.0%} R@5)")
    print(f"  4. Vector beats BM25 on chaos queries ({vec['per_strategy']['chaos_monkey']['recall_at_k']['5']:.0%} vs {bm25['per_strategy']['chaos_monkey']['recall_at_k']['5']:.0%})")
    print(f"  5. Top similarity for failures is {hybrid['failure_avg_sim']:.4f} — moderate misses, not complete semantic gaps")


if __name__ == '__main__':
    analyze_failures()

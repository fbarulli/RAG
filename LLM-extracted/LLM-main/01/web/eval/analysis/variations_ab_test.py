"""
eval/analysis/variations_ab_test.py
====================================
A/B testing for test_variations.py results.

Run:    uv run python eval/analysis/variations_ab_test.py --a bm25_balanced --b hybrid_70_30_vec
"""
import sys, os, json, glob, argparse
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

RESULTS_DIR = 'experiments/results'


def load_latest():
    files = sorted(glob.glob(f'{RESULTS_DIR}/variations_*.json'))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def run_ab_test(config_a, config_b):
    data = load_latest()
    if not data:
        print("No variations results")
        return
    
    a = next((c for c in data['configs'] if c['name'] == config_a), None)
    b = next((c for c in data['configs'] if c['name'] == config_b), None)
    
    if not a or not b:
        print(f"Available: {[c['name'] for c in data['configs']]}")
        return
    
    print(f"\n{'='*60}")
    print(f"A/B TEST: {config_a} vs {config_b}")
    print(f"{'='*60}")
    print(f"\n{'Metric':<18} {'A':>10} {'B':>10} {'Diff':>10} {'Winner':>8}")
    print(f"{'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    
    # Simple direct comparisons
    pairs = [
        ('Hit Rate', a['hit_rate'], b['hit_rate'], '{:.2%}'),
        ('R@1', a['recall_at_k']['1'], b['recall_at_k']['1'], '{:.2%}'),
        ('R@5', a['recall_at_k']['5'], b['recall_at_k']['5'], '{:.2%}'),
        ('R@10', a['recall_at_k']['10'], b['recall_at_k']['10'], '{:.2%}'),
        ('MRR', a['mrr'], b['mrr'], '{:.4f}'),
        ('NDCG@10', a['ndcg@10'], b['ndcg@10'], '{:.4f}'),
        ('P50 ms', a['p50_latency'], b['p50_latency'], '{:.1f}'),
        ('P95 ms', a['p95_latency'], b['p95_latency'], '{:.1f}'),
        ('Failures', a['failure_count'], b['failure_count'], '{}'),
    ]
    
    for name, av, bv, fmt in pairs:
        diff = bv - av
        if 'ms' in name or 'Fail' in name:
            winner = 'A' if av < bv else 'B' if bv < av else '-'
        else:
            winner = 'B' if bv > av else 'A' if av > bv else '-'
        if 'ms' in name:
            diff_str = f'{diff:+.1f}'
        elif 'Fail' in name:
            diff_str = f'{diff:+.0f}'
        elif 'R@' in name or 'Hit' in name:
            diff_str = f'{diff:+.2%}'
        else:
            diff_str = f'{diff:+.4f}'
        print(f'{name:<18} {fmt.format(av):>10} {fmt.format(bv):>10} {diff_str:>10} {winner:>8}')
    
    # Per-strategy
    if a.get('per_strategy') and b.get('per_strategy'):
        print(f"\n{'='*60}")
        print("PER-STRATEGY R@5")
        print(f"{'Strategy':<25} {'A':>8} {'B':>8} {'Diff':>8}")
        for s in a['per_strategy']:
            if s in b['per_strategy']:
                ar = a['per_strategy'][s]['recall_at_k']['5']
                br = b['per_strategy'][s]['recall_at_k']['5']
                print(f"{s:<25} {ar:>7.2%} {br:>7.2%} {br-ar:>+7.2%}")
    
    # Recommendation
    print(f"\n{'='*60}")
    a_wins = sum(1 for name, av, bv, _ in pairs[:-2] if bv > av)  # higher is better
    b_wins = sum(1 for name, av, bv, _ in pairs[:-2] if av > bv)
    if b_wins > a_wins:
        print(f"RECOMMENDATION: {config_b}")
    else:
        print(f"RECOMMENDATION: {config_a}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', required=True)
    parser.add_argument('--b', required=True)
    args = parser.parse_args()
    run_ab_test(args.a, args.b)

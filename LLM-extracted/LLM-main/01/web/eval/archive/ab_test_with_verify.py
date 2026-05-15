# /home/admin/LLM/LLM/01/web/eval/ab_test_with_verify.py

import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

import json
from eval.visualizer import RAGVisualizer
from eval.eval_set import get_eval_set_from_es

def verify_and_compare(config_a: str, config_b: str, k: int = 5):
    eval_set = get_eval_set_from_es()
    total_questions = len(eval_set)
    print(f"Total questions in Elasticsearch: {total_questions}")
    
    viz = RAGVisualizer()
    registry = viz.get_experiment_registry()
    filenames = registry['filename'].tolist()
    
    print("\n=== VALIDATING RESULT COUNTS ===")
    for filename in filenames:
        with open(f'/home/admin/LLM/LLM/01/web/experiments/results/{filename}', 'r') as f:
            data = json.load(f)
        
        metadata = data['metadata']
        results = data['results']
        unique_queries = len(set(r['query'] for r in results if r['k'] == k))
        
        status = "OK" if unique_queries == total_questions else "MISMATCH"
        print(f"{metadata['name']:25} | Queries: {unique_queries:4} | Expected: {total_questions:4} | {status}")
    
    print(f"\n=== A/B TEST: {config_a} vs {config_b} at K={k} ===")
    
    config_a_file = None
    config_b_file = None
    
    for f in filenames:
        if config_a in f:
            config_a_file = f
        if config_b in f:
            config_b_file = f
    
    if not config_a_file:
        print(f"ERROR: {config_a} not found in results")
        return
    if not config_b_file:
        print(f"ERROR: {config_b} not found in results")
        return
    
    with open(f'/home/admin/LLM/LLM/01/web/experiments/results/{config_a_file}', 'r') as f:
        config_a_data = json.load(f)
    with open(f'/home/admin/LLM/LLM/01/web/experiments/results/{config_b_file}', 'r') as f:
        config_b_data = json.load(f)
    
    config_a_results = {r['query']: r for r in config_a_data['results'] if r['k'] == k}
    config_b_results = {r['query']: r for r in config_b_data['results'] if r['k'] == k}
    
    common_queries = list(set(config_a_results.keys()) & set(config_b_results.keys()))
    print(f"Common queries: {len(common_queries)} / {total_questions}")
    
    a_wins = 0
    b_wins = 0
    ties = 0
    
    for query in common_queries:
        a = config_a_results[query]
        b = config_b_results[query]
        
        if a['success'] and not b['success']:
            a_wins += 1
        elif b['success'] and not a['success']:
            b_wins += 1
        else:
            ties += 1
    
    print(f"\n{config_a} wins: {a_wins}")
    print(f"{config_b} wins: {b_wins}")
    print(f"Ties: {ties}")
    
    if a_wins > b_wins:
        print(f"\nWINNER: {config_a}")
    elif b_wins > a_wins:
        print(f"\nWINNER: {config_b}")
    else:
        print(f"\nTIE")
    
    print("\n=== SAMPLE QUERIES (first 5) ===")
    for i, query in enumerate(list(common_queries)[:5]):
        a = config_a_results[query]
        b = config_b_results[query]
        print(f"\n{i+1}. {query[:60]}...")
        print(f"   {config_a}: {'PASS' if a['success'] else 'FAIL'} (score: {a['score']:.2f})")
        print(f"   {config_b}: {'PASS' if b['success'] else 'FAIL'} (score: {b['score']:.2f})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="Config A name")
    parser.add_argument("--b", required=True, help="Config B name")
    parser.add_argument("--k", type=int, default=5, help="K value")
    args = parser.parse_args()
    
    verify_and_compare(args.a, args.b, args.k)

# /home/admin/LLM/LLM/01/web/eval/ab_test.py

import sys
import os
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

import logging
import traceback
from typing import List, Dict, Any, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from eval.benchmark_runner import BenchmarkRunner


def run_ab_test(config_a: str, config_b: str, k: int = 5, num_queries: Optional[int] = None) -> pd.DataFrame:
    logger.info(f"Running A/B test: {config_a} vs {config_b} at k={k}")
    
    runner_a = BenchmarkRunner(config_a)
    runner_b = BenchmarkRunner(config_b)
    
    results_a = runner_a.run_benchmark(k_values=[k])
    results_b = runner_b.run_benchmark(k_values=[k])
    
    results_a_by_query = {r['query']: r for r in results_a['results']}
    results_b_by_query = {r['query']: r for r in results_b['results']}
    
    common_queries = set(results_a_by_query.keys()) & set(results_b_by_query.keys())
    
    if num_queries:
        common_queries = list(common_queries)[:num_queries]
    
    rows = []
    for query in common_queries:
        ra = results_a_by_query[query]
        rb = results_b_by_query[query]
        
        rows.append({
            'query': query[:80],
            'expected_course': ra.get('found_course', 'NONE'),
            f'{config_a}_success': ra['success'],
            f'{config_a}_score': ra['score'],
            f'{config_a}_found_id': ra['found_id'],
            f'{config_b}_success': rb['success'],
            f'{config_b}_score': rb['score'],
            f'{config_b}_found_id': rb['found_id']
        })
    
    df = pd.DataFrame(rows)
    
    print("\n" + "=" * 80)
    print(f"A/B TEST: {config_a} vs {config_b} at K={k}")
    print("=" * 80)
    
    for _, row in df.iterrows():
        print(f"\nQ: {row['query']}")
        print(f"[A] {config_a}: {'✅' if row[f'{config_a}_success'] else '❌'} (score: {row[f'{config_a}_score']:.2f})")
        print(f"[B] {config_b}: {'✅' if row[f'{config_b}_success'] else '❌'} (score: {row[f'{config_b}_score']:.2f})")
        print("-" * 40)
    
    a_wins = sum(row[f'{config_a}_success'] and not row[f'{config_b}_success'] for _, row in df.iterrows())
    b_wins = sum(row[f'{config_b}_success'] and not row[f'{config_a}_success'] for _, row in df.iterrows())
    both_correct = sum(row[f'{config_a}_success'] and row[f'{config_b}_success'] for _, row in df.iterrows())
    both_wrong = sum(not row[f'{config_a}_success'] and not row[f'{config_b}_success'] for _, row in df.iterrows())
    
    print("\n" + "=" * 40)
    print("WINNER SUMMARY")
    print("=" * 40)
    print(f"Config A ({config_a}) wins: {a_wins}")
    print(f"Config B ({config_b}) wins: {b_wins}")
    print(f"Both correct: {both_correct}")
    print(f"Both wrong: {both_wrong}")
    print(f"Total queries: {len(df)}")
    
    if a_wins > b_wins:
        print(f"\nWINNER: {config_a}")
    elif b_wins > a_wins:
        print(f"\nWINNER: {config_b}")
    else:
        print("\nTIE")
    
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="Config A name")
    parser.add_argument("--b", required=True, help="Config B name")
    parser.add_argument("--k", type=int, default=5, help="K value")
    parser.add_argument("--limit", type=int, help="Limit number of queries")
    
    args = parser.parse_args()
    
    df = run_ab_test(args.a, args.b, k=args.k, num_queries=args.limit)
    print("\n" + "=" * 40)
    print("SAMPLE RESULTS")
    print("=" * 40)
    print(df.head(10).to_string())

#!/usr/bin/env python
"""
eval/benchmarks/run_full_benchmark.py
======================================
Runs all retrieval configs: BM25 variants, vector (cosine/dot), hybrid weights,
and Qdrant (cosine/dot/euclidean).

Output: experiments/results/<config_name>.json

Run:    uv run python eval/benchmarks/run_full_benchmark.py
        uv run python eval/benchmarks/run_full_benchmark.py --config bm25_default
"""
import sys
import os
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

import time
import json
import argparse
from typing import List, Dict, Any

from eval.benchmarks.benchmark_runner import run_benchmark
from eval.benchmarks.qdrant_benchmark_runner import run_qdrant_benchmark

CONFIGS = {
    # BM25 variants
    'bm25_default': 'BM25 Default (q20/t1)',
    'bm25_balanced': 'BM25 Balanced (q5/t5)',
    'bm25_high_question': 'BM25 High Question (q50/t1)',
    'bm25_high_text': 'BM25 High Text (q1/t10)',
    # Vector variants
    'vector_default': 'Vector Cosine',
    'vector_dot': 'Vector Dot Product',
    # Hybrid variants
    'hybrid_default': 'Hybrid Default (50/50)',
    'hybrid_balanced': 'Hybrid Balanced',
    'hybrid_vector_heavy': 'Hybrid Vector Heavy (70/30)',
    'hybrid_bm25_heavy': 'Hybrid BM25 Heavy (30/70)',
    # Qdrant variants
    'qdrant_default': 'Qdrant Cosine',
    'qdrant_dot': 'Qdrant Dot Product',
    'qdrant_euclidean': 'Qdrant Euclidean',
}


def run_all_benchmarks(batch_size: int = 50) -> List[Dict[str, Any]]:
    results = []
    total_start = time.time()
    
    print("=" * 60)
    print("RUNNING FULL BENCHMARK - ALL CONFIGS")
    print("=" * 60)
    print(f"Configs: {len(CONFIGS)}\n")
    
    for config_name, display_name in CONFIGS.items():
        start = time.time()
        print(f"[{config_name}] {display_name}...")
        
        try:
            if 'qdrant' in config_name:
                filename = run_qdrant_benchmark(config_name, batch_size=batch_size)
            else:
                filename = run_benchmark(config_name, batch_size=batch_size)
            
            elapsed = time.time() - start
            
            with open(filename, 'r') as f:
                data = json.load(f)
                total_queries = data['metadata']['total_queries']
                results_data = data['results']
                
                recall_at_5 = sum(1 for r in results_data if r['k'] == 5 and r['success']) / total_queries * 100
                
            results.append({
                'config': config_name,
                'display_name': display_name,
                'time': round(elapsed, 2),
                'recall_at_5': round(recall_at_5, 1),
                'file': filename
            })
            
            print(f"  Done in {elapsed:.2f}s | Recall@5: {recall_at_5:.1f}%\n")
            
        except Exception as e:
            print(f"  Failed: {e}\n")
            results.append({
                'config': config_name,
                'display_name': display_name,
                'error': str(e)
            })
    
    total_elapsed = time.time() - total_start
    
    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total time: {total_elapsed:.2f}s\n")
    
    for r in results:
        if 'recall_at_5' in r:
            print(f"  {r['display_name']:35}: {r['recall_at_5']}% ({r['time']}s)")
        else:
            print(f"  {r['display_name']:35}: FAILED")
    
    return results


def run_single_benchmark(config_name: str, batch_size: int = 50) -> Dict[str, Any]:
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Choose from: {list(CONFIGS.keys())}")
    
    print(f"Running {CONFIGS[config_name]}...")
    start = time.time()
    
    if 'qdrant' in config_name:
        filename = run_qdrant_benchmark(config_name, batch_size=batch_size)
    else:
        filename = run_benchmark(config_name, batch_size=batch_size)
    
    elapsed = time.time() - start
    
    with open(filename, 'r') as f:
        data = json.load(f)
        total_queries = data['metadata']['total_queries']
        results_data = data['results']
        recall_at_5 = sum(1 for r in results_data if r['k'] == 5 and r['success']) / total_queries * 100
    
    return {
        'config': config_name,
        'display_name': CONFIGS[config_name],
        'time': round(elapsed, 2),
        'recall_at_5': round(recall_at_5, 1),
        'file': filename
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Run only a specific config')
    parser.add_argument('--batch-size', type=int, default=50)
    args = parser.parse_args()
    
    if args.config:
        result = run_single_benchmark(args.config, batch_size=args.batch_size)
        print(f"\n{result['display_name']} - Recall@5: {result['recall_at_5']}% in {result['time']}s")
    else:
        run_all_benchmarks(batch_size=args.batch_size)

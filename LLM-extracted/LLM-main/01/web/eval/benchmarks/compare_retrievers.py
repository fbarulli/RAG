"""
compare_retrievers.py
=====================
Runs all retrievers and generates a comparison report.

Retrievers: ES BM25, ES Vector, ES Hybrid, Qdrant Vector

Output: experiments/results/comparison_report.json

Run:    uv run python compare_retrievers.py
"""
import json
import os
from eval.benchmarks.benchmark_runner import BenchmarkRunner
from eval.benchmarks.qdrant_benchmark_runner import QdrantBenchmarkRunner

CONFIGS = [
    'bm25_default',
    'vector_default',
    'hybrid_default',
]


def run_all():
    results = {}

    # ES retrievers
    for config in CONFIGS:
        print(f"\n{'='*50}")
        print(f"Running {config}")
        print(f"{'='*50}")
        try:
            runner = BenchmarkRunner(config)
            output = runner.run_benchmark()
            runner.save_results(output)
            
            k5_results = [r for r in output['results'] if r['k'] == 5]
            hits = sum(1 for r in k5_results if r['success'])
            total = len(k5_results)
            results[config] = {
                'hit_rate_at_5': hits / total if total else 0,
                'total_queries': total,
                'hits': hits,
            }
            print(f"  Hit rate @5: {results[config]['hit_rate_at_5']:.2%}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[config] = {'error': str(e)}

    # Qdrant
    print(f"\n{'='*50}")
    print("Running qdrant_default")
    print(f"{'='*50}")
    try:
        runner = QdrantBenchmarkRunner()
        output = runner.run_benchmark()
        runner.save_results(output)
        
        k5_results = [r for r in output['results'] if r['k'] == 5]
        hits = sum(1 for r in k5_results if r['success'])
        total = len(k5_results)
        results['qdrant_default'] = {
            'hit_rate_at_5': hits / total if total else 0,
            'total_queries': total,
            'hits': hits,
        }
        print(f"  Hit rate @5: {results['qdrant_default']['hit_rate_at_5']:.2%}")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['qdrant_default'] = {'error': str(e)}

    # Save comparison
    os.makedirs('experiments/results', exist_ok=True)
    with open('experiments/results/comparison_report.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print("SUMMARY: Hit Rate @5")
    print(f"{'='*50}")
    for name, data in results.items():
        if 'error' in data:
            print(f"  {name}: ERROR - {data['error']}")
        else:
            print(f"  {name}: {data['hit_rate_at_5']:.2%} ({data['hits']}/{data['total_queries']})")


if __name__ == '__main__':
    run_all()

"""
eval/qdrant_benchmark_runner.py
===============================
Benchmark runner for Qdrant vector search.

Mirrors the ES BenchmarkRunner interface so results are comparable.
Output format matches experiments/results/*.json

Run:    uv run python eval/qdrant_benchmark_runner.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from typing import List, Dict, Any
from datetime import datetime
from src.retrieval.qdrant import QdrantRetriever
from eval.eval_set import get_eval_set_from_es


COLLECTION_MAP = {
    'qdrant_default': 'faqs',
    'qdrant_dot': 'faqs_dot', 
    'qdrant_euclidean': 'faqs_euclidean',
}

def run_qdrant_benchmark(config_name: str = 'qdrant_default', batch_size: int = 50) -> str:
    """Run Qdrant benchmark and return the filename."""
    collection = COLLECTION_MAP.get(config_name, 'faqs')
    retriever = QdrantRetriever(collection_name=collection)
    eval_set = get_eval_set_from_es()
    
    k_values = [1, 3, 5, 10]
    all_results = []
    total_queries = len(eval_set)

    for k in k_values:
        print(f"  Processing k={k}")
        for item in eval_set:
            query = item['original_doc'].get('question', '')
            expected_id = item['expected_id']
            course = item['original_doc'].get('course', '')

            if not query:
                continue

            t0 = time.time()
            hits = retriever.search(query, size=k, course_filter=course)
            latency_ms = (time.time() - t0) * 1000

            hit_ids = [h['id'] for h in hits]
            rank = None
            for pos, hid in enumerate(hit_ids[:k], start=1):
                if hid == expected_id:
                    rank = pos
                    break
            success = rank is not None
            found_id = expected_id if success else (hit_ids[0] if hit_ids else 'NONE')

            all_results.append({
                'k': k,
                'query': query,
                'expected_id': expected_id,
                'found_id': found_id,
                'success': success,
                'rank': rank if rank else -1,
                'score': hits[0]['score'] if hits else 0.0,
                'found_course': hits[0].get('course', 'NONE') if hits else 'NONE',
                'contexts': [h.get('answer', '') for h in hits[:k]],
                'latency_ms': round(latency_ms, 2),
            })

    output = {
        'metadata': {
            f'name': f'Qdrant {config_name}',
            f'config_name': config_name,
            'retriever': 'qdrant',
            'collection': 'faqs',
            'timestamp': datetime.now().isoformat(),
            'total_queries': total_queries,
            'k_values': k_values,
        },
        'results': all_results,
    }

    results_dir = 'experiments/results'
    os.makedirs(results_dir, exist_ok=True)
    filename = f'{results_dir}/{config_name}.json'
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    return filename

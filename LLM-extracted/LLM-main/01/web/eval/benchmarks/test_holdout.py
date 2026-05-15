"""
eval/benchmarks/test_holdout.py
===============================
Evaluates Qdrant retrieval using holdout test queries.
Measures: hit rate, MRR, latency on unseen questions.

Run:    uv run python eval/benchmarks/test_holdout.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import time
import numpy as np

TEST_FILE = 'data_cleaning/data/processed/test.jsonl'
K_VALUES = [1, 3, 5, 10]

# We'll import the retriever directly to avoid module issues
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


def main():
    # Load test queries
    test_docs = []
    with open(TEST_FILE) as f:
        for line in f:
            test_docs.append(json.loads(line))
    
    print(f"Test queries: {len(test_docs)}")
    
    # Setup retriever
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = QdrantClient('localhost', port=6333)
    
    # Run queries
    results = []
    latencies = []
    
    for doc in test_docs:
        query_vec = model.encode(doc['question']).tolist()
        
        t0 = time.time()
        hits = client.query_points(
            collection_name='faqs',
            query=query_vec,
            limit=max(K_VALUES),
            with_payload=True,
        )
        elapsed = (time.time() - t0) * 1000
        latencies.append(elapsed)
        
        # Check rank of expected document
        hit_ids = [h.payload.get('es_id', '') for h in hits.points]
        rank = None
        for pos, hid in enumerate(hit_ids, start=1):
            if hid == doc['id']:
                rank = pos
                break
        
        results.append({
            'query': doc['question'][:80],
            'expected_id': doc['id'],
            'course': doc['course'],
            'rank': rank,
            'found': rank is not None,
            'latency_ms': round(elapsed, 1),
        })
    
    # Stats
    total = len(results)
    found = sum(1 for r in results if r['found'])
    
    print(f"\n{'='*50}")
    print(f"RESULTS: Holdout Test Set")
    print(f"{'='*50}")
    print(f"Total queries: {total}")
    print(f"Found: {found} ({found/total:.1%})")
    print(f"Not found: {total - found}")
    
    for k in K_VALUES:
        hits = sum(1 for r in results if r['found'] and r['rank'] <= k)
        print(f"  Recall@{k}: {hits}/{total} = {hits/total:.2%}")
    
    # MRR
    mrr = sum(1.0 / r['rank'] for r in results if r['found']) / total
    print(f"  MRR: {mrr:.4f}")
    
    # Latency
    print(f"\n  Latency (ms): P50={np.percentile(latencies, 50):.1f}  "
          f"P95={np.percentile(latencies, 95):.1f}  "
          f"P99={np.percentile(latencies, 99):.1f}")
    
    # Show some failures
    failures = [r for r in results if not r['found']]
    if failures:
        print(f"\nSample failures ({min(5, len(failures))} of {len(failures)}):")
        for r in failures[:5]:
            print(f"  [{r['course']}] {r['query']}")


if __name__ == '__main__':
    main()

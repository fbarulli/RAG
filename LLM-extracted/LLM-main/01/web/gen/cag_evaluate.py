"""
gen/cag_evaluate.py
===================
Tests CAG: query → embed → find closest FAQ → return cached answer → check if correct.

Uses the 420 generated test queries to validate the CAG pipeline.

Run:    uv run python gen/cag_evaluate.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, time, numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

CAG_FILE = 'experiments/cag_answers.json'
QUERIES_FILE = 'experiments/eval_queries.json'
MODEL_NAME = 'all-MiniLM-L6-v2'


def load_cag():
    with open(CAG_FILE) as f:
        return json.load(f)['answers']


def load_queries():
    with open(QUERIES_FILE) as f:
        data = json.load(f)
    queries = []
    for doc in data['queries']:
        for strategy, variations in doc['prompt_results'].items():
            for query in variations:
                queries.append({
                    'query': query,
                    'expected_id': doc['expected_id'],
                    'strategy': strategy,
                    'course': doc['course'],
                })
    return queries


def main():
    cag = load_cag()
    test_queries = load_queries()
    
    print(f"CAG answers: {len(cag)}")
    print(f"Test queries: {len(test_queries)}")
    
    # Only evaluate queries whose expected_id has a CAG answer
    evaluable = [q for q in test_queries if q['expected_id'] in cag]
    print(f"Evaluable (have CAG answer): {len(evaluable)}/{len(test_queries)}\n")
    
    if not evaluable:
        print("No queries with CAG answers yet. Generate more CAG answers first.")
        return
    
    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient('localhost', port=6333)
    
    # For each query: embed → search Qdrant → get top FAQ ID → check CAG
    results = []
    latencies = []
    
    for tq in evaluable:
        query_vec = model.encode(tq['query']).tolist()
        
        t0 = time.time()
        hits = client.query_points(
            collection_name='faqs',
            query=query_vec,
            limit=5,
            with_payload=True,
        )
        elapsed = (time.time() - t0) * 1000
        latencies.append(elapsed)
        
        # Get top hit's FAQ ID
        hit_ids = [h.payload.get('es_id', '') for h in hits.points]
        top_id = hit_ids[0] if hit_ids else None
        
        # Check if expected FAQ is in top-k
        rank = next((pos for pos, hid in enumerate(hit_ids, 1) if hid == tq['expected_id']), None)
        
        # Does the top hit have a CAG answer?
        has_cag = top_id in cag if top_id else False
        
        results.append({
            'query': tq['query'][:80],
            'expected_id': tq['expected_id'],
            'top_id': top_id,
            'found': rank is not None,
            'rank': rank,
            'has_cag': has_cag,
            'strategy': tq['strategy'],
            'latency_ms': round(elapsed, 1),
        })
    
    # Stats
    total = len(results)
    found = sum(1 for r in results if r['found'])
    has_cag_count = sum(1 for r in results if r['has_cag'])
    
    print(f"\n{'='*60}")
    print(f"CAG EVALUATION ({len(evaluable)} queries)")
    print(f"{'='*60}")
    print(f"  Retrieval: found {found}/{total} ({found/total:.1%})")
    print(f"  CAG coverage: {has_cag_count}/{total} queries have cached answer for top hit")
    
    # Per-rank
    for k in [1, 3, 5]:
        hits = sum(1 for r in results if r['found'] and r['rank'] and r['rank'] <= k)
        print(f"  Recall@{k}: {hits}/{total} = {hits/total:.1%}")
    
    # Latency
    print(f"\n  Latency: P50={np.percentile(latencies, 50):.1f}ms  P95={np.percentile(latencies, 95):.1f}ms")
    
    # The missing piece: how many queries get the right answer even if retrieval "fails"?
    got_right_answer = sum(1 for r in results if r['found'] and r['has_cag'])
    print(f"\n  Queries where top hit HAS CAG answer: {has_cag_count}")
    print(f"  Queries with correct FAQ AND CAG available: {got_right_answer}")
    print(f"\n  ⚠ CAG only covers {len(cag)} of 1140 FAQs. Need {1140 - len(cag)} more.")
    
    # Per-strategy
    from collections import defaultdict
    by_strat = defaultdict(list)
    for r in results:
        by_strat[r['strategy']].append(r['found'])
    print(f"\n  Per-strategy R@5:")
    for s, founds in sorted(by_strat.items()):
        print(f"    {s:<25}: {sum(founds)/len(founds):.1%}")


if __name__ == '__main__':
    main()

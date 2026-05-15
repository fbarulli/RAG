"""
eval/benchmarks/test_variations.py
===================================
Comprehensive retriever test suite:
- BM25, Vector, Hybrid (RRF + linear), Qdrant
- Cross-encoder reranking
- NDCG, per-strategy, per-course, failure analysis
- Cross-course contamination

Run:    uv run python eval/benchmarks/test_variations.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json, time, math
import numpy as np
from datetime import datetime
from collections import defaultdict
from sentence_transformers import SentenceTransformer, CrossEncoder
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

MODEL_NAME = 'all-MiniLM-L6-v2'
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
ES_INDEX = 'faqs_complete'
QDRANT_COLLECTION = 'faqs'
K_VALUES = [1, 3, 5, 10]
RERANK_K = 20


def load_test_queries(path='experiments/eval_queries.json'):
    if not os.path.exists(path):
        print(f"No {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    queries = []
    for doc in data['queries']:
        for strategy, variations in doc['prompt_results'].items():
            for query in variations:
                queries.append({
                    'query': query, 'expected_id': doc['expected_id'],
                    'original': doc['original_question'], 'course': doc['course'],
                    'strategy': strategy,
                })
    return queries


def compute_ndcg(results, k=10):
    ndcg_scores = []
    for r in results:
        dcg = 0.0
        idcg = 1.0 / math.log2(2)
        for pos in range(1, k+1):
            rel = 1 if (r['found'] and r['rank'] and r['rank'] == pos) else 0
            dcg += rel / math.log2(pos + 1)
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcg_scores))


def evaluate(name, search_fn, test_queries, k=10, use_course_filter=True, model=None, expected_vecs=None):
    results, latencies = [], []
    for tq in test_queries:
        course = tq['course'] if use_course_filter else None
        t0 = time.time()
        hits = search_fn(tq['query'], k, course)
        elapsed = (time.time() - t0) * 1000
        latencies.append(elapsed)
        
        hit_ids = [h.get('id', h.get('es_id', '')) for h in hits]
        hit_courses = [h.get('course', '') for h in hits]
        hit_questions = [h.get('question', '') for h in hits]
        
        rank = next((pos for pos, hid in enumerate(hit_ids, 1) if hid == tq['expected_id']), None)
        wrong_course = sum(1 for c in hit_courses[:5] if c != tq['course'] and c)
        
        top_sim = None
        if model and expected_vecs and tq['expected_id'] in expected_vecs:
            query_vec = model.encode(tq['query'])
            top_vec = model.encode(hit_questions[0]) if hit_questions else None
            if top_vec is not None:
                top_sim = float(np.dot(query_vec, top_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(top_vec)))
        
        results.append({
            'query': tq['query'], 'expected_id': tq['expected_id'],
            'strategy': tq['strategy'], 'course': tq['course'],
            'rank': rank, 'found': rank is not None,
            'top3_ids': hit_ids[:3], 'top3_questions': hit_questions[:3],
            'cross_course': wrong_course, 'top_similarity': top_sim,
            'latency_ms': round(elapsed, 1),
        })
    return results, latencies


def compute_metrics(results, latencies, total):
    found = sum(1 for r in results if r['found'])
    recall_at = {}
    for kv in K_VALUES:
        hits = sum(1 for r in results if r['found'] and r['rank'] and r['rank'] <= kv)
        recall_at[kv] = hits / total if total else 0
    mrr = sum(1.0 / r['rank'] for r in results if r['found'] and r['rank']) / total if total else 0
    ndcg = compute_ndcg(results, 10)
    cross = sum(r['cross_course'] for r in results) / (total * 5) if total else 0
    ranks = [r['rank'] for r in results if r['found'] and r['rank']]
    rank_std = float(np.std(ranks)) if ranks else 0
    fail_sims = [r['top_similarity'] for r in results if not r['found'] and r['top_similarity']]
    
    return {
        'total': total, 'found': found, 'hit_rate': found/total if total else 0,
        'recall_at_k': recall_at, 'mrr': round(mrr, 4), 'ndcg@10': round(ndcg, 4),
        'cross_course_rate': round(cross, 3), 'rank_std': round(rank_std, 2),
        'failure_count': total - found, 'failure_avg_sim': round(float(np.mean(fail_sims)), 4) if fail_sims else 0,
        'p50_latency': round(float(np.percentile(latencies, 50)), 1),
        'p95_latency': round(float(np.percentile(latencies, 95)), 1),
    }


def slice_by(results, latencies, key):
    groups = defaultdict(list)
    groups_lat = defaultdict(list)
    for r, l in zip(results, latencies):
        groups[r[key]].append(r)
        groups_lat[r[key]].append(l)
    return {k: compute_metrics(groups[k], groups_lat[k], len(groups[k])) for k in sorted(groups)}


def failures(results):
    return [
        {'query': r['query'][:80], 'strategy': r['strategy'], 'top_sim': r['top_similarity'],
         'returned': r['top3_questions'][:2]}
        for r in results if not r['found']
    ]


# ── Search functions ──────────────────────────────────────────────────────────
def make_bm25(es, boost_q, boost_t):
    def search(query, size, course=None):
        q = {'multi_match': {'query': query, 'fields': [f'question^{boost_q}', f'answer^{boost_t}'], 'type': 'best_fields'}}
        body = {'size': size, 'query': {'bool': {'must': q, 'filter': {'term': {'course': course}}}} if course else q}
        return [h['_source'] for h in es.search(index=ES_INDEX, **body)['hits']['hits']]
    return search


def make_vector(es, model):
    def search(query, size, course=None):
        vec = model.encode(query).tolist()
        q = {'script_score': {'query': {'match_all': {}}, 'script': {'source': "cosineSimilarity(params.query_vector, 'question_vector') + 1.0", 'params': {'query_vector': vec}}}}
        if course:
            q['script_score']['query'] = {'bool': {'filter': {'term': {'course': course}}}}
        return [h['_source'] for h in es.search(index=ES_INDEX, body={'size': size, 'query': q})['hits']['hits']]
    return search


def make_hybrid(es, model, method, bm25_w=0.5, vec_w=0.5, rrf_k=60):
    bm25_fn = make_bm25(es, 20, 1)
    vec_fn = make_vector(es, model)
    
    def search(query, size, course=None):
        bm25_hits = bm25_fn(query, size * 2, course)
        vec_hits = vec_fn(query, size * 2, course)
        if not bm25_hits and not vec_hits:
            return []
        
        scores = {}
        if method == 'rrf':
            for rank, hit in enumerate(bm25_hits):
                if hid := hit.get('id', ''):
                    scores[hid] = scores.get(hid, 0) + 1.0 / (rrf_k + rank + 1)
            for rank, hit in enumerate(vec_hits):
                if hid := hit.get('id', ''):
                    scores[hid] = scores.get(hid, 0) + 1.0 / (rrf_k + rank + 1)
        else:
            for rank, hit in enumerate(bm25_hits):
                if hid := hit.get('id', ''):
                    scores[hid] = scores.get(hid, 0) + bm25_w * (1.0 / (rank + 1))
            for rank, hit in enumerate(vec_hits):
                if hid := hit.get('id', ''):
                    scores[hid] = scores.get(hid, 0) + vec_w * (1.0 / (rank + 1))
        
        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:size]
        all_hits = {h['id']: h for h in bm25_hits + vec_hits if h.get('id')}
        return [all_hits[id] for id in sorted_ids if id in all_hits]
    return search


def make_rerank(base_search, cross_encoder, rerank_k=RERANK_K):
    def search(query, size, course=None):
        candidates = base_search(query, rerank_k, course)
        if len(candidates) <= size:
            return candidates[:size]
        pairs = [[query, c.get('answer', c.get('text', ''))[:500]] for c in candidates]
        scores = cross_encoder.predict(pairs)
        return [c for c, _ in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:size]]
    return search


def make_qdrant(client, model):
    def search(query, size, course=None):
        vec = model.encode(query).tolist()
        qfilter = Filter(must=[FieldCondition(key='course', match=MatchValue(value=course))]) if course else None
        return [h.payload for h in client.query_points(collection_name=QDRANT_COLLECTION, query=vec, limit=size, query_filter=qfilter, with_payload=True).points]
    return search


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    test_queries = load_test_queries()
    if not test_queries:
        print("No test queries. Run: uv run python eval/generation/generate_test_queries.py")
        return
    
    total = len(test_queries)
    print(f"Test queries: {total}")
    print(f"Strategies: {set(q['strategy'] for q in test_queries)}")
    print(f"Courses: {set(q['course'] for q in test_queries)}\n")
    
    print("Loading models...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  {MODEL_NAME}")
    
    has_reranker = False
    cross_encoder = None
    try:
        cross_encoder = CrossEncoder(RERANK_MODEL)
        has_reranker = True
        print(f"  {RERANK_MODEL}")
    except Exception as e:
        print(f"  No reranker: {e}")
    
    es = Elasticsearch('http://localhost:9200')
    client = QdrantClient('localhost', port=6333)
    
    # Pre-compute expected vectors
    print("  Computing expected vectors...")
    expected_vecs = {}
    for tq in test_queries:
        eid = tq['expected_id']
        if eid not in expected_vecs:
            try:
                expected_vecs[eid] = model.encode(es.get(index=ES_INDEX, id=eid)['_source']['question'])
            except:
                pass
    
    configs = [
        ('bm25_default', make_bm25(es, 20, 1)),
        ('bm25_balanced', make_bm25(es, 5, 5)),
        ('bm25_high_text', make_bm25(es, 1, 10)),
        ('vector_cosine', make_vector(es, model)),
        ('qdrant_cosine', make_qdrant(client, model)),
        ('hybrid_rrf', make_hybrid(es, model, 'rrf')),
        ('hybrid_50_50', make_hybrid(es, model, 'linear', 0.5, 0.5)),
        ('hybrid_70_30_vec', make_hybrid(es, model, 'linear', 0.3, 0.7)),
        ('hybrid_30_70_vec', make_hybrid(es, model, 'linear', 0.7, 0.3)),
    ]
    if has_reranker:
        configs += [
            ('vector_reranked', make_rerank(make_vector(es, model), cross_encoder)),
            ('bm25_reranked', make_rerank(make_bm25(es, 20, 1), cross_encoder)),
        ]
    
    all_metrics = []
    for name, fn in configs:
        print(f"\nTesting {name}...", flush=True)
        results, lats = evaluate(name, fn, test_queries, model=model, expected_vecs=expected_vecs)
        m = compute_metrics(results, lats, total)
        m['name'] = name
        m['per_strategy'] = slice_by(results, lats, 'strategy')
        m['per_course'] = slice_by(results, lats, 'course')
        m['failures_sample'] = failures(results)[:5]
        all_metrics.append(m)
        print(f"  hit={m['hit_rate']:.2%} R@5={m['recall_at_k'][5]:.2%} MRR={m['mrr']} NDCG={m['ndcg@10']}", flush=True)
    
    # ── FULL SUMMARY ──────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"{'Retriever':<22} {'Hit':>7} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'MRR':>7} {'NDCG':>7} {'Fail':>5} {'Sim':>6} {'Cross':>5} {'StdRk':>5} {'P50ms':>7} {'P95ms':>7}")
    print(f"{'='*100}")
    for m in all_metrics:
        print(f"{m['name']:<22} {m['hit_rate']:>6.2%} {m['recall_at_k'][1]:>6.2%} {m['recall_at_k'][5]:>6.2%} "
              f"{m['recall_at_k'][10]:>6.2%} {m['mrr']:>7.4f} {m['ndcg@10']:>7.4f} {m['failure_count']:>5} "
              f"{m['failure_avg_sim']:>6.4f} {m['cross_course_rate']:>4.1%} {m['rank_std']:>5.2f} "
              f"{m['p50_latency']:>6.1f} {m['p95_latency']:>6.1f}")
    
    # ── PER-STRATEGY ──────────────────────────────────────────────────────────
    best = max(all_metrics, key=lambda m: m['recall_at_k'][5])
    print(f"\n{'='*60}")
    print(f"PER-STRATEGY BREAKDOWN ({best['name']})")
    print(f"{'Strategy':<25} {'Hit':>8} {'R@5':>8} {'MRR':>8}")
    for s, sm in best['per_strategy'].items():
        print(f"{s:<25} {sm['hit_rate']:>7.2%} {sm['recall_at_k'][5]:>7.2%} {sm['mrr']:>8.4f}")
    
    # ── PER-COURSE ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PER-COURSE BREAKDOWN ({best['name']})")
    print(f"{'Course':<30} {'Hit':>8} {'R@5':>8}")
    for course, cm in sorted(best['per_course'].items()):
        print(f"{course:<30} {cm['hit_rate']:>7.2%} {cm['recall_at_k'][5]:>7.2%}")
    
    # ── FAILURES ──────────────────────────────────────────────────────────────
    worst = min(all_metrics, key=lambda m: m['recall_at_k'][5])
    print(f"\n{'='*60}")
    print(f"FAILURE ANALYSIS ({worst['name']} - {worst['failure_count']} failures, avg sim: {worst['failure_avg_sim']})")
    for f in worst.get('failures_sample', [])[:5]:
        print(f"  Q: {f['query'][:80]}")
        print(f"     Strategy: {f['strategy']} | Top sim: {f.get('top_sim', 'N/A')}")
    
    # ── CROSS-COURSE ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CROSS-COURSE CONTAMINATION (top-5 results from wrong course)")
    for m in all_metrics[:5]:
        print(f"  {m['name']:<25}: {m['cross_course_rate']:.1%}")
    
    # ── SAVE ──────────────────────────────────────────────────────────────────
    output = {'metadata': {'total_queries': total, 'embedding_model': MODEL_NAME, 'timestamp': datetime.now().isoformat()}, 'configs': all_metrics}
    path = f'experiments/results/variations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    os.makedirs('experiments/results', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()

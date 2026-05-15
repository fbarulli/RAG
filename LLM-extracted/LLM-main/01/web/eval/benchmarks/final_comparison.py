"""
eval/benchmarks/final_comparison.py
====================================
Final retrieval comparison with bge-base 768d.
Open search, no course filter.

Run:    uv run python eval/benchmarks/final_comparison.py
"""
import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json, time, numpy as np
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient

MODEL_NAME = 'BAAI/bge-base-en-v1.5'
ES_INDEX = 'faqs_complete'
QDRANT_COLLECTION = 'faqs_bge_base_en_v1.5'
TOP_K = 5
RRF_K = 60


def load_queries():
    with open('experiments/eval_queries.json') as f:
        data = json.load(f)
    queries = []
    for doc in data['queries']:
        for strategy, variations in doc['prompt_results'].items():
            for query in variations:
                queries.append({
                    'query': query, 'expected_id': doc['expected_id'],
                    'strategy': strategy, 'course': doc['course'],
                })
    return queries


def rrf_score(ranks, k=RRF_K):
    return sum(1.0 / (k + r) for r in ranks)


def multi_rrf(*ranked_id_lists):
    """Fuse any number of ranked ID lists with RRF. Returns top TOP_K IDs per query."""
    combined = []
    for id_sets in zip(*ranked_id_lists):
        candidates = set().union(*id_sets)
        ranks = [{doc_id: r+1 for r, doc_id in enumerate(ids)} for ids in id_sets]
        scores = {}
        for doc_id in candidates:
            scores[doc_id] = rrf_score([r for r, rank_dict in [(ranks[i].get(doc_id), ranks[i]) for i in range(len(ranks))] if doc_id in rank_dict])
        combined.append(sorted(scores, key=scores.__getitem__, reverse=True)[:TOP_K])
    return combined


def compute_r1(ids_list, test_queries):
    return sum(1 for tq, ids in zip(test_queries, ids_list) if ids and ids[0] == tq['expected_id']) / len(test_queries)


def compute_r5(ids_list, test_queries):
    return sum(1 for tq, ids in zip(test_queries, ids_list) if tq['expected_id'] in ids) / len(test_queries)


def print_row(label, r1, r5, p50):
    r1_str = f"{r1:>7.1%}" if r1 is not None else f"{'—':>7}"
    r5_str = f"{r5:>7.1%}" if r5 is not None else f"{'—':>7}"
    p50_str = f"{p50:>7.1f}" if p50 is not None else f"{'—':>7}"
    print(f"  {label:<35} {r1_str} {r5_str} {p50_str}")


es = Elasticsearch('http://localhost:9200')
client = QdrantClient('localhost', port=6333)
test_queries = load_queries()
total = len(test_queries)

collections = {c.name for c in client.get_collections().collections}
assert QDRANT_COLLECTION in collections, f"Collection '{QDRANT_COLLECTION}' not found"

print(f"Testing {total} queries — NO course filter")
print(f"Model: {MODEL_NAME} (768d)\n")

# ── Encode all queries once ──────────────────────────────────────────────────
print("Encoding queries...")
model = SentenceTransformer(MODEL_NAME)
vectors = model.encode([tq['query'] for tq in test_queries], batch_size=64, show_progress_bar=False)

# ── BM25 ─────────────────────────────────────────────────────────────────────
print("Running BM25...")
bm25_ids, bm25_lats = [], []
for tq in test_queries:
    t0 = time.time()
    hits = es.search(index=ES_INDEX, size=TOP_K, query={
        'multi_match': {'query': tq['query'], 'fields': ['question^5', 'answer^5'], 'type': 'best_fields'}
    })['hits']['hits']
    bm25_lats.append((time.time() - t0) * 1000)
    bm25_ids.append([h['_id'] for h in hits])
bm25_r1 = compute_r1(bm25_ids, test_queries)
bm25_r5 = compute_r5(bm25_ids, test_queries)

# ── ES kNN ───────────────────────────────────────────────────────────────────
print("Running ES kNN...")
knn_ids, knn_lats = [], []
try:
    for tq, vec in zip(test_queries, vectors):
        t0 = time.time()
        hits = es.search(index=ES_INDEX, knn={
            'field': 'question_vector', 'query_vector': vec.tolist(),
            'k': TOP_K, 'num_candidates': TOP_K * 10,
        }, size=TOP_K)['hits']['hits']
        knn_lats.append((time.time() - t0) * 1000)
        knn_ids.append([h['_id'] for h in hits])
    knn_r1 = compute_r1(knn_ids, test_queries)
    knn_r5 = compute_r5(knn_ids, test_queries)
except Exception as e:
    print(f"  ES kNN failed: {e}, using script_score fallback")
    knn_ids, knn_lats = [], []
    for tq, vec in zip(test_queries, vectors):
        t0 = time.time()
        hits = es.search(index=ES_INDEX, size=TOP_K, query={
            'script_score': {
                'query': {'match_all': {}},
                'script': {
                    'source': "cosineSimilarity(params.query_vector, 'question_vector') + 1.0",
                    'params': {'query_vector': vec.tolist()}
                }
            }
        })['hits']['hits']
        knn_lats.append((time.time() - t0) * 1000)
        knn_ids.append([h['_id'] for h in hits])
    knn_r1 = compute_r1(knn_ids, test_queries)
    knn_r5 = compute_r5(knn_ids, test_queries)

# ── Qdrant ───────────────────────────────────────────────────────────────────
print("Running Qdrant...")
qdrant_ids, qdrant_lats = [], []
for tq, vec in zip(test_queries, vectors):
    t0 = time.time()
    results = client.query_points(collection_name=QDRANT_COLLECTION, query=vec.tolist(), limit=TOP_K, with_payload=True)
    qdrant_lats.append((time.time() - t0) * 1000)
    qdrant_ids.append([h.payload.get('es_id', '') for h in results.points])
qdrant_r1 = compute_r1(qdrant_ids, test_queries)
qdrant_r5 = compute_r5(qdrant_ids, test_queries)

# ── Hybrid RRF combinations ──────────────────────────────────────────────────
print("Running Hybrid RRF...")

bm25_knn = multi_rrf(bm25_ids, knn_ids)
bm25_knn_r1 = compute_r1(bm25_knn, test_queries)
bm25_knn_r5 = compute_r5(bm25_knn, test_queries)
bm25_knn_lat = [b + k for b, k in zip(bm25_lats, knn_lats)]

bm25_qdrant = multi_rrf(bm25_ids, qdrant_ids)
bm25_qdrant_r1 = compute_r1(bm25_qdrant, test_queries)
bm25_qdrant_r5 = compute_r5(bm25_qdrant, test_queries)
bm25_qdrant_lat = [b + q for b, q in zip(bm25_lats, qdrant_lats)]

knn_qdrant = multi_rrf(knn_ids, qdrant_ids)
knn_qdrant_r1 = compute_r1(knn_qdrant, test_queries)
knn_qdrant_r5 = compute_r5(knn_qdrant, test_queries)
knn_qdrant_lat = [k + q for k, q in zip(knn_lats, qdrant_lats)]

all_three = multi_rrf(bm25_ids, knn_ids, qdrant_ids)
all_three_r1 = compute_r1(all_three, test_queries)
all_three_r5 = compute_r5(all_three, test_queries)
all_three_lat = [b + k + q for b, k, q in zip(bm25_lats, knn_lats, qdrant_lats)]

# ── Results ──────────────────────────────────────────────────────────────────
print(f"\n  {'Config':<35} {'R@1':>8} {'R@5':>8} {'P50ms':>8}")
print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
print_row('BM25 (q^5+a^5)', bm25_r1, bm25_r5, np.percentile(bm25_lats, 50))
print_row('ES kNN (bge-base 768d)', knn_r1, knn_r5, np.percentile(knn_lats, 50))
print_row('Qdrant (bge-base 768d)', qdrant_r1, qdrant_r5, np.percentile(qdrant_lats, 50))
print_row('Hybrid: BM25 + ES kNN', bm25_knn_r1, bm25_knn_r5, np.percentile(bm25_knn_lat, 50))
print_row('Hybrid: BM25 + Qdrant', bm25_qdrant_r1, bm25_qdrant_r5, np.percentile(bm25_qdrant_lat, 50))
print_row('Hybrid: ES kNN + Qdrant', knn_qdrant_r1, knn_qdrant_r5, np.percentile(knn_qdrant_lat, 50))
print_row('Hybrid: BM25 + kNN + Qdrant', all_three_r1, all_three_r5, np.percentile(all_three_lat, 50))

del model
gc.collect()

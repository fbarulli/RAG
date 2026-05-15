"""
eval/benchmarks/test_no_filter.py
==================================
Tests retrieval WITHOUT course filters - cross-course search.
BM25 searches both question AND answer fields.

Run:    uv run python eval/benchmarks/test_no_filter.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json, time, numpy as np
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient

MODEL_NAME        = 'BAAI/bge-small-en-v1.5'
ES_INDEX          = 'faqs_complete'
QDRANT_COLLECTION = 'faqs_bge'
TOP_K             = 5
RRF_K             = 60   # RRF constant — higher = less rank-sensitive


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_queries():
    with open('experiments/eval_queries.json') as f:
        data = json.load(f)
    queries = []
    for doc in data['queries']:
        for strategy, variations in doc['prompt_results'].items():
            for query in variations:
                queries.append({
                    'query': query, 'expected_id': doc['expected_id'],
                    'course': doc['course'], 'strategy': strategy,
                })
    return queries


def rrf_score(ranks: list, k: int = RRF_K) -> float:
    """Reciprocal Rank Fusion score for a document appearing at given ranks."""
    return sum(1.0 / (k + r) for r in ranks)


def print_row(label, found, total, p50):
    print(f"  {label:<36} {found:>6}/{total:<6} {found/total:>7.1%} {p50:>7.1f}ms")


def run_es_benchmark(label, test_queries, build_query_fn, top_k=TOP_K):
    """Run an ES search benchmark. Returns list of hit-lists for reuse in RRF."""
    total = len(test_queries)
    latencies, found, all_hits = [], 0, []
    for tq in test_queries:
        t0   = time.time()
        hits = es.search(index=ES_INDEX, **build_query_fn(tq))['hits']['hits']
        latencies.append((time.time() - t0) * 1000)
        all_hits.append(hits)
        if tq['expected_id'] in [h['_id'] for h in hits]:
            found += 1
    p50 = np.percentile(latencies, 50)
    print_row(label, found, total, p50)
    return all_hits, latencies


def hybrid_rrf(label, test_queries, bm25_hits, qdrant_ids, bm25_latencies, qdrant_latencies):
    """Fuse BM25 and Qdrant results with RRF and report accuracy + latency."""
    total = len(test_queries)
    rrf_latencies, found = [], 0
    for i, tq in enumerate(test_queries):
        t0 = time.time()
        b_ids = [h['_id'] for h in bm25_hits[i]]
        q_ids = qdrant_ids[i]
        candidates  = set(b_ids) | set(q_ids)
        b_rank      = {doc_id: r + 1 for r, doc_id in enumerate(b_ids)}
        q_rank      = {doc_id: r + 1 for r, doc_id in enumerate(q_ids)}
        scores      = {
            doc_id: rrf_score(
                [r for r, lu in [(b_rank.get(doc_id), b_rank), (q_rank.get(doc_id), q_rank)]
                 if doc_id in lu]
            )
            for doc_id in candidates
        }
        top_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:TOP_K]
        rrf_latencies.append((time.time() - t0) * 1000)
        if tq['expected_id'] in top_ids:
            found += 1
    # Total latency = upstream BM25 + Qdrant + RRF fusion (all per-query)
    combined = [b + q + r for b, q, r in zip(bm25_latencies, qdrant_latencies, rrf_latencies)]
    print_row(label, found, total, np.percentile(combined, 50))


# ── Init ─────────────────────────────────────────────────────────────────────

model        = SentenceTransformer(MODEL_NAME)
es           = Elasticsearch('http://localhost:9200')
client       = QdrantClient('localhost', port=6333)
test_queries = load_queries()
total        = len(test_queries)

# Pre-encode all vectors once so repeated runs don't re-encode per benchmark
print("Encoding queries…")
vectors = [model.encode(tq['query']).tolist() for tq in test_queries]
print(f"Done. Testing {total} queries — NO course filter\n")

print(f"{'Config':<38} {'Hit':>14} {'R@5':>8} {'P50ms':>8}")
print(f"{'-'*38} {'-'*14} {'-'*8} {'-'*8}")

# ── BM25 variants ────────────────────────────────────────────────────────────

bm25_q_hits, bm25_q_lat = run_es_benchmark(
    "BM25 (question only)",
    test_queries,
    lambda tq: dict(size=TOP_K, query={'match': {'question': tq['query']}}),
)

run_es_benchmark(
    "BM25 (q^2 + a)",
    test_queries,
    lambda tq: dict(size=TOP_K, query={
        'multi_match': {'query': tq['query'], 'fields': ['question^2', 'answer'],
                        'type': 'best_fields'}
    }),
)

bm25_55_hits, bm25_55_lat = run_es_benchmark(
    "BM25 (q^5 + a^5)",
    test_queries,
    lambda tq: dict(size=TOP_K, query={
        'multi_match': {'query': tq['query'], 'fields': ['question^5', 'answer^5'],
                        'type': 'best_fields'}
    }),
)

run_es_benchmark(
    "BM25 (q + a^10)",
    test_queries,
    lambda tq: dict(size=TOP_K, query={
        'multi_match': {'query': tq['query'], 'fields': ['question', 'answer^10'],
                        'type': 'best_fields'}
    }),
)

# ── cross_fields (treats query holistically across fields) ───────────────────
run_es_benchmark(
    "BM25 cross_fields (q^5 + a^5)",
    test_queries,
    lambda tq: dict(size=TOP_K, query={
        'multi_match': {'query': tq['query'], 'fields': ['question^5', 'answer^5'],
                        'type': 'cross_fields'}
    }),
)

# ── Boost sweep ──────────────────────────────────────────────────────────────
print(f"\n  {'— boost sweep (best_fields) —'}")
best_sweep_hits, best_sweep_lat, best_sweep_score = None, None, -1
for q_b, a_b in [(1,3),(3,1),(3,5),(5,3),(7,3),(3,7),(10,1),(1,10)]:
    hits, lats = run_es_benchmark(
        f"BM25 (q^{q_b} + a^{a_b})",
        test_queries,
        lambda tq, qb=q_b, ab=a_b: dict(size=TOP_K, query={
            'multi_match': {'query': tq['query'],
                            'fields': [f'question^{qb}', f'answer^{ab}'],
                            'type': 'best_fields'}
        }),
    )
    score = sum(
        tq['expected_id'] in [h['_id'] for h in hit_list]
        for tq, hit_list in zip(test_queries, hits)
    )
    if score > best_sweep_score:
        best_sweep_hits, best_sweep_lat, best_sweep_score = hits, lats, score

print()

# ── Native kNN (ES dense_vector with index:true) ─────────────────────────────
# Requires `question_vector` mapped as dense_vector with index:true.
# Falls back gracefully if the mapping isn't ready yet.
knn_hits, knn_latencies = None, None
try:
    knn_latencies, knn_found, knn_hits = [], 0, []
    for tq, vec in zip(test_queries, vectors):
        t0   = time.time()
        hits = es.search(index=ES_INDEX, knn={
            'field': 'question_vector', 'query_vector': vec,
            'k': TOP_K, 'num_candidates': TOP_K * 10,
        }, size=TOP_K)['hits']['hits']
        knn_latencies.append((time.time() - t0) * 1000)
        knn_hits.append(hits)
        if tq['expected_id'] in [h['_id'] for h in hits]:
            knn_found += 1
    print_row("ES kNN (question_vector)", knn_found, total, np.percentile(knn_latencies, 50))
except Exception as e:
    knn_hits, knn_latencies = None, None  # ensure guard works below
    print(f"  ES kNN skipped: {type(e).__name__}: {e}")
    if 'unexpected keyword argument' in str(e):
        print("  => upgrade elasticsearch-py to v8+: pip install 'elasticsearch>=8' --upgrade")

# ── Vector via script_score (always available as fallback) ───────────────────
script_latencies, script_found, script_hits = [], 0, []
for tq, vec in zip(test_queries, vectors):
    t0   = time.time()
    hits = es.search(index=ES_INDEX, size=TOP_K, query={
        'script_score': {
            'query': {'match_all': {}},
            'script': {
                'source': "cosineSimilarity(params.query_vector, 'question_vector') + 1.0",
                'params': {'query_vector': vec},
            }
        }
    })['hits']['hits']
    script_latencies.append((time.time() - t0) * 1000)
    script_hits.append(hits)
    if tq['expected_id'] in [h['_id'] for h in hits]:
        script_found += 1
print_row("ES script_score (cosine)", script_found, total, np.percentile(script_latencies, 50))

# ── Qdrant (no filter) ───────────────────────────────────────────────────────
qdrant_latencies, qdrant_found, qdrant_ids = [], 0, []
for tq, vec in zip(test_queries, vectors):
    t0      = time.time()
    results = client.query_points(
        collection_name=QDRANT_COLLECTION, query=vec, limit=TOP_K, with_payload=True
    )
    qdrant_latencies.append((time.time() - t0) * 1000)
    ids = [h.payload.get('es_id', '') for h in results.points]
    qdrant_ids.append(ids)
    if tq['expected_id'] in ids:
        qdrant_found += 1
print_row("Qdrant (question only)", qdrant_found, total, np.percentile(qdrant_latencies, 50))

# ── Hybrid RRF ───────────────────────────────────────────────────────────────
print(f"\n  {'— hybrid RRF —'}")
hybrid_rrf(
    "RRF: BM25 (q^5+a^5) + Qdrant",
    test_queries, bm25_55_hits, qdrant_ids, bm25_55_lat, qdrant_latencies,
)
hybrid_rrf(
    "RRF: BM25 (q-only) + Qdrant",
    test_queries, bm25_q_hits, qdrant_ids, bm25_q_lat, qdrant_latencies,
)
if best_sweep_hits is not None:
    hybrid_rrf(
        "RRF: best sweep BM25 + Qdrant",
        test_queries, best_sweep_hits, qdrant_ids, best_sweep_lat, qdrant_latencies,
    )
if knn_hits and len(knn_hits) == total:
    knn_ids = [[h['_id'] for h in hits] for hits in knn_hits]
    hybrid_rrf(
        "RRF: BM25 (q^5+a^5) + ES kNN",
        test_queries, bm25_55_hits, knn_ids,
        bm25_55_lat, knn_latencies,
    )

# ── Jupyter dependencies sanity check ────────────────────────────────────────
print(f"\n{'='*52}")
print("JUPYTER DEPENDENCIES CHECK")
print(f"{'='*52}")
for tq in test_queries:
    if tq['expected_id'] == 'ee564fdf82':
        hits = es.search(
            index=ES_INDEX, size=TOP_K,
            query={'multi_match': {
                'query': tq['query'],
                'fields': ['question', 'answer^10'],
                'type': 'best_fields',
            }}
        )['hits']['hits']
        top_id     = hits[0]['_id'] if hits else 'NONE'
        found_flag = tq['expected_id'] in [h['_id'] for h in hits]
        print(f"  Query: {tq['query'][:70]}")
        print(f"  Found: {'✓' if found_flag else '✗'} | Top: {top_id}")
        if not found_flag:
            print(f"  Top question: {hits[0]['_source']['question'][:80] if hits else 'NONE'}")
        print()
        break
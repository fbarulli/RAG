"""
eval/benchmarks/compare_models.py
==================================
Compares embedding models on retrieval with BM25 + hybrid RRF.
Open search, no course filter.

Run:    uv run python eval/benchmarks/compare_models.py
"""
import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json, time, numpy as np
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient

ES_INDEX = 'faqs_complete'
TOP_K = 5
RRF_K = 60

MODELS = [
    ('BAAI/bge-small-en-v1.5', 'faqs_bge_small_en_v1.5'),
    ('intfloat/e5-small-v2',   'faqs_e5_small_v2'),
    ('BAAI/bge-base-en-v1.5',  'faqs_bge_base_en_v1.5'),
    ('intfloat/e5-base-v2',    'faqs_e5_base_v2'),
]

COL_MODEL   = 30
COL_METRIC  = 9   # Qdrant@5, Qdrant@1, Hybrid@5
COL_ENC     = 8
COL_Q       = 7


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


def hybrid_rrf(bm25_hits, qdrant_ids):
    combined = []
    for b_hits, q_ids in zip(bm25_hits, qdrant_ids):
        b_ids = [h['_id'] for h in b_hits]
        candidates = set(b_ids) | set(q_ids)
        b_rank = {doc_id: r + 1 for r, doc_id in enumerate(b_ids)}
        q_rank = {doc_id: r + 1 for r, doc_id in enumerate(q_ids)}
        scores = {
            doc_id: rrf_score(
                [r for r, lu in [(b_rank.get(doc_id), b_rank), (q_rank.get(doc_id), q_rank)]
                 if doc_id in lu]
            )
            for doc_id in candidates
        }
        combined.append(sorted(scores, key=scores.__getitem__, reverse=True)[:TOP_K])
    return combined


def print_header():
    print(f"\n{'Model':<{COL_MODEL}} {'Qdrant@5':>{COL_METRIC}} {'Qdrant@1':>{COL_METRIC}} {'Hybrid@5':>{COL_METRIC}} {'Enc(ms)':>{COL_ENC}} {'Q(ms)':>{COL_Q}}")
    print(f"{'-'*COL_MODEL} {'-'*COL_METRIC} {'-'*COL_METRIC} {'-'*COL_METRIC} {'-'*COL_ENC} {'-'*COL_Q}")


def print_row(label, qdrant5, qdrant1, hybrid5, enc_ms, q_ms):
    q5  = f"{qdrant5:>{COL_METRIC}.1%}" if qdrant5 is not None else f"{'—':>{COL_METRIC}}"
    q1  = f"{qdrant1:>{COL_METRIC}.1%}" if qdrant1 is not None else f"{'—':>{COL_METRIC}}"
    h5  = f"{hybrid5:>{COL_METRIC}.1%}" if hybrid5 is not None else f"{'—':>{COL_METRIC}}"
    enc = f"{enc_ms:>{COL_ENC}.1f}"     if enc_ms  is not None else f"{'—':>{COL_ENC}}"
    q   = f"{q_ms:>{COL_Q}.1f}"         if q_ms    is not None else f"{'—':>{COL_Q}}"
    print(f"{label:<{COL_MODEL}} {q5} {q1} {h5} {enc} {q}")


# ── Init ─────────────────────────────────────────────────────────────────────

es     = Elasticsearch('http://localhost:9200')
client = QdrantClient('localhost', port=6333)
test_queries = load_queries()
total  = len(test_queries)

# Verify all collections exist before starting
collections = {c.name for c in client.get_collections().collections}
for _, coll in MODELS:
    assert coll in collections, f"Collection '{coll}' not found — run ingest_models.py first."

# ── Phase 1: encode all models upfront (keeps progress bars out of the table) ──
print(f"Encoding {total} queries with {len(MODELS)} models...")
all_vectors = {}
for model_name, _ in MODELS:
    short = model_name.split('/')[-1]
    print(f"  {short}...", end='', flush=True)
    model = SentenceTransformer(model_name)
    vecs  = model.encode(
        [tq['query'] for tq in test_queries],
        batch_size=64,
        show_progress_bar=False,
    )
    all_vectors[model_name] = (vecs, time.time())  # store with timestamp to calc enc time
    del model
    gc.collect()
    print(" done")

# ── Phase 2: BM25 baseline ───────────────────────────────────────────────────
print("\nRunning BM25...")
bm25_hits, bm25_lats = [], []
for tq in test_queries:
    t0   = time.time()
    hits = es.search(index=ES_INDEX, size=TOP_K, query={
        'multi_match': {
            'query': tq['query'],
            'fields': ['question^5', 'answer^5'],
            'type': 'best_fields',
        }
    })['hits']['hits']
    bm25_lats.append((time.time() - t0) * 1000)
    bm25_hits.append(hits)
bm25_r5 = sum(
    1 for tq, hits in zip(test_queries, bm25_hits)
    if tq['expected_id'] in [h['_id'] for h in hits]
) / total
bm25_r1 = sum(
    1 for tq, hits in zip(test_queries, bm25_hits)
    if hits and hits[0]['_id'] == tq['expected_id']
) / total

# ── Phase 3: print table ─────────────────────────────────────────────────────
print_header()
print_row("BM25 (q^5+a^5)", None, bm25_r1, bm25_r5, None, np.percentile(bm25_lats, 50))

best_hybrid, best_qdrant = None, None
best_h_score, best_q_score = 0, 0

for model_name, collection in MODELS:
    short   = model_name.split('/')[-1]
    vectors = all_vectors[model_name][0]

    # Measure per-query encoding time by re-encoding one sample
    model    = SentenceTransformer(model_name)
    t0       = time.time()
    model.encode([test_queries[0]['query']], show_progress_bar=False)
    enc_ms   = (time.time() - t0) * 1000  # single-query latency (realistic serving cost)
    del model
    gc.collect()

    # Qdrant search
    q_lats, q_found, q_ids = [], 0, []
    for tq, vec in zip(test_queries, vectors):
        t0      = time.time()
        results = client.query_points(
            collection_name=collection, query=vec.tolist(), limit=TOP_K, with_payload=True
        )
        q_lats.append((time.time() - t0) * 1000)
        ids = [h.payload.get('es_id', '') for h in results.points]
        q_ids.append(ids)
        if tq['expected_id'] in ids:
            q_found += 1
    qdrant_r5 = q_found / total
    qdrant_r1 = sum(
        1 for tq, ids in zip(test_queries, q_ids)
        if ids and ids[0] == tq['expected_id']
    ) / total

    # Hybrid RRF
    hybrid_ids   = hybrid_rrf(bm25_hits, q_ids)
    hybrid_r5    = sum(
        1 for tq, ids in zip(test_queries, hybrid_ids)
        if tq['expected_id'] in ids
    ) / total

    # Combined p50 latency = BM25 + Qdrant (RRF fusion is negligible)
    hybrid_lat = [b + q for b, q in zip(bm25_lats, q_lats)]

    print_row(short, qdrant_r5, qdrant_r1, hybrid_r5, enc_ms, np.percentile(hybrid_lat, 50))

    if hybrid_r5 > best_h_score:
        best_h_score, best_hybrid = hybrid_r5, short
    if qdrant_r5 > best_q_score:
        best_q_score, best_qdrant = qdrant_r5, short

print(f"\nBest Qdrant-only: {best_qdrant} ({best_q_score:.1%})")
print(f"Best Hybrid RRF:  {best_hybrid} ({best_h_score:.1%})")
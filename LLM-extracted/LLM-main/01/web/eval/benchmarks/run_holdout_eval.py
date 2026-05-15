"""
eval/benchmarks/run_holdout_eval.py
====================================
Re-ingests train data into ES and Qdrant, then evaluates all retrievers
against the holdout test set. Uses semantic matching (cosine similarity)
since test questions don't exist verbatim in the train index.

Output: experiments/results/holdout_comparison.json

Run:    uv run python eval/benchmarks/run_holdout_eval.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import time
import numpy as np
from collections import defaultdict
from datetime import datetime
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ── Config ───────────────────────────────────────────────────────────────────
TRAIN_FILE = 'data_cleaning/data/processed/train.jsonl'
TEST_FILE = 'data_cleaning/data/processed/test.jsonl'
MODEL_NAME = 'all-MiniLM-L6-v2'
K_VALUES = [1, 3, 5, 10]
SIMILARITY_THRESHOLD = 0.85  # Min cosine similarity to count as a "hit"

ES_INDEX = 'faqs_holdout'
QDRANT_COLLECTION = 'faqs_holdout'


def load_jsonl(path):
    docs = []
    with open(path) as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def ingest_es(es, docs, model):
    """Ingest train docs into ES with vectors."""
    if es.indices.exists(index=ES_INDEX):
        es.indices.delete(index=ES_INDEX)

    es.indices.create(index=ES_INDEX, mappings={
        "properties": {
            "id": {"type": "keyword"},
            "question": {"type": "text"},
            "answer": {"type": "text"},
            "course": {"type": "keyword"},
            "section": {"type": "keyword"},
            "question_vector": {"type": "dense_vector", "dims": 384, "index": True, "similarity": "cosine"}
        }
    })

    for i, doc in enumerate(docs):
        vec = model.encode(doc['question']).tolist()
        es.index(index=ES_INDEX, id=doc['id'], document={
            'id': doc['id'], 'question': doc['question'],
            'answer': doc['answer'], 'course': doc['course'],
            'section': doc['section'], 'question_vector': vec,
        })
        if (i+1) % 200 == 0:
            print(f'  ES: {i+1}/{len(docs)}')

    es.indices.refresh(index=ES_INDEX)
    return es.count(index=ES_INDEX)['count']


def ingest_qdrant(client, docs, model):
    """Ingest train docs into Qdrant."""
    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    points = []
    for i, doc in enumerate(docs):
        vec = model.encode(doc['question']).tolist()
        points.append(PointStruct(id=i, vector=vec, payload={
            'es_id': doc['id'], 'question': doc['question'],
            'answer': doc['answer'], 'course': doc['course'], 'section': doc['section'],
        }))

    for i in range(0, len(points), 100):
        client.upsert(collection_name=QDRANT_COLLECTION, points=points[i:i+100])

    return client.count(collection_name=QDRANT_COLLECTION).count


def search_es_vector(es, query_vec, size):
    """ES vector search."""
    result = es.search(index=ES_INDEX, body={
        'size': size,
        'query': {
            'script_score': {
                'query': {'match_all': {}},
                'script': {
                    'source': "cosineSimilarity(params.query_vector, 'question_vector') + 1.0",
                    'params': {'query_vector': query_vec}
                }
            }
        }
    })
    return [h['_source'] for h in result['hits']['hits']]


def search_es_bm25(es, query, size):
    """ES BM25 search."""
    result = es.search(index=ES_INDEX, body={
        'size': size,
        'query': {'match': {'question': query}}
    })
    return [h['_source'] for h in result['hits']['hits']]


def search_qdrant(client, query_vec, size):
    """Qdrant vector search."""
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vec,
        limit=size,
        with_payload=True,
    )
    return [h.payload for h in results.points]


def evaluate_retriever(name, search_fn, test_docs, model, is_vector=True):
    """Evaluate a retriever against the holdout test set."""
    results = []
    latencies = []

    for doc in test_docs:
        query = doc['question']
        expected_id = doc['id']

        if is_vector:
            query_vec = model.encode(query).tolist()
            t0 = time.time()
            hits = search_fn(query_vec, max(K_VALUES))
        else:
            t0 = time.time()
            hits = search_fn(query, max(K_VALUES))

        elapsed = (time.time() - t0) * 1000
        latencies.append(elapsed)

        # Check if expected doc is in results (exact ID match)
        hit_ids = [h.get('id', h.get('es_id', '')) for h in hits]
        rank = None
        for pos, hid in enumerate(hit_ids, start=1):
            if hid == expected_id:
                rank = pos
                break

        # Also check semantic similarity of top results to the test question
        test_vec = model.encode(query)
        hit_sims = []
        for hit in hits[:5]:
            hit_question = hit.get('question', '')
            if hit_question:
                hit_vec = model.encode(hit_question)
                sim = np.dot(test_vec, hit_vec) / (np.linalg.norm(test_vec) * np.linalg.norm(hit_vec))
                hit_sims.append(sim)

        found_similar = any(s >= SIMILARITY_THRESHOLD for s in hit_sims) if hit_sims else False

        results.append({
            'query': query[:80],
            'expected_id': expected_id,
            'course': doc['course'],
            'rank': rank,
            'exact_match': rank is not None,
            'similar_match': found_similar,
            'top_similarities': [round(s, 3) for s in hit_sims[:3]],
            'latency_ms': round(elapsed, 1),
        })

    return results, latencies


def print_results(name, results, latencies, total):
    exact = sum(1 for r in results if r['exact_match'])
    similar = sum(1 for r in results if r['similar_match'])

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"  Exact matches: {exact}/{total} ({exact/total:.1%})")
    print(f"  Similar matches (≥{SIMILARITY_THRESHOLD}): {similar}/{total} ({similar/total:.1%})")

    for k in K_VALUES:
        hits = sum(1 for r in results if r['exact_match'] and r['rank'] and r['rank'] <= k)
        print(f"  Recall@{k} (exact): {hits}/{total} = {hits/total:.2%}")

    mrr = sum(1.0 / r['rank'] for r in results if r['exact_match'] and r['rank']) / total
    print(f"  MRR (exact): {mrr:.4f}")

    print(f"  Latency: P50={np.percentile(latencies, 50):.1f}ms  "
          f"P95={np.percentile(latencies, 95):.1f}ms  P99={np.percentile(latencies, 99):.1f}ms")


def main():
    print("Loading model and data...")
    model = SentenceTransformer(MODEL_NAME)
    train_docs = load_jsonl(TRAIN_FILE)
    test_docs = load_jsonl(TEST_FILE)

    print(f"Train: {len(train_docs)} docs")
    print(f"Test:  {len(test_docs)} docs\n")

    # ── Ingest to ES ──────────────────────────────────────────────────────────
    print("Ingesting to Elasticsearch...")
    es = Elasticsearch('http://localhost:9200')
    es_count = ingest_es(es, train_docs, model)
    print(f"  ES {ES_INDEX}: {es_count} docs")

    # ── Ingest to Qdrant ──────────────────────────────────────────────────────
    print("\nIngesting to Qdrant...")
    client = QdrantClient('localhost', port=6333)
    qdrant_count = ingest_qdrant(client, train_docs, model)
    print(f"  Qdrant {QDRANT_COLLECTION}: {qdrant_count} docs")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    total = len(test_docs)

    # ES BM25
    bm25_results, bm25_lat = evaluate_retriever(
        "ES BM25", lambda q, s: search_es_bm25(es, q, s), test_docs, model, is_vector=False
    )
    print_results("ES BM25", bm25_results, bm25_lat, total)

    # ES Vector
    vec_results, vec_lat = evaluate_retriever(
        "ES Vector", lambda q, s: search_es_vector(es, q, s), test_docs, model, is_vector=True
    )
    print_results("ES Vector", vec_results, vec_lat, total)

    # Qdrant
    qdrant_results, qdrant_lat = evaluate_retriever(
        "Qdrant", lambda q, s: search_qdrant(client, q, s), test_docs, model, is_vector=True
    )
    print_results("Qdrant", qdrant_results, qdrant_lat, total)

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        'metadata': {
            'description': 'Holdout evaluation with semantic matching',
            'similarity_threshold': SIMILARITY_THRESHOLD,
            'train_size': len(train_docs),
            'test_size': len(test_docs),
            'timestamp': datetime.now().isoformat(),
        },
        'results': {
            'bm25': {
                'exact_matches': sum(1 for r in bm25_results if r['exact_match']),
                'similar_matches': sum(1 for r in bm25_results if r['similar_match']),
                'p50_latency': round(float(np.percentile(bm25_lat, 50)), 1),
                'p95_latency': round(float(np.percentile(bm25_lat, 95)), 1),
            },
            'vector': {
                'exact_matches': sum(1 for r in vec_results if r['exact_match']),
                'similar_matches': sum(1 for r in vec_results if r['similar_match']),
                'p50_latency': round(float(np.percentile(vec_lat, 50)), 1),
                'p95_latency': round(float(np.percentile(vec_lat, 95)), 1),
            },
            'qdrant': {
                'exact_matches': sum(1 for r in qdrant_results if r['exact_match']),
                'similar_matches': sum(1 for r in qdrant_results if r['similar_match']),
                'p50_latency': round(float(np.percentile(qdrant_lat, 50)), 1),
                'p95_latency': round(float(np.percentile(qdrant_lat, 95)), 1),
            },
        }
    }

    os.makedirs('experiments/results', exist_ok=True)
    path = f'experiments/results/holdout_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {path}")


if __name__ == '__main__':
    main()

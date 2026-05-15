"""
data_cleaning/ingest_es_bge_base.py
====================================
Updates ES faqs_complete with bge-base-en-v1.5 (768d) vectors.
Streams documents to avoid holding everything in RAM.

Run:    uv run python data_cleaning/ingest_es_bge_base.py
        uv run python data_cleaning/ingest_es_bge_base.py --swap  # swap old index
"""
import json, time, argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

MODEL_NAME = 'BAAI/bge-base-en-v1.5'
INPUT = 'data_cleaning/data/processed/clean.jsonl'
ES_HOST = 'http://localhost:9200'
OLD_INDEX = 'faqs_complete'
NEW_INDEX = 'faqs_complete_bge_base'
BATCH_SIZE = 100


def main(swap=False):
    t_total = time.time()
    
    # ── Load model ──────────────────────────────────────────────────────────
    print(f"Loading {MODEL_NAME}...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    dims = model.get_sentence_embedding_dimension()
    print(f"  Dimensions: {dims} ({time.time()-t0:.1f}s)")

    # ── Load documents ──────────────────────────────────────────────────────
    print(f"\nLoading documents...")
    t0 = time.time()
    with open(INPUT) as f:
        docs = [json.loads(line) for line in f]
    questions = [doc['question'] for doc in docs]
    print(f"  {len(docs)} documents ({time.time()-t0:.1f}s)")

    # ── Connect to ES ───────────────────────────────────────────────────────
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        raise RuntimeError(f"Cannot connect to ES at {ES_HOST}")

    # ── Create new index ────────────────────────────────────────────────────
    print(f"\nCreating new index: {NEW_INDEX}")
    if es.indices.exists(index=NEW_INDEX):
        print(f"  Deleting existing {NEW_INDEX}...")
        es.indices.delete(index=NEW_INDEX)

    es.indices.create(index=NEW_INDEX, mappings={
        "properties": {
            "id": {"type": "keyword"},
            "question": {"type": "text"},
            "answer": {"type": "text"},
            "course": {"type": "keyword"},
            "section": {"type": "keyword"},
            "question_vector": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "cosine"
            }
        }
    })
    print(f"  Created with {dims}d dense_vector")

    # ── Encode and stream to ES ─────────────────────────────────────────────
    print(f"\nEncoding and indexing...")
    t_enc = time.time()
    all_vectors = model.encode(questions, batch_size=64, show_progress_bar=True)
    enc_time = time.time() - t_enc
    print(f"  Encoding: {enc_time:.1f}s ({enc_time/len(docs)*1000:.1f}ms/doc)")

    t_idx = time.time()
    total_errors = 0
    
    # Stream in batches - don't build full actions list
    for i in range(0, len(docs), BATCH_SIZE):
        batch_docs = docs[i:i + BATCH_SIZE]
        batch_vecs = all_vectors[i:i + BATCH_SIZE]
        
        actions = []
        for doc, vec in zip(batch_docs, batch_vecs):
            actions.append({
                "_index": NEW_INDEX,
                "_id": doc['id'],
                "_source": {
                    "id": doc['id'],
                    "question": doc['question'],
                    "answer": doc['answer'],
                    "course": doc['course'],
                    "section": doc['section'],
                    "question_vector": vec.tolist(),
                }
            })
        
        success, errors = bulk(es, actions, refresh=False, raise_on_error=False)
        if errors:
            total_errors += len(errors)
            for err in errors[:3]:
                print(f"  Error: {err['index']['error']['reason'][:100]}")
    
    es.indices.refresh(index=NEW_INDEX)
    idx_time = time.time() - t_idx
    print(f"  Indexing: {idx_time:.1f}s ({idx_time/len(docs)*1000:.1f}ms/doc)")

    # ── Verify ──────────────────────────────────────────────────────────────
    new_count = es.count(index=NEW_INDEX)['count']
    old_count = es.count(index=OLD_INDEX)['count']

    print(f"\nVerification:")
    print(f"  Old index ({OLD_INDEX}): {old_count} docs")
    print(f"  New index ({NEW_INDEX}): {new_count} docs")
    if total_errors:
        print(f"  Indexing errors: {total_errors}")
    
    if new_count != old_count:
        print(f"  WARNING: Count mismatch! Missing {old_count - new_count} documents")
    else:
        print(f"  \u2713 Counts match")

    # ── Test search ─────────────────────────────────────────────────────────
    print(f"\nTesting vector search...")
    test_vec = model.encode("how do I install docker").tolist()
    try:
        result = es.search(index=NEW_INDEX, knn={
            'field': 'question_vector',
            'query_vector': test_vec,
            'k': 3,
            'num_candidates': 10,
        }, size=3)
        hits = result['hits']['hits']
        if hits:
            print(f"  \u2713 Search works, top result: {hits[0]['_source']['question'][:80]}")
            print(f"    Score: {hits[0]['_score']:.4f}")
    except Exception as e:
        print(f"  knn search failed: {e}")
        print(f"  Trying script_score fallback...")
        result = es.search(index=NEW_INDEX, size=3, query={
            'script_score': {
                'query': {'match_all': {}},
                'script': {
                    'source': "cosineSimilarity(params.query_vector, 'question_vector') + 1.0",
                    'params': {'query_vector': test_vec}
                }
            }
        })
        hits = result['hits']['hits']
        if hits:
            print(f"  \u2713 Fallback works, top result: {hits[0]['_source']['question'][:80]}")

    # ── Swap if requested ───────────────────────────────────────────────────
    if swap:
        print(f"\nSwapping index: {NEW_INDEX} → {OLD_INDEX}")
        es.indices.delete(index=OLD_INDEX)
        es.indices.put_alias(index=NEW_INDEX, name=OLD_INDEX)
        print(f"  Done. '{OLD_INDEX}' now points to {NEW_INDEX}")
    
    print(f"\nTotal time: {time.time() - t_total:.1f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--swap', action='store_true', help='Delete old index and alias new index')
    args = parser.parse_args()
    main(swap=args.swap)

"""
data_cleaning/ingest_bge.py
============================
Re-ingests documents into ES and Qdrant using BAAI/bge-small-en-v1.5 embeddings.
Compares against current all-MiniLM-L6-v2.

Input:  data_cleaning/data/processed/clean.jsonl
Output: ES index 'faqs_complete' + Qdrant collection 'faqs_bge'

Run:    uv run python data_cleaning/ingest_bge.py
"""
import json
import time
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

MODEL_NAME = 'BAAI/bge-small-en-v1.5'
ES_INDEX = 'faqs_complete'
QDRANT_COLLECTION = 'faqs_bge'
INPUT = 'data_cleaning/data/processed/clean.jsonl'

print(f"Loading {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
dims = model.get_sentence_embedding_dimension()
print(f"  Dimensions: {dims}")

# Load data
with open(INPUT) as f:
    docs = [json.loads(line) for line in f]
print(f"  Documents: {len(docs)}")

# ── Qdrant ────────────────────────────────────────────────────────────────────
print(f"\nIngesting to Qdrant ({QDRANT_COLLECTION})...")
client = QdrantClient('localhost', port=6333)

if client.collection_exists(QDRANT_COLLECTION):
    client.delete_collection(QDRANT_COLLECTION)

client.create_collection(
    collection_name=QDRANT_COLLECTION,
    vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
)

points = []
for i, doc in enumerate(docs):
    vec = model.encode(doc['question']).tolist()
    points.append(PointStruct(
        id=i, vector=vec,
        payload={
            'es_id': doc['id'], 'question': doc['question'],
            'answer': doc['answer'], 'course': doc['course'],
            'section': doc['section'],
        }
    ))
    if (i+1) % 200 == 0:
        print(f"  Encoded {i+1}/{len(docs)}...")

for i in range(0, len(points), 100):
    client.upsert(collection_name=QDRANT_COLLECTION, points=points[i:i+100])

print(f"  Qdrant: {client.count(collection_name=QDRANT_COLLECTION).count} docs")

# ── ES ────────────────────────────────────────────────────────────────────────
print(f"\nUpdating ES vectors ({ES_INDEX})...")
es = Elasticsearch('http://localhost:9200')

for i, doc in enumerate(docs):
    vec = model.encode(doc['question']).tolist()
    es.update(index=ES_INDEX, id=doc['id'], body={
        'doc': {'question_vector': vec}
    })
    if (i+1) % 200 == 0:
        print(f"  Updated {i+1}/{len(docs)}...")

es.indices.refresh(index=ES_INDEX)
print(f"  ES: {es.count(index=ES_INDEX)['count']} docs")

print(f"\nDone! Model: {MODEL_NAME} ({dims}d)")
print(f"  ES index: {ES_INDEX}")
print(f"  Qdrant collection: {QDRANT_COLLECTION}")

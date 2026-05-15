"""
data_cleaning/ingest_qdrant.py
===============================
Ingests train documents into Qdrant vector database.

Input:  data_cleaning/data/processed/train.jsonl
Output: Qdrant collection 'faqs'

Run:    uv run python data_cleaning/ingest_qdrant.py
"""
import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

INPUT = 'data_cleaning/data/processed/train.jsonl'
COLLECTION_NAME = 'faqs'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
QDRANT_HOST = 'localhost'
QDRANT_PORT = 6333
VECTOR_SIZE = 384


def main():
    print(f'Loading embedding model: {EMBEDDING_MODEL}')
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    points = []
    total = 0

    with open(INPUT) as f:
        for line in f:
            doc = json.loads(line)
            embedding = model.encode(doc['question']).tolist()
            points.append(PointStruct(
                id=total,
                vector=embedding,
                payload={
                    'es_id': doc['id'],
                    'question': doc['question'],
                    'answer': doc['answer'],
                    'course': doc['course'],
                    'section': doc['section'],
                }
            ))
            total += 1
            if total % 100 == 0:
                print(f'  Encoded {total} documents...')

    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)

    print(f'Ingested {total} documents into Qdrant')


if __name__ == '__main__':
    main()

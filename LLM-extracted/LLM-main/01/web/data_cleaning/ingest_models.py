"""
data_cleaning/ingest_models.py
===============================
Ingests documents with multiple embedding models into Qdrant.
Creates separate collections for each model.

Models: E5-small (384d), BGE-small (384d), BGE-base (768d), E5-base (768d)

Run:    uv run python data_cleaning/ingest_models.py
"""
import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

INPUT = 'data_cleaning/data/processed/clean.jsonl'
MODELS = [
    'BAAI/bge-small-en-v1.5',    # 384d - current best
    'intfloat/e5-small-v2',       # 384d - top performer
    'BAAI/bge-base-en-v1.5',      # 768d - bigger BGE
    'intfloat/e5-base-v2',        # 768d - bigger E5
]

# Load documents once
with open(INPUT) as f:
    docs = [json.loads(line) for line in f]
questions = [doc['question'] for doc in docs]
print(f"Documents: {len(docs)}")

client = QdrantClient('localhost', port=6333)

for model_name in MODELS:
    short_name = model_name.split('/')[-1].replace('-', '_')
    collection = f'faqs_{short_name}'
    
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"Collection: {collection}")
    
    model = SentenceTransformer(model_name)
    dims = model.get_sentence_embedding_dimension()
    print(f"Dims: {dims}")
    
    # Batch encode all questions
    print("Encoding...")
    vectors = model.encode(
        questions,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    
    # Create collection
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
    )
    
    # Upsert in batches during encoding
    batch = []
    for i, (doc, vec) in enumerate(zip(docs, vectors)):
        batch.append(PointStruct(
            id=i, vector=vec.tolist(),
            payload={
                'es_id': doc['id'], 'question': doc['question'],
                'answer': doc['answer'], 'course': doc['course'],
                'section': doc['section'],
            }
        ))
        if len(batch) == 100:
            client.upsert(collection_name=collection, points=batch)
            batch = []
    if batch:
        client.upsert(collection_name=collection, points=batch)
    
    # Verify
    count = client.count(collection_name=collection).count
    assert count == len(docs), f"Expected {len(docs)}, got {count}"
    print(f"  Done: {count} docs")
    
    # Free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\nAll models ingested!")

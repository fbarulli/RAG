"""
data_cleaning/ingest.py
=======================
Ingests clean documents into Elasticsearch with proper deduplication
and question vectors for vector search.

Deletes existing faqs_complete index and recreates it with the cleaned
data. Uses the document's id field as the ES document ID.
Generates question vectors using all-MiniLM-L6-v2.

Input:  data_cleaning/data/processed/clean.jsonl
Output: Elasticsearch index 'faqs_complete'

Run:    uv run python data_cleaning/ingest.py
"""
import os
import json
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

INPUT = 'data_cleaning/data/processed/clean.jsonl'
ES_HOST = 'http://localhost:9200'
INDEX_NAME = 'faqs_complete'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'


def main():
    # Load embedding model
    print(f'Loading embedding model: {EMBEDDING_MODEL}')
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Connect
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        print(f'ERROR: Cannot connect to Elasticsearch at {ES_HOST}')
        return

    # Delete existing index
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
        print(f'Deleted existing index: {INDEX_NAME}')

    # Create index with mappings (including dense_vector for question_vector)
    mappings = {
        "properties": {
            "id": {"type": "keyword"},
            "question": {"type": "text"},
            "answer": {"type": "text"},
            "course": {"type": "keyword"},
            "section": {"type": "keyword"},
            "question_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
    es.indices.create(index=INDEX_NAME, mappings=mappings)
    print(f'Created index: {INDEX_NAME}')

    # Ingest documents with vectors
    total = 0
    with open(INPUT) as f:
        for line in f:
            doc = json.loads(line)
            
            # Generate question vector
            question_vector = model.encode(doc['question']).tolist()
            
            es.index(
                index=INDEX_NAME,
                id=doc['id'],
                document={
                    'id': doc['id'],
                    'question': doc['question'],
                    'answer': doc['answer'],
                    'course': doc['course'],
                    'section': doc['section'],
                    'question_vector': question_vector,
                }
            )
            total += 1
            
            if total % 100 == 0:
                print(f'  Ingested {total} documents...')

    # Refresh index
    es.indices.refresh(index=INDEX_NAME)
    
    # Verify
    count = es.count(index=INDEX_NAME)['count']
    print(f'Ingested {total} documents')
    print(f'Verified: {count} documents in index')
    
    if total != count:
        print(f'WARNING: Count mismatch! Ingested {total} but index has {count}')


if __name__ == '__main__':
    main()

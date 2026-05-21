"""
p00_ingest_qdrant.py
====================
Ingests FAQ documents into Qdrant using a single embedding model.
Creates one collection per model for isolated evaluation.

Run: uv run python -m rag_pipeline.ingestion.p00_ingest_qdrant --model "BAAI/bge-base-en-v1.5"
"""
import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rag_pipeline.core.paths import Paths
from rag_pipeline.core.schemas import FAQDocument
from rag_pipeline.core.logging import get_logger
logger = get_logger(__name__)
DEFAULT_INPUT = Paths.processed_dir() / 'clean.jsonl'
QDRANT_HOST = 'localhost'
QDRANT_PORT = 6333
BATCH_SIZE = 100
DEFAULT_TOPIC_ASSIGNMENTS = Path('rag_pipeline/p02_eda/experiments/topic_assignments_all.json')

def load_ner_map(model_name: str) -> dict[str, dict]:
    """Load NER tags and topic assignments keyed by document id."""
    path = DEFAULT_TOPIC_ASSIGNMENTS
    if not path.exists():
        logger.warning(f'Topic assignments not found at {path} — skipping NER enrichment')
        return {}
    data = json.load(open(path))
    if 'results' not in data or model_name not in data['results']:
        logger.warning(f'No topic assignments for {model_name} — skipping NER enrichment')
        return {}
    return {a['id']: a for a in data['results'][model_name]['assignments']}

def main(model_name: str, input_path: Path=DEFAULT_INPUT, host: str=QDRANT_HOST, port: int=QDRANT_PORT):
    docs = [FAQDocument.from_dict(json.loads(line)) for line in open(input_path) if line.strip()]
    logger.info(f'Loaded {len(docs)} documents from {input_path}')
    ner_map = load_ner_map(model_name)
    logger.info(f'NER map loaded: {len(ner_map)} entries')
    client = QdrantClient(host=host, port=port)
    questions = [d.question for d in docs]
    short_name = model_name.split('/')[-1].replace('-', '_').replace('.', '_')
    collection = f'faqs_{short_name}'
    logger.info(f'Model: {model_name}')
    logger.info(f'Collection: {collection}')
    logger.info('Loading embedding model...')
    model = SentenceTransformer(model_name)
    dims = model.encode('test').shape[0]
    logger.info(f'Embedding dimensions: {dims}')
    logger.info('Encoding questions...')
    vectors = model.encode(questions, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    if client.collection_exists(collection):
        logger.info(f'Deleting existing collection: {collection}')
        client.delete_collection(collection)
    client.create_collection(collection_name=collection, vectors_config=VectorParams(size=dims, distance=Distance.COSINE))
    logger.info(f'Upserting {len(docs)} points to Qdrant...')
    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc='Batches'):
        batch_docs = docs[i:i + BATCH_SIZE]
        batch_vecs = vectors[i:i + BATCH_SIZE]
        points = [PointStruct(id=j, vector=vec.tolist(), payload={'es_id': doc.id, 'question': doc.question, 'answer': doc.answer, 'course': doc.course, 'section': doc.section, 'ner_category': ner_map.get(doc.id, {}).get('ner_category', 'OTHER'), 'ner_primary_entity': ner_map.get(doc.id, {}).get('ner_primary_entity'), 'topic': ner_map.get(doc.id, {}).get('topic', -1), 'subtopic': ner_map.get(doc.id, {}).get('subtopic')}) for j, (doc, vec) in enumerate(zip(batch_docs, batch_vecs), start=i)]
        client.upsert(collection_name=collection, points=points)
    count = client.count(collection_name=collection).count
    assert count == len(docs), f'Expected {len(docs)}, got {count}'
    logger.info(f'✅ Done: {count} documents indexed in {collection}')
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ingest FAQs into Qdrant')
    parser.add_argument('--model', type=str, required=True, help='Embedding model name (e.g., BAAI/bge-base-en-v1.5)')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--host', type=str, default=QDRANT_HOST)
    parser.add_argument('--port', type=int, default=QDRANT_PORT)
    args = parser.parse_args()
    main(args.model, args.input, args.host, args.port)
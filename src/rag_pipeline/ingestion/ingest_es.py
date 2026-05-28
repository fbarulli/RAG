"""
ingest_es.py
================
Ingests FAQ documents into Elasticsearch with hybrid (text + vector) support.
Creates a single index 'faqs' with mappings for keyword, text, and dense_vector fields.

Run: uv run python -m rag_pipeline.ingestion.ingest_es --model "BAAI/bge-base-en-v1.5"
"""
import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from rag_pipeline.core.paths import Paths
from rag_pipeline.core.schemas import FAQDocument
from rag_pipeline.logging import get_logger
logger = get_logger(__name__)
DEFAULT_INPUT = Paths.processed_dir() / 'clean.jsonl'
ES_HOST = 'http://localhost:9200'
ES_INDEX = 'faqs'
BATCH_SIZE = 100
VECTOR_FIELD = 'question_vector'
VECTOR_DIM = 768

def create_index_mapping(es: Elasticsearch, index: str, dims: int):
    """Create index with text + dense_vector mappings for hybrid search."""
    mapping = {'mappings': {'properties': {'es_id': {'type': 'keyword'}, 'question': {'type': 'text', 'analyzer': 'standard', 'fields': {'keyword': {'type': 'keyword', 'ignore_above': 256}}}, 'answer': {'type': 'text', 'analyzer': 'standard'}, 'course': {'type': 'keyword'}, 'section': {'type': 'keyword'}, VECTOR_FIELD: {'type': 'dense_vector', 'dims': dims, 'index': True, 'similarity': 'cosine'}}}}
    if es.indices.exists(index=index):
        logger.info(f'Deleting existing index: {index}')
        es.indices.delete(index=index)
    es.indices.create(index=index, body=mapping)
    logger.info(f"Created index '{index}' with dense_vector field ({dims}d)")

def main(model_name: str, input_path: Path=DEFAULT_INPUT, host: str=ES_HOST, index: str=ES_INDEX):
    docs = [FAQDocument.from_dict(json.loads(line)) for line in open(input_path) if line.strip()]
    logger.info(f'Loaded {len(docs)} documents from {input_path}')
    es = Elasticsearch(hosts=[host], request_timeout=30)
    if not es.ping():
        logger.error(f'Cannot connect to Elasticsearch at {host}')
        return
    logger.info(f'Loading embedding model: {model_name}')
    model = SentenceTransformer(model_name)
    dims = model.get_sentence_embedding_dimension()
    logger.info(f'Embedding dimensions: {dims}')
    create_index_mapping(es, index, dims)
    logger.info('Encoding questions...')
    questions = [d.question for d in docs]
    vectors = model.encode(questions, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    logger.info(f'Indexing {len(docs)} documents to Elasticsearch...')
    actions = []
    for i, (doc, vec) in enumerate(zip(docs, vectors)):
        action = {'_index': index, '_id': doc.id, '_source': {'es_id': doc.id, 'question': doc.question, 'answer': doc.answer, 'course': doc.course, 'section': doc.section, VECTOR_FIELD: vec.tolist()}}
        actions.append(action)
        if len(actions) >= BATCH_SIZE:
            helpers.bulk(es, actions, request_timeout=60)
            logger.debug(f'Indexed {len(actions)} docs')
            actions = []
    if actions:
        helpers.bulk(es, actions, request_timeout=60)
    es.indices.refresh(index=index)
    count = es.count(index=index)['count']
    assert count == len(docs), f'Expected {len(docs)}, got {count}'
    logger.info(f"✅ Done: {count} documents indexed in '{index}'")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ingest FAQs into Elasticsearch')
    parser.add_argument('--model', type=str, required=True, help='Embedding model name')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--host', type=str, default=ES_HOST)
    parser.add_argument('--index', type=str, default=ES_INDEX)
    args = parser.parse_args()
    main(args.model, args.input, args.host, args.index)
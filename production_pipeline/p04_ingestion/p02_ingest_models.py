"""
p02_ingest_models.py
====================
Ingests FAQ documents into Qdrant using multiple embedding models.
Creates separate collections per model for comparative evaluation.

Run:    just run ingest-models
"""
import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from rag_pipeline.paths import Paths
from rag_pipeline.schemas import FAQDocument
from rag_pipeline.logging import get_logger

logger = get_logger(__name__)

DEFAULT_INPUT = Paths.processed_dir() / "clean.jsonl"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
BATCH_SIZE = 100

MODELS = [
    'BAAI/bge-small-en-v1.5',    # 384d - current baseline
    'intfloat/e5-small-v2',       # 384d - strong alternative
    'BAAI/bge-base-en-v1.5',      # 768d - larger BGE
    'intfloat/e5-base-v2',        # 768d - larger E5
]

def main(input_path: Path = DEFAULT_INPUT, host: str = QDRANT_HOST, port: int = QDRANT_PORT):
    docs = [FAQDocument.from_dict(json.loads(line)) for line in open(input_path) if line.strip()]
    logger.info(f"Loaded {len(docs)} documents from {input_path}")

    client = QdrantClient(host=host, port=port)
    questions = [d.question for d in docs]

    for model_name in MODELS:
        short_name = model_name.split('/')[-1].replace('-', '_')
        collection = f'faqs_{short_name}'
        logger.info(f"\n{'='*50}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Collection: {collection}")

        model = SentenceTransformer(model_name)
        dims = model.get_sentence_embedding_dimension()
        logger.info(f"Dimensions: {dims}")

        logger.info("Encoding questions...")
        vectors = model.encode(
            questions,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        if client.collection_exists(collection):
            logger.info(f"Deleting existing collection: {collection}")
            client.delete_collection(collection)
            
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
        )

        logger.info("Upserting to Qdrant...")
        batch = []
        for i, (doc, vec) in enumerate(zip(docs, vectors)):
            batch.append(PointStruct(
                id=i,
                vector=vec.tolist(),
                payload={
                    'es_id': doc.id,
                    'question': doc.question,
                    'answer': doc.answer,
                    'course': doc.course,
                    'section': doc.section,
                }
            ))
            if len(batch) == BATCH_SIZE:
                client.upsert(collection_name=collection, points=batch)
                batch = []
        if batch:
            client.upsert(collection_name=collection, points=batch)

        count = client.count(collection_name=collection).count
        assert count == len(docs), f"Expected {len(docs)}, got {count}"
        logger.info(f"✅ Done: {count} documents indexed")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("\n🎉 All models ingested successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--host', type=str, default=QDRANT_HOST)
    parser.add_argument('--port', type=int, default=QDRANT_PORT)
    args = parser.parse_args()
    main(args.input, args.host, args.port)

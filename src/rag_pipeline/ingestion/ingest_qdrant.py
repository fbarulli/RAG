# rag_pipeline/ingestion/ingest_qdrant.py
"""
Ingests FAQ documents into Qdrant using a single embedding model.
Creates one collection per model for isolated evaluation.

Run: uv run python -m rag_pipeline.ingestion.ingest_qdrant --model "BAAI/bge-base-en-v1.5"
"""
import argparse
import json
import torch
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, HnswConfigDiff, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src.rag_pipeline.core.paths import Paths
from rag_pipeline.core.schemas import FAQDocument
from rag_pipeline.ingestion.embedding_cache import load_cached_embeddings, save_embeddings_cache
from src.rag_pipeline.logging import get_logger

logger = get_logger(__name__)

_CACHE_PREFIX = "corpus_full_"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_corpus(input_path: Path) -> list[FAQDocument]:
    docs = [FAQDocument.from_dict(json.loads(line)) for line in open(input_path, encoding="utf-8") if line.strip()]
    logger.info("Loaded %d documents from %s", len(docs), input_path)
    return docs


def _load_ner_map(model_name: str) -> dict[str, dict]:
    path = Paths.topic_assignments()
    data = json.load(open(path, encoding="utf-8"))
    if "results" not in data or model_name not in data["results"]:
        logger.warning("No topic assignments for %s — skipping NER enrichment", model_name)
        return {}
    mapping = {a["id"]: a for a in data["results"][model_name]["assignments"]}
    logger.info("NER map loaded: %d entries", len(mapping))
    return mapping


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def _cache_key(model_name: str) -> str:
    return _CACHE_PREFIX + model_name.split("/")[-1].replace("-", "_").replace(".", "_")


def _encode(docs: list[FAQDocument], model_name: str, encode_batch_size: int) -> np.ndarray:
    logger.info("Loading embedding model %s...", model_name)
    model = SentenceTransformer(model_name)
    texts = [f"{d.question} {d.answer}" for d in docs]
    vectors = model.encode(texts, batch_size=encode_batch_size, show_progress_bar=True, convert_to_numpy=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vectors


def _get_embeddings(docs: list[FAQDocument], model_name: str, cache_dir: Path, encode_batch_size: int) -> np.ndarray:
    short = _cache_key(model_name)
    cached = load_cached_embeddings(cache_dir, short, len(docs))
    if cached is not None:
        return cached
    vectors = _encode(docs, model_name, encode_batch_size)
    save_embeddings_cache(cache_dir, short, vectors)
    return vectors


# ---------------------------------------------------------------------------
# Point construction
# ---------------------------------------------------------------------------

def _build_payload(doc: FAQDocument, ner_map: dict) -> dict:
    ner = ner_map.get(doc.id, {})
    return {
        "es_id":              doc.id,
        "question":           doc.question,
        "answer":             doc.answer,
        "course":             doc.course,
        "section":            doc.section,
        "ner_category":       ner.get("ner_category", "OTHER"),
        "ner_primary_entity": ner.get("ner_primary_entity"),
        "topic":              ner.get("topic", -1),
        "subtopic":           ner.get("subtopic"),
    }


def _build_points(docs: list[FAQDocument], vectors: np.ndarray, ner_map: dict) -> list[PointStruct]:
    return [
        PointStruct(id=i, vector=vec.tolist(), payload=_build_payload(doc, ner_map))
        for i, (doc, vec) in enumerate(zip(docs, vectors))
    ]


# ---------------------------------------------------------------------------
# Qdrant operations
# ---------------------------------------------------------------------------

def _ensure_collection(client: QdrantClient, collection: str, dims: int) -> None:
    if client.collection_exists(collection):
        logger.info("Deleting existing collection: %s", collection)
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=200, full_scan_threshold=10000),
    )


def _upload_points(client: QdrantClient, collection: str, points: list[PointStruct]) -> None:
    logger.info("Uploading %d points to %s...", len(points), collection)
    client.upload_points(collection_name=collection, points=points)


def _verify_count(client: QdrantClient, collection: str, expected: int) -> None:
    count = client.count(collection_name=collection).count
    assert count == expected, f"Expected {expected}, got {count} in '{collection}'"
    logger.info("Done: %d documents indexed in %s", count, collection)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(model_name: str, input_path: Path = None, host: str = None, port: int = None) -> None:
    d                 = Paths.defaults()
    input_path        = input_path or Paths.clean_jsonl()
    host              = host       or d["qdrant"]["host"]
    port              = port       or d["qdrant"]["port"]
    encode_batch_size = d["ingestion"]["encode_batch_size"]
    cache_dir         = Paths.embeddings_cache_dir()

    docs       = _load_corpus(input_path)
    ner_map    = _load_ner_map(model_name)
    vectors    = _get_embeddings(docs, model_name, cache_dir, encode_batch_size)
    collection = Paths.collection_for_model(model_name)
    client     = QdrantClient(host=host, port=port)
    points     = _build_points(docs, vectors, ner_map)

    _ensure_collection(client, collection, vectors.shape[1])
    _upload_points(client, collection, points)
    _verify_count(client, collection, len(docs))


if __name__ == "__main__":
    from configs.benchmark_cli import create_ingestion_parser
    args = create_ingestion_parser().parse_args()
    main(
        model_name=args.model,
        input_path=args.clean_path,
        host=args.qdrant_host,
        port=args.qdrant_port,
    )
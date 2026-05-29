# rag_pipeline/ingestion/ingest_es.py
"""
Ingests FAQ documents into Elasticsearch with hybrid (text + vector) support.
Creates a single index 'faqs' with mappings for keyword, text, and dense_vector fields.

Run: uv run python -m rag_pipeline.ingestion.ingest_es --model "BAAI/bge-base-en-v1.5"
"""
import json
import torch
from pathlib import Path

import numpy as np
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

from rag_pipeline.core.paths import Paths
from rag_pipeline.core.schemas import FAQDocument
from rag_pipeline.ingestion.embedding_cache import load_cached_embeddings, save_embeddings_cache
from rag_pipeline.logging import get_logger

logger = get_logger(__name__)

VECTOR_FIELD = "question_vector"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_corpus(input_path: Path) -> list[FAQDocument]:
    docs = [FAQDocument.from_dict(json.loads(line)) for line in open(input_path, encoding="utf-8") if line.strip()]
    logger.info("Loaded %d documents from %s", len(docs), input_path)
    return docs


# ---------------------------------------------------------------------------
# Embeddings — encodes questions only; shares cache with ingest_models.py
# ---------------------------------------------------------------------------

def _cache_key(model_name: str) -> str:
    return model_name.split("/")[-1].replace("-", "_").replace(".", "_")


def _encode(docs: list[FAQDocument], model_name: str, encode_batch_size: int) -> np.ndarray:
    logger.info("Loading embedding model %s...", model_name)
    model = SentenceTransformer(model_name)
    vectors = model.encode(
        [d.question for d in docs],
        batch_size=encode_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
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
# Index operations
# ---------------------------------------------------------------------------

def _create_index(es: Elasticsearch, index: str, dims: int) -> None:
    mapping = {
        "mappings": {
            "properties": {
                "es_id":      {"type": "keyword"},
                "question":   {"type": "text", "analyzer": "standard",
                               "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
                "answer":     {"type": "text", "analyzer": "standard"},
                "course":     {"type": "keyword"},
                "section":    {"type": "keyword"},
                VECTOR_FIELD: {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine"},
            }
        }
    }
    if es.indices.exists(index=index):
        logger.info("Deleting existing index: %s", index)
        es.indices.delete(index=index)
    es.indices.create(index=index, body=mapping)
    logger.info("Created index '%s' with dense_vector field (%dd)", index, dims)


# ---------------------------------------------------------------------------
# Document actions
# ---------------------------------------------------------------------------

def _build_actions(docs: list[FAQDocument], vectors: np.ndarray, index: str) -> list[dict]:
    return [
        {
            "_index":  index,
            "_id":     doc.id,
            "_source": {
                "es_id":      doc.id,
                "question":   doc.question,
                "answer":     doc.answer,
                "course":     doc.course,
                "section":    doc.section,
                VECTOR_FIELD: vec.tolist(),
            },
        }
        for doc, vec in zip(docs, vectors)
    ]


def _bulk_upload(es: Elasticsearch, actions: list[dict], bulk_timeout: int) -> None:
    logger.info("Indexing %d documents to Elasticsearch...", len(actions))
    helpers.bulk(es, actions, request_timeout=bulk_timeout)


def _verify_count(es: Elasticsearch, index: str, expected: int) -> None:
    es.indices.refresh(index=index)
    count = es.count(index=index)["count"]
    assert count == expected, f"Expected {expected}, got {count} in '{index}'"
    logger.info("Done: %d documents indexed in '%s'", count, index)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(model_name: str, input_path: Path, host: str, index: str) -> None:
    d                 = Paths.defaults()
    encode_batch_size = d["ingestion"]["encode_batch_size"]
    bulk_timeout      = d["elasticsearch"]["bulk_timeout"]
    request_timeout   = d["elasticsearch"]["request_timeout"]
    cache_dir         = Paths.embeddings_cache_dir()

    es = Elasticsearch(hosts=[host], request_timeout=request_timeout)
    if not es.ping():
        raise RuntimeError("Cannot connect to Elasticsearch at %s" % host)

    docs    = _load_corpus(input_path)
    vectors = _get_embeddings(docs, model_name, cache_dir, encode_batch_size)
    actions = _build_actions(docs, vectors, index)

    _create_index(es, index, vectors.shape[1])
    _bulk_upload(es, actions, bulk_timeout)
    _verify_count(es, index, len(docs))


if __name__ == "__main__":
    from configs.benchmark_cli import create_ingestion_parser
    _args = create_ingestion_parser().parse_args()
    _d    = Paths.defaults()
    main(
        model_name=_args.model,
        input_path=_args.clean_path or Paths.clean_jsonl(),
        host=_args.es_host   or _d["elasticsearch"]["host"],
        index=_args.es_index or _d["elasticsearch"]["index"],
    )
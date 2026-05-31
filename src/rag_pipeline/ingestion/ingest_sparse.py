"""
Ingest SPLADE sparse embeddings into Qdrant sparse vector collections.

Each sparse model gets its own collection with a single named sparse vector "sparse".
Payloads are identical to dense collections (copied from the bge-base collection).

Usage:
    uv run python -m rag_pipeline.ingestion.ingest_sparse
    uv run python -m rag_pipeline.ingestion.ingest_sparse --model prithivida__Splade_PP_en_v1
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, SparseVector, SparseVectorParams, SparseIndexParams,
)
from rag_pipeline.core.paths import Paths
from rag_pipeline.logging import get_logger
from rag_pipeline.ingestion.ingest_models import load_corpus, _load_ner_map, _doc_id_to_uuid, _prepare_payload
from rag_pipeline.core.models import DocNERInfo, EncodeMode

logger = get_logger(__name__)

SPARSE_MODELS = {
    "prithivida__Splade_PP_en_v1":              "sparse_prithivida__Splade_PP_en_v1.npy",
    "prithivida__Splade_PP_en_v2":              "sparse_prithivida__Splade_PP_en_v2.npy",
    "naver__splade-cocondenser-ensembledistil": "sparse_naver__splade-cocondenser-ensembledistil.npy",
    "naver__splade_v2_max":                     "sparse_naver__splade_v2_max.npy",
}

SPARSE_VECTOR_NAME = "sparse"


def collection_name(model_key: str) -> str:
    return "faqs_" + model_key.replace("-", "_").replace("/", "_")


def load_sparse_npy(path: Path) -> list[tuple[list[int], list[float]]]:
    """Load a dense SPLADE matrix (n_docs x vocab) and convert each row to sparse (indices, values)."""
    logger.info(f"Loading sparse embeddings from {path} ...")
    mat = np.load(path, allow_pickle=False)
    logger.info(f"  shape={mat.shape}  nnz/row~={int((mat != 0).sum() / mat.shape[0])}")
    out = []
    for row in mat:
        nz = np.nonzero(row)[0]
        out.append((nz.tolist(), row[nz].tolist()))
    return out


def ingest_sparse_model(model_key: str, npy_filename: str, docs, ner_map, client: QdrantClient) -> bool:
    gpu_dir = Paths.experiments_dir() / "_embeddings_gpu"
    npy_path = gpu_dir / npy_filename
    if not npy_path.exists():
        logger.error(f"NPY not found: {npy_path}")
        return False

    sparse_vecs = load_sparse_npy(npy_path)
    if len(sparse_vecs) != len(docs):
        logger.error(f"Shape mismatch: {len(sparse_vecs)} vectors vs {len(docs)} docs")
        return False

    coll = collection_name(model_key)
    logger.info(f"Creating collection: {coll}")
    if client.collection_exists(coll):
        client.delete_collection(coll)
    client.create_collection(
        collection_name=coll,
        vectors_config={},  # no dense vectors
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )

    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        batch_vecs = sparse_vecs[i:i + batch_size]
        points = []
        for doc, (indices, values) in zip(batch_docs, batch_vecs):
            ner_info = ner_map.get(doc.id)
            if ner_info is None:
                logger.warning(f"Missing NER for {doc.id}, using defaults")
                ner_info = DocNERInfo()
            points.append(PointStruct(
                id=_doc_id_to_uuid(doc.id),
                vector={SPARSE_VECTOR_NAME: SparseVector(indices=indices, values=values)},
                payload=_prepare_payload(doc, ner_info),
            ))
        client.upsert(collection_name=coll, points=points)
        logger.info(f"  upserted {min(i+batch_size, len(docs))}/{len(docs)}")

    count = client.count(collection_name=coll).count
    if count != len(docs):
        logger.error(f"Count mismatch: {count} vs {len(docs)}")
        return False
    logger.info(f"✅ {coll}: {count} docs ingested")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Single model key to ingest (default: all)")
    args = parser.parse_args()

    docs = load_corpus(Paths.clean_jsonl())
    production_model = Paths.defaults()["production_model"]
    ner_map = _load_ner_map(Paths.topic_assignments(), production_model)

    from rag_pipeline.ingestion.benchmark_config import BenchmarkConfig
    cfg = BenchmarkConfig.from_defaults()
    client = cfg.make_qdrant_client()

    models = {args.model: SPARSE_MODELS[args.model]} if args.model else SPARSE_MODELS
    failed = []
    for key, filename in models.items():
        ok = ingest_sparse_model(key, filename, docs, ner_map, client)
        if not ok:
            failed.append(key)

    if failed:
        logger.error(f"Failed: {failed}")
    else:
        logger.info("All sparse models ingested successfully.")


if __name__ == "__main__":
    main()

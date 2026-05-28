"""
def doc_id_to_point_id(doc_id: str) -> str:
    Convert document ID string to a Qdrant point ID.
    I/O: doc_id (str) -> str

def get_cache_path(cache_dir: Path, model_short_name: str) -> Path:
    Return path for cached embeddings .npy file.
    I/O: cache_dir (Path), model_short_name (str) -> Path

def load_cached_embeddings(cache_dir: Path, model_short_name: str, expected_count: int) -> Optional[np.ndarray]:
    Return cached embeddings array or None if cache is absent/stale.
    I/O: cache_dir (Path), model_short_name (str), expected_count (int) -> Optional[np.ndarray]

def save_embeddings_cache(cache_dir: Path, model_short_name: str, vectors: np.ndarray) -> None:
    Save embeddings to cache using atomic write.
    I/O: cache_dir (Path), model_short_name (str), vectors (np.ndarray) -> None

def load_corpus(input_path: Path) -> list[FAQDocument]:
    Load corpus with robust error handling.
    I/O: input_path (Path) -> list[FAQDocument]

def load_ner_map(topic_assignments_path: Optional[Path], model_name: str) -> dict[str, dict]:
    Load topic/NER assignments for a specific model.
    I/O: topic_assignments_path (Optional[Path]), model_name (str) -> dict[str, dict]


    
ingest_models.py
====================
Ingests FAQ documents into Qdrant using all embedding models declared in
``configs/models.json``.

Creates separate collections per model for comparative evaluation.

Run:
    uv run python -m rag_pipeline.ingestion.ingest_models
    uv run python -m rag_pipeline.ingestion.ingest_models --model BAAI/bge-base-en-v1.5
"""
from __future__ import annotations
import gc
import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from rag_pipeline.logging import get_logger
from rag_pipeline.core.schemas import FAQDocument
from .benchmark_config import BenchmarkConfig
from configs.benchmark_cli import create_ingestion_parser
logger = get_logger(__name__)

def doc_id_to_point_id(doc_id: str) -> str:
    """Convert document ID string to Qdrant point ID."""
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(doc_id)))

def get_cache_path(cache_dir: Path, model_short_name: str) -> Path:
    """Return path for cached embeddings .npy file."""
    return cache_dir / f'{model_short_name}.npy'

def load_cached_embeddings(cache_dir: Path, model_short_name: str, expected_count: int) -> Optional[np.ndarray]:
    """Return cached embeddings array or None if cache is absent/stale."""
    path = get_cache_path(cache_dir, model_short_name)
    if not path.exists():
        return None
    try:
        arr = np.load(path, allow_pickle=False)
    except ValueError as e:
        logger.warning(f'Failed to load cache {path}: {e}')
        return None
    if arr.shape[0] != expected_count:
        logger.warning(f"Cache '{path}' has {arr.shape[0]} rows but corpus has {expected_count} — ignoring stale cache")
        return None
    logger.info(f'Loaded embeddings from cache: {path} shape={arr.shape}')
    return arr

def save_embeddings_cache(cache_dir: Path, model_short_name: str, vectors: np.ndarray) -> None:
    """Save embeddings to cache using atomic write."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = get_cache_path(cache_dir, model_short_name)
    with tempfile.NamedTemporaryFile(dir=cache_dir, prefix=f'.tmp_{model_short_name}_', suffix='.npy', delete=False) as tmp_file:
        np.save(tmp_file, vectors)
        tmp_path = Path(tmp_file.name)
    shutil.move(str(tmp_path), str(path))
    logger.info(f'Saved embeddings cache: {path} shape={vectors.shape}')

def load_corpus(input_path: Path) -> list[FAQDocument]:
    """Load corpus with robust error handling."""
    docs = []
    skipped = 0
    with input_path.open(encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                doc = FAQDocument.from_dict(json.loads(line))
                docs.append(doc)
            except json.JSONDecodeError as e:
                logger.warning(f'Line {line_num}: Invalid JSON — {e}')
                skipped += 1
            except Exception as e:
                logger.warning(f'Line {line_num}: Failed to parse document — {e}')
                skipped += 1
    logger.info(f'Loaded {len(docs)} documents from {input_path} ({skipped} skipped)')
    return docs

def load_ner_map(topic_assignments_path: Optional[Path], model_name: str) -> dict[str, dict]:
    """
    Load topic/NER assignments for a specific model.

    Soft failure: returns {} rather than raising so ingestion continues
    without enrichment when the topic file is absent or the model is missing.
    """
    if not topic_assignments_path or not topic_assignments_path.exists():
        logger.warning(f'Topic assignments not found at {topic_assignments_path}')
        return {}
    with topic_assignments_path.open(encoding='utf-8') as f:
        data = json.load(f)
    if 'results' not in data:
        logger.warning("Topic assignments file missing 'results' key")
        return {}
    if model_name not in data['results']:
        logger.warning(f"No topic assignments for '{model_name}' — skipping NER enrichment")
        return {}
    mapping = {a['id']: {'ner_category': a.get('ner_category', 'OTHER'), 'ner_primary_entity': a.get('ner_primary_entity'), 'topic': a.get('topic', -1), 'subtopic': a.get('subtopic')} for a in data['results'][model_name].get('assignments', [])}
    logger.info(f"Loaded NER enrichment for {len(mapping)} documents for model '{model_name}'")
    return mapping

def _doc_id_to_uuid(doc_id: str) -> str:
    """Convert document ID to a UUID string for Qdrant point ID."""
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(doc_id)))

def _prepare_payload(doc: FAQDocument, ner_info: dict) -> dict:
    """Prepare Qdrant payload with NER enrichment."""
    return {'es_id': doc.id, 'question': doc.question, 'answer': doc.answer, 'course': doc.course, 'section': doc.section, 'ner_category': ner_info['ner_category'], 'ner_primary_entity': ner_info['ner_primary_entity'], 'topic': ner_info['topic'], 'subtopic': ner_info['subtopic']}

def _create_points_batch(batch_docs: list[FAQDocument], batch_vecs: np.ndarray, ner_map: dict) -> list[PointStruct]:
    """Create a batch of Qdrant points."""
    points = []
    for doc, vec in zip(batch_docs, batch_vecs):
        ner_info = ner_map.get(doc.id)
        if ner_info is None:
            raise KeyError(f"Document '{doc.id}' missing NER data. Run topic modeling first.")
        points.append(PointStruct(id=_doc_id_to_uuid(doc.id), vector=vec.tolist(), payload=_prepare_payload(doc, ner_info)))
    return points

def _ensure_collection(client: QdrantClient, collection: str, dims: int) -> bool:
    """Create or recreate Qdrant collection. Returns True if successful."""
    try:
        if client.collection_exists(collection):
            logger.info(f'Deleting existing collection: {collection}')
            client.delete_collection(collection)
        client.create_collection(collection_name=collection, vectors_config=VectorParams(size=dims, distance=Distance.COSINE))
        return True
    except UnexpectedResponse as e:
        logger.error(f"Failed to create collection '{collection}': {e}")
        return False

def _verify_count(client: QdrantClient, collection: str, expected: int) -> bool:
    """Verify collection has expected number of points."""
    try:
        count = client.count(collection_name=collection).count
        if count != expected:
            logger.error(f"Expected {expected} docs, got {count} in '{collection}'")
            return False
        return True
    except UnexpectedResponse as e:
        logger.error(f"Failed to verify count for '{collection}': {e}")
        return False

def _get_embeddings(model_entry: dict, docs: list[FAQDocument], config: BenchmarkConfig) -> Optional[np.ndarray]:
    """Get embeddings from cache or compute them."""
    model_name = model_entry['name']
    short_name = model_entry['short_name']
    n_docs = len(docs)
    if not config.force_encode:
        vectors = load_cached_embeddings(config.cache_dir, short_name, n_docs)
        if vectors is not None:
            return vectors
    try:
        logger.info(f"Loading model '{model_name}' for encoding…")
        trust = model_entry.get('trust_remote_code', False)
        model = SentenceTransformer(model_name, trust_remote_code=trust)
        logger.info('Encoding questions…')
        vectors = model.encode([d.question for d in docs], batch_size=config.encode_batch_size, show_progress_bar=True, convert_to_numpy=True)
        save_embeddings_cache(config.cache_dir, short_name, vectors)
        del model
        gc.collect()
        logger.info('Model unloaded from memory.')
        return vectors
    except Exception as e:
        logger.error(f"Failed to encode model '{model_name}': {e}")
        return None

def _load_ner_map(topic_path: Path, model_name: str) -> dict[str, dict]:
    """Load NER/topic enrichment for a model. Raises if missing."""
    import json
    if not topic_path.exists():
        raise FileNotFoundError(f'Topic assignments not found at {topic_path}')
    with topic_path.open(encoding='utf-8') as f:
        data = json.load(f)
    if 'results' not in data:
        raise KeyError(f"Topic assignments file missing 'results' key")
    if model_name not in data['results']:
        raise KeyError(f"Model '{model_name}' not found in topic assignments. Run: uv run python -m rag_pipeline.eda.p02_topic_modeling --embedding-model {model_name} --run-all")
    assignments = data['results'][model_name].get('assignments', [])
    mapping = {}
    for a in assignments:
        doc_id = a['id']
        mapping[doc_id] = {'ner_category': a['ner_category'], 'ner_primary_entity': a.get('ner_primary_entity'), 'topic': a['topic'], 'subtopic': a.get('subtopic')}
    logger.info(f"Loaded NER enrichment for {len(mapping)} documents for '{model_name}'")
    return mapping

def ingest_one_model(*, model_entry: dict, docs: list[FAQDocument], client: QdrantClient, config: BenchmarkConfig) -> bool:
    """Ingest a single model into Qdrant. Hard fails on missing NER data."""
    model_name = model_entry['name']
    short_name = model_entry['short_name']
    collection = model_entry['collection']
    n_docs = len(docs)
    logger.info('=' * 60)
    logger.info(f'Model      : {model_name}')
    logger.info(f'Collection : {collection}')
    if config.skip_existing:
        try:
            if client.collection_exists(collection):
                existing = client.count(collection_name=collection).count
                if existing == n_docs:
                    logger.info(f"Collection '{collection}' already has {existing} docs — skipping.")
                    return True
                logger.info(f"Collection '{collection}' has {existing}/{n_docs} docs — re-indexing.")
        except UnexpectedResponse as e:
            logger.error(f"Failed to check collection '{collection}': {e}")
            return False
    vectors = _get_embeddings(model_entry, docs, config)
    if vectors is None:
        return False
    dims = vectors.shape[1]
    logger.info(f'Embedding dims: {dims}')
    ner_map = _load_ner_map(config.topic_path, model_name)
    missing_docs = [doc.id for doc in docs if doc.id not in ner_map]
    if missing_docs:
        raise RuntimeError(f"❌ Model '{model_name}' missing NER data for {len(missing_docs)} documents.\n   First 5 missing IDs: {missing_docs[:5]}\n\n   Generate missing data with:\n   uv run python -m rag_pipeline.eda.p02_topic_modeling \\\n       --embedding-model {model_name} \\\n       --run-all\n\n   Then re-run this ingestion.")
    logger.info(f'✅ NER coverage: {len(ner_map)}/{n_docs} documents (100%)')
    if not _ensure_collection(client, collection, dims):
        return False
    logger.info(f'Upserting {n_docs} points to Qdrant…')
    for i in tqdm(range(0, n_docs, config.batch_size), desc='Batches'):
        batch_docs = docs[i:i + config.batch_size]
        batch_vecs = vectors[i:i + config.batch_size]
        points = _create_points_batch(batch_docs, batch_vecs, ner_map)
        try:
            client.upsert(collection_name=collection, points=points)
        except UnexpectedResponse as e:
            logger.error(f"Failed to upsert batch to '{collection}': {e}")
            return False
    if not _verify_count(client, collection, n_docs):
        return False
    logger.info(f"✅ Done: {n_docs} documents indexed in '{collection}'")
    return True

def main() -> None:
    parser = create_ingestion_parser()
    args = parser.parse_args()
    config = BenchmarkConfig.from_defaults().merge_args(args)
    if not config.clean_path or not config.clean_path.exists():
        logger.error(f'Corpus not found: {config.clean_path}')
        return
    docs = load_corpus(config.clean_path)
    if not docs:
        logger.error('No documents loaded. Exiting.')
        return
    try:
        client = config.make_qdrant_client()
        logger.info(f'Connected to Qdrant at {config.qdrant_host}:{config.qdrant_port}')
    except Exception as e:
        logger.error(f'Failed to connect to Qdrant: {e}')
        return
    try:
        model_entries = config.get_model_entries()
    except (FileNotFoundError, ValueError, KeyError) as e:
        logger.error(f'Failed to load model registry: {e}')
        return
    if not model_entries:
        logger.error('No models to ingest. Check models.json')
        return
    logger.info(f"Ingesting {len(model_entries)} model(s): {[m['name'] for m in model_entries]}")
    failed_models: list[str] = []
    successful_models: list[str] = []
    for entry in model_entries:
        success = ingest_one_model(model_entry=entry, docs=docs, client=client, config=config)
        if success:
            successful_models.append(entry['name'])
        else:
            failed_models.append(entry['name'])
    logger.info('=' * 60)
    logger.info('INGESTION COMPLETE')
    logger.info('=' * 60)
    logger.info(f'Successful: {len(successful_models)} models')
    for name in successful_models:
        logger.info(f'   - {name}')
    if failed_models:
        logger.warning(f'Failed: {len(failed_models)} models')
        for name in failed_models:
            logger.warning(f'   - {name}')
    else:
        logger.info('All models ingested successfully.')
if __name__ == '__main__':
    main()
"""
rag_pipeline/p04_ingestion/_onnx_bench_engine.py
RESPONSIBILITY: Manages Qdrant network operations and payload transformation mappings.
"""
import logging
from typing import Any, Dict, List
from qdrant_client import QdrantClient
from ._onnx_cross_encoder import ONNXCrossEncoder
from ._onnx_bench_id import normalize_target_ids
logger = logging.getLogger(__name__)
PAYLOAD_KEY_TEXT = 'text'
PAYLOAD_KEY_ANSWER = 'answer'

def compile_onnx_runtime_node(model_key: str, max_length: int=512, provider: str='CPUExecutionProvider') -> ONNXCrossEncoder:
    """RESPONSIBILITY: Instantiates localized ONNX cross-encoder model engines."""
    logger.info('Compiling/Loading ONNX Node Graph | model=%s provider=%s', model_key, provider)
    try:
        return ONNXCrossEncoder(model_name=model_key, max_length=max_length, provider=provider)
    except Exception as e:
        logger.error('Failed to compile ONNX runtime for %s: %s', model_key, e)
        raise

def _execute_qdrant_retrieve(client: QdrantClient, collection: str, valid_ids: List[str]) -> List[Any]:
    """RESPONSIBILITY: Handles the direct retrieval network transmission pass to the Qdrant cluster."""
    try:
        return client.retrieve(collection_name=collection, ids=valid_ids, with_payload=True, with_vectors=False)
    except Exception as e:
        logger.error('Dynamic Qdrant point retrieval network call failed: %s', e)
        return []

def _build_candidate_record(point: Any, id_mapping: Dict[str, str]) -> Dict[str, Any]:
    """RESPONSIBILITY: Parses text payloads from a database point into structured domain dictionaries."""
    pid = str(point.id).lower()
    payload = point.payload or {}
    text_content = payload.get(PAYLOAD_KEY_TEXT, payload.get(PAYLOAD_KEY_ANSWER, ''))
    if not text_content:
        text_content = '[NO CONTENT]'
    return {'es_id': id_mapping.get(pid, pid), 'question': payload.get('question', ''), 'answer': text_content, 'payload': payload}

def prepare_candidates_from_hits(client: QdrantClient, collection: str, hit_ids: List[Any], top_k: int=20) -> List[Dict[str, Any]]:
    """RESPONSIBILITY: Top-level orchestrator organizing key lookups and payload data mapping streams."""
    if not hit_ids:
        return []
    limit = min(len(hit_ids), top_k)
    cleaned_ids, id_mapping = normalize_target_ids(hit_ids, limit)
    points = _execute_qdrant_retrieve(client, collection, cleaned_ids)
    if not points:
        logger.warning('Qdrant retrieve call returned zero points for verified UUID list: %s', cleaned_ids)
        return []
    return [_build_candidate_record(point, id_mapping) for point in points]

def verify_live_infrastructure(client: QdrantClient, collection_name: str) -> None:
    """RESPONSIBILITY: Pre-flight check verifying database collections are healthy and reachable."""
    try:
        collection_info = client.get_collection(collection_name=collection_name)
        logger.info("Collection '%s' verified live with %d active points", collection_name, collection_info.points_count)
    except Exception as e:
        logger.error("Pre-flight cluster validation failed for '%s': %s", collection_name, e)
        import sys
        sys.exit(1)
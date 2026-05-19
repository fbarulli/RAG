"""
production_pipeline/p04_ingestion/_onnx_bench_engine.py

Core engine components for ONNX benchmarking.
Handles file-level data extraction, deterministic UUID mapping, and ONNX graph compilation.
"""
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from qdrant_client import QdrantClient
from ._onnx_cross_encoder import ONNXCrossEncoder

logger = logging.getLogger(__name__)

# Constants for payload keys to avoid magic strings
PAYLOAD_KEY_TEXT = "text"
PAYLOAD_KEY_ANSWER = "answer"
DEFAULT_COLLECTION = "faqs_bge_base_en_v1_5"
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"


def load_matrix_configs(config_path: str = "configs/rerankers.json") -> List[Dict[str, Any]]:
    """
    Loads and returns cross-encoder models from the matrix configuration file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        A list of model configuration dictionaries.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    path = Path(config_path)
    if not path.exists():
        logger.error("Matrix configuration file missing: %s", config_path)
        raise FileNotFoundError(f"Critical configuration missing: {config_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            models = data.get("models", [])
            if not isinstance(models, list):
                logger.warning("Config 'models' key is not a list, returning empty list.")
                return []
            return models
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in matrix configuration: %s", e)
        raise


def extract_active_environment() -> Tuple[str, str]:
    """
    Reads system defaults to isolate active production models and target collections.
    Attempts to import from local benchmark config. Falls back to defaults if configuration is unavailable.

    Returns:
        A tuple containing (model_name, collection_name).
    """
    model = DEFAULT_MODEL
    collection = DEFAULT_COLLECTION

    try:
        # Dynamic import to avoid circular dependencies if possible
        from production_pipeline.p04_ingestion._benchmark_config import load_defaults
        defaults_data = load_defaults()
        model = defaults_data.get("production_model", model)
        collection = defaults_data.get("qdrant", {}).get("collection", collection)
        logger.info("Loaded environment defaults: model=%s, collection=%s", model, collection)
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning("Benchmark config module not found, using defaults. Error: %s", e)
    except Exception as e:
        logger.error("Failed to load environment defaults: %s", e)
        logger.warning("Falling back to default model and collection.")

    return model, collection


def compile_onnx_runtime_node(
    model_key: str,
    max_length: int = 512,
    provider: str = "CPUExecutionProvider"
) -> ONNXCrossEncoder:
    """
    Compiles a specific Hugging Face Transformer checkpoint into an ONNX graph.

    NOTE: Caching has been removed to prevent mutable state leakage.
    The caller is responsible for lifecycle management (instantiating once per benchmark run).

    Args:
        model_key: The Hugging Face model identifier or path.
        max_length: Maximum sequence length for tokenization.
        provider: ONNX execution provider (e.g., CPUExecutionProvider, CUDAExecutionProvider).

    Returns:
        An initialized ONNXCrossEncoder instance.
    """
    logger.info("Compiling/Loading ONNX Node Graph | model=%s provider=%s", model_key, provider)
    try:
        return ONNXCrossEncoder(
            model_name=model_key,
            max_length=max_length,
            provider=provider
        )
    except Exception as e:
        logger.error("Failed to compile ONNX runtime for %s: %s", model_key, e)
        raise


def map_to_valid_uuid(raw_id: Any) -> str:
    """
    Ensures an ID complies with Qdrant schemas.
    Converts strings to deterministic UUIDv3 format if they are not already valid UUIDs or numeric strings.
    Output is normalized to lowercase to ensure consistent dictionary lookups.

    Args:
        raw_id: The raw identifier (int, str, etc.).

    Returns:
        A string representation of the ID suitable for Qdrant.
    """
    raw_id_str = str(raw_id).strip()

    # 1. Check if it's a numeric string (Qdrant supports int IDs)
    if raw_id_str.isdigit():
        return raw_id_str

    # 2. Check if it's already a valid UUID
    try:
        # Validate UUID format
        uuid_obj = uuid.UUID(raw_id_str)
        # Return normalized lowercase string
        return str(uuid_obj).lower()
    except ValueError:
        pass

    # 3. Convert to deterministic UUIDv3 if neither
    logger.debug("Mapping non-UUID ID '%s' to UUIDv3", raw_id_str)
    return str(uuid.uuid3(uuid.NAMESPACE_DNS, raw_id_str)).lower()


def prepare_candidates_from_hits(
    client: QdrantClient,
    collection: str,
    hit_ids: List[str],
    top_k: int = 20
) -> List[Dict[str, Any]]:
    """
    Retrieves text payloads from Qdrant, abstracting ID translation mappings.
    Uses case-insensitive ID matching to prevent UUID casing mismatches.

    Args:
        client: Active QdrantClient instance.
        collection: Name of the Qdrant collection.
        hit_ids: List of original hit IDs to retrieve.
        top_k: Maximum number of candidates to retrieve.

    Returns:
        A list of dictionaries containing 'es_id', 'question', 'answer', and 'payload'.
    """
    if not hit_ids:
        return []

    limit = min(len(hit_ids), top_k)

    # Create mapping: Normalized Qdrant ID -> Original Hit ID
    # We normalize keys to lowercase to handle UUID case-insensitivity
    id_mapping: Dict[str, str] = {}
    valid_qdrant_ids: List[Union[str, int]] = []

    for hid in hit_ids[:limit]:
        qdrant_id = map_to_valid_uuid(hid)
        valid_qdrant_ids.append(qdrant_id)
        # Map Normalized Qdrant ID back to Original ID
        id_mapping[qdrant_id.lower()] = str(hid)

    try:
        points = client.retrieve(
            collection_name=collection,
            ids=valid_qdrant_ids,
            with_payload=True,
            with_vectors=False
        )
    except Exception as e:
        logger.error("Dynamic Qdrant point retrieval failed: %s", e)
        # Return empty list to signal failure without polluting benchmarks with mock data
        return []

    candidates = []
    retrieved_ids_normalized = set()

    for point in points:
        # Normalize the returned ID for consistent lookup
        point_id_str = str(point.id).lower()
        retrieved_ids_normalized.add(point_id_str)

        # Safely extract dynamic text keys
        payload = point.payload or {}
        text_content = payload.get(PAYLOAD_KEY_TEXT, payload.get(PAYLOAD_KEY_ANSWER, ""))

        # Fallback if content is missing
        if not text_content:
            logger.warning("Point %s has no text content in payload.", point.id)
            text_content = "[NO CONTENT]"

        # Retrieve original ID using normalized key
        original_id = id_mapping.get(point_id_str, str(point.id))

        candidates.append({
            "es_id": original_id,
            "question": text_content,
            "answer": text_content,
            "payload": payload
        })

    # Check for missing IDs to log warnings (Qdrant retrieve doesn't return missing IDs)
    missing_count = limit - len(retrieved_ids_normalized)
    if missing_count > 0:
        logger.warning(
            "Qdrant retrieval missing %d/%d expected points for collection %s",
            missing_count, limit, collection
        )

    return candidates

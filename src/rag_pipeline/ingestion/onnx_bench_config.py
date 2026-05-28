"""
rag_pipeline/p04_ingestion/_onnx_bench_config.py
RESPONSIBILITY: Manages loading and resolving filesystem configuration matrices.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
logger = logging.getLogger(__name__)

def load_matrix_configs(config_path: str='configs/rerankers.json') -> List[Dict[str, Any]]:
    """RESPONSIBILITY: Parses cross-encoder matrix models out of the JSON file configuration."""
    path = Path(config_path)
    if not path.exists():
        logger.error('Matrix configuration file missing: %s', config_path)
        raise FileNotFoundError(f'Critical configuration missing: {config_path}')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            models = data.get('models', [])
            return models if isinstance(models, list) else []
    except json.JSONDecodeError as e:
        logger.error('Invalid JSON in matrix configuration: %s', e)
        raise

def extract_active_environment() -> Tuple[str, str]:
    """RESPONSIBILITY: Isolates operational database collection names and production model identifiers."""
    try:
        from rag_pipeline.ingestion.benchmark_loader import load_defaults
        defaults_data = load_defaults()
        model = defaults_data.get("production_model")
        collection = defaults_data.get("qdrant", {}).get("collection")
        if not model:
            raise ValueError("production_model missing in defaults.json")
        logger.info("Loaded environment defaults: model=%s, collection=%s", model, collection)
    except Exception as e:
        logger.error("Failed to load defaults.json: %s", e)
        raise
    return (model, collection)

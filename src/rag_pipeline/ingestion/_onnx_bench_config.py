"""
production_pipeline/p04_ingestion/_onnx_bench_config.py
RESPONSIBILITY: Manages loading and resolving filesystem configuration matrices.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
logger = logging.getLogger(__name__)
DEFAULT_COLLECTION = 'faqs_bge_base_en_v1_5'
DEFAULT_MODEL = 'BAAI/bge-base-en-v1.5'

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
    model = DEFAULT_MODEL
    collection = DEFAULT_COLLECTION
    try:
        from rag_pipeline.ingestion._benchmark_config import load_defaults
        defaults_data = load_defaults()
        model = defaults_data.get('production_model', model)
        collection = defaults_data.get('qdrant', {}).get('collection', collection)
        logger.info('Loaded environment defaults: model=%s, collection=%s', model, collection)
    except Exception as e:
        logger.warning('Failed to load environment defaults, using primitives fallback. Error: %s', e)
    return (model, collection)
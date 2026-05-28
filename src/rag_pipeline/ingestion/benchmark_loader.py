"""
_benchmark_loader.py
====================
Load test data, topic assignments, retrieval configs, and the model registry
for the benchmark pipeline.

Single responsibility: I/O for benchmark inputs.
No metric computation, no retrieval logic, no reporting.

Public API
----------
    load_defaults()                          -> dict
    load_model_registry()                    -> list[dict]
    get_model_entry(name)                    -> dict
    load_test_set(path, clean_path)          -> list[dict]
    load_topic_assignments(path, model)      -> dict[str, dict]
    load_configs(path)                       -> dict
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
from rag_pipeline.logging import get_logger
logger = get_logger(__name__)
_HERE = Path(__file__).resolve().parent

def find_project_root(start_path: Path) -> Path:
    """Find project root by looking for 'configs' directory."""
    for parent in start_path.parents:
        if (parent / 'configs').exists():
            return parent
    return start_path.parents[3]
_PROJECT_ROOT = find_project_root(_HERE)
_CONFIGS_DIR = _PROJECT_ROOT / 'configs'
DEFAULTS_PATH = _CONFIGS_DIR / 'defaults.json'
MODELS_CONFIG_PATH = _CONFIGS_DIR / 'models.json'

def load_defaults() -> dict:
    """Load defaults.json, raising FileNotFoundError if missing."""
    if not DEFAULTS_PATH.exists():
        raise FileNotFoundError(f'defaults.json not found at {DEFAULTS_PATH}')
    with DEFAULTS_PATH.open(encoding='utf-8') as f:
        return json.load(f)

def load_model_registry(enabled_only: bool=True) -> list[dict]:
    """
    Return the list of model definitions from configs/models.json.

    Parameters
    ----------
    enabled_only:
        When True (default), only models with ``"enabled": true`` are returned.
    """
    if not MODELS_CONFIG_PATH.exists():
        raise FileNotFoundError(f'models.json not found at {MODELS_CONFIG_PATH}')
    with MODELS_CONFIG_PATH.open(encoding='utf-8') as f:
        data = json.load(f)
    models = data.get('models', [])
    if not models:
        raise ValueError(f'No models found in {MODELS_CONFIG_PATH}')
    if enabled_only:
        models = [m for m in models if m.get('enabled', True)]
        if not models:
            raise ValueError(f'No enabled models found in {MODELS_CONFIG_PATH}')
    logger.info(f'Loaded {len(models)} model entries from {MODELS_CONFIG_PATH}')
    return models

def get_model_entry(model_name: str) -> dict:
    """Return the registry entry for *model_name*, raising KeyError if absent."""
    registry = load_model_registry(enabled_only=False)
    for entry in registry:
        if entry['name'] == model_name:
            return entry
    available = [m['name'] for m in registry]
    raise KeyError(f"Model '{model_name}' not found in models.json. Available: {available}")

def get_winner_model() -> dict:
    """Return the model entry marked ``"winner": true``."""
    for entry in load_model_registry(enabled_only=False):
        if entry.get('winner'):
            return entry
    raise RuntimeError('No model marked as winner in models.json.')

def _course_name_map() -> dict[str, str]:
    """Load course name map from defaults.json. Raise if missing."""
    try:
        defaults = load_defaults()
        course_map = defaults.get('course_name_map')
        if not course_map:
            raise KeyError('course_name_map missing from defaults.json')
        return course_map
    except FileNotFoundError:
        logger.warning('defaults.json not found, using hardcoded course map')
        return {'ml-zoomcamp': 'machine-learning-zoomcamp', 'de-zoomcamp': 'data-engineering-zoomcamp', 'mlops-zoomcamp': 'mlops-zoomcamp', 'llm-zoomcamp': 'llm-zoomcamp'}

def load_valid_ids(clean_path: Path) -> set[str]:
    """Return the set of all document IDs present in clean.jsonl."""
    if not clean_path.exists():
        raise FileNotFoundError(f'clean.jsonl not found at {clean_path}')
    ids: set[str] = set()
    with clean_path.open(encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                if 'id' not in doc:
                    raise KeyError(f"Line {line_num}: missing 'id' field")
                ids.add(doc['id'])
            except json.JSONDecodeError as e:
                raise ValueError(f'Line {line_num}: invalid JSON - {e}')
    return ids

def load_test_set(path: Path, clean_path: Optional[Path]=None) -> list[dict]:
    """
    Load and validate the benchmark test set from a JSONL file.
    
    Handles OOD queries where expected_id is intentionally None.
    """
    if not path.exists():
        raise FileNotFoundError(f'Test set not found: {path}')
    valid_ids = load_valid_ids(clean_path) if clean_path else None
    course_map = _course_name_map()
    required_fields = {'id', 'question', 'answer', 'course'}
    tests: list[dict] = []
    skipped = 0
    with path.open(encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f'Test set line {line_num}: JSON error - {e}')
                skipped += 1
                continue
            missing = required_fields - doc.keys()
            if missing:
                logger.warning(f'Test set line {line_num}: missing fields {missing}')
                skipped += 1
                continue
            expected_id = doc.get('expected_id') or doc.get('expected_doc_id') or doc['id']
            is_ood = expected_id is None or doc.get('is_ood', False)
            if not is_ood and valid_ids and (expected_id not in valid_ids):
                logger.warning(f"Test set line {line_num}: expected_id '{expected_id}' not in corpus - skipping")
                skipped += 1
                continue
            tests.append({'query_id': doc['id'], 'query': doc['question'], 'query_type': doc.get('query_type', 'unknown'), 'expected_id': expected_id, 'course': course_map.get(doc['course'], doc['course']) if doc.get('course') else None, 'section': doc.get('section', ''), 'answer': doc['answer'], 'is_ood': is_ood})
    logger.info(f'Loaded {len(tests)} test queries from {path} ({skipped} skipped)')
    ood_count = sum((1 for t in tests if t.get('is_ood', False)))
    if ood_count:
        logger.info(f'  Including {ood_count} OOD queries (expected_id=None)')
    return tests

def load_topic_assignments(path: Path, model: str) -> dict[str, dict]:
    """
    Load topic/NER assignments keyed by document ID.

    Expects format: {results: {model_name: {assignments: [...]}}}
    Raises KeyError if model not found.
    """
    if not path.exists():
        raise FileNotFoundError(f'Topic assignments not found: {path}')
    with path.open(encoding='utf-8') as f:
        data = json.load(f)
    if 'results' not in data:
        raise KeyError(f"Topic assignments file missing 'results' key: {path}")
    if model not in data['results']:
        available = list(data['results'].keys())
        raise KeyError(f"Model '{model}' not found in topic assignments. Available: {available}")
    assignments_list = data['results'][model].get('assignments', [])
    mapping = {a['id']: a for a in assignments_list}
    logger.info(f"Loaded {len(mapping)} topic assignments for model '{model}' from {path}")
    return mapping

def load_configs(path: Path) -> dict:
    """Load retrieval configuration definitions from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f'Retrieval configs not found: {path}')
    with path.open(encoding='utf-8') as f:
        configs = json.load(f)
    if not configs:
        raise ValueError(f'No configurations found in {path}')
    logger.info(f'Loaded {len(configs)} retrieval configs from {path}')
    return configs
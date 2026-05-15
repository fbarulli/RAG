# /home/admin/LLM/LLM/01/web/src/config_manager.py

import json
import logging
import traceback
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _load_json_config(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise


def load_base_config() -> Dict[str, Any]:
    web_root = '/home/admin/LLM/LLM/01/web'
    base_config_path = f'{web_root}/configs/settings.json'
    return _load_json_config(base_config_path)


def load_search_config(config_name: str) -> Dict[str, Any]:
    web_root = '/home/admin/LLM/LLM/01/web'
    search_configs_path = f'{web_root}/configs/search_configs.json'
    all_configs = _load_json_config(search_configs_path)
    
    if config_name not in all_configs:
        raise KeyError(f"Config '{config_name}' not found in search_configs.json. Available: {list(all_configs.keys())}")
    
    return all_configs[config_name]


def load_full_config(config_name: str) -> Dict[str, Any]:
    base = load_base_config()
    search = load_search_config(config_name)
    
    merged = {**base, **search}
    
    if 'name' not in merged:
        merged['name'] = config_name
    
    logger.info(f"Loaded config: {merged.get('name')} (search_type: {merged.get('search_type')})")
    return merged

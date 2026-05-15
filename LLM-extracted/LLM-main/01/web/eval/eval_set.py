import json
import logging
import traceback
from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def clean_query(q: str) -> str:
    """Remove common artifacts from eval set questions."""
    q = q.strip()
    # Remove surrounding quotes
    if (q.startswith("'") and q.endswith("'")) or (q.startswith('"') and q.endswith('"')):
        q = q[1:-1]
    # Remove 'Course:' prefix (case‑insensitive)
    if q.lower().startswith("course:"):
        q = q[6:].lstrip()
    # Remove stray leading colon or space
    if q.startswith(':'):
        q = q[1:].lstrip()
    return q

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

def get_eval_set_from_es() -> List[Dict[str, Any]]:
    web_root = '/home/admin/LLM/LLM/01/web'
    settings_path = f'{web_root}/configs/settings.json'
    es_search_path = f'{web_root}/configs/elastic_search.json'
    
    settings = _load_json_config(settings_path)
    es_search_config = _load_json_config(es_search_path)
    
    es_host = settings.get('es_host')
    index_name = settings.get('index_name')
    size_limit = es_search_config.get('size_limit', 10000)
    
    if not es_host:
        raise ValueError("es_host not found in configs/settings.json")
    if not index_name:
        raise ValueError("index_name not found in configs/settings.json")
    
    try:
        es = Elasticsearch(es_host)
        if not es.ping():
            raise ConnectionError(f"Cannot connect to Elasticsearch at {es_host}")
        
        response = es.search(
            index=index_name,
            size=size_limit,
            query={"match_all": {}}
        )
        
        eval_set = []
        for hit in response['hits']['hits']:
            original_doc = hit['_source'].copy()
            if 'question' in original_doc:
                original_doc['question'] = clean_query(original_doc['question'])
            eval_set.append({
                'expected_id': hit['_id'],
                'original_doc': original_doc
            })
        
        logger.info(f"Loaded and cleaned {len(eval_set)} documents from index '{index_name}'")
        return eval_set
        
    except ConnectionError as e:
        logger.error(f"Failed to connect to Elasticsearch: {e}")
        logger.error(traceback.format_exc())
        raise
    except NotFoundError as e:
        logger.error(f"Index '{index_name}' not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        raise
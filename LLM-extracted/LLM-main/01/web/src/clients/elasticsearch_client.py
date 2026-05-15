from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch
from src.logger_config import logger
import traceback

class ElasticsearchClient:
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.es_client: Optional[Elasticsearch] = None
        self.index_name = settings.get("index_name", "course-questions")
    
    def connect(self) -> None:
        host = self.settings.get("es_host", "http://localhost:9200")
        try:
            self.es_client = Elasticsearch(host)
            if not self.es_client.info():
                raise ConnectionError("ES info check failed")
            logger.info(f"Connected to Elasticsearch at {host}")
        except Exception:
            logger.error(f"Elasticsearch connection failed: {traceback.format_exc()}")
            raise
    
    def search(self, query_body: Dict[str, Any], size: int, **kwargs) -> List[Dict]:
        if not self.es_client:
            raise RuntimeError("Elasticsearch client not connected")
        try:
            response = self.es_client.search(index=self.index_name, size=size, body=query_body, **kwargs)
            return response.get('hits', {}).get('hits', [])
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def msearch(self, searches: List[Dict]) -> List[List[Dict]]:
        if not self.es_client:
            raise RuntimeError("Elasticsearch client not connected")
        try:
            response = self.es_client.msearch(body=searches)
            results = []
            for resp in response['responses']:
                hits = resp['hits']['hits'] if 'hits' in resp else []
                results.append(hits)
            return results
        except Exception as e:
            logger.error(f"msearch failed: {e}")
            return [[] for _ in range(len(searches)//2)]

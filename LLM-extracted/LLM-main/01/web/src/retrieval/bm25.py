from typing import List, Dict, Any, Optional
from src.clients.elasticsearch_client import ElasticsearchClient
from src.logger_config import logger
import traceback

class BM25Retriever:
    def __init__(self, es_client: ElasticsearchClient, settings: Dict[str, Any]):
        self.es_client = es_client
        self.settings = settings
    
    def search(self, query: str, size: int, course_context: Optional[str]) -> List[Dict]:
        boost_q = self.settings.get('boost_question', 20)
        boost_t = self.settings.get('boost_text', 1)
        mm_query = {
            "multi_match": {
                "query": query,
                "fields": [f"question^{boost_q}", f"text^{boost_t}"],
                "type": self.settings.get("bm25_type", "best_fields")
            }
        }
        if "minimum_should_match" in self.settings:
            mm_query["multi_match"]["minimum_should_match"] = self.settings["minimum_should_match"]
        
        if course_context:
            final_query = {
                "bool": {
                    "must": mm_query,
                    "filter": {"term": {"course": course_context}}
                }
            }
        else:
            final_query = mm_query
        
        try:
            response = self.es_client.search(final_query, size)
            return response
        except Exception:
            logger.error(f"BM25 search failed: {traceback.format_exc()}")
            return []
    
    def batch_search(self, queries: List[str], k: int, course_context: Optional[str]) -> List[List[Dict]]:
        searches = []
        boost_q = self.settings.get('boost_question', 20)
        boost_t = self.settings.get('boost_text', 1)
        for query in queries:
            mm_query = {
                "multi_match": {
                    "query": query,
                    "fields": [f"question^{boost_q}", f"text^{boost_t}"],
                    "type": "best_fields"
                }
            }
            if course_context:
                final_query = {
                    "bool": {
                        "must": mm_query,
                        "filter": {"term": {"course": course_context}}
                    }
                }
            else:
                final_query = mm_query
            searches.append({"index": self.es_client.index_name})
            searches.append({"size": k, "query": final_query})
        return self.es_client.msearch(searches)
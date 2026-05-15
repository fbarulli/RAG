from typing import List, Dict, Any, Optional
from src.clients.elasticsearch_client import ElasticsearchClient
from src.embedding.embedding_service import EmbeddingService
from src.logger_config import logger
import traceback

class VectorRetriever:
    def __init__(self, es_client: ElasticsearchClient, settings: Dict[str, Any], embed_service: EmbeddingService):
        self.es_client = es_client
        self.settings = settings
        self.embed_service = embed_service
    
    def search(self, query: str, size: int, course_context: Optional[str]) -> List[Dict]:
        query_vector = self.embed_service.get_embedding(query)
        if not query_vector:
            return []
        script_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'question_vector') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
        if course_context:
            final_query = {
                "bool": {
                    "must": script_query,
                    "filter": {"term": {"course": course_context}}
                }
            }
        else:
            final_query = script_query
        try:
            response = self.es_client.search(final_query, size)
            return response
        except Exception:
            logger.error(f"Vector search failed: {traceback.format_exc()}")
            return []
    
    def batch_search(self, queries: List[str], k: int, course_context: Optional[str]) -> List[List[Dict]]:
        # Fall back to individual searches for simplicity
        results = []
        for q in queries:
            hits = self.search(q, k, course_context)
            results.append(hits)
        return results

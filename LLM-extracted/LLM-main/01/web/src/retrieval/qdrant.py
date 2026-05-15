"""
src/retrieval/qdrant.py
=======================
Qdrant vector retriever for semantic search.

Uses sentence-transformers for query embedding and Qdrant for
vector similarity search.

Requires: qdrant-client, sentence-transformers
"""
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


class QdrantRetriever:
    def __init__(self, host: str = 'localhost', port: int = 6333,
                 collection_name: str = 'faqs',
                 model_name: str = 'all-MiniLM-L6-v2'):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, size: int = 5,
               course_filter: Optional[str] = None) -> List[Dict]:
        """Search for documents similar to query."""
        query_vector = self.model.encode(query).tolist()

        search_filter = None
        if course_filter:
            search_filter = Filter(
                must=[FieldCondition(key='course', match=MatchValue(value=course_filter))]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=size,
            query_filter=search_filter,
            with_payload=True,
        )

        return [
            {
                'id': hit.payload.get('es_id', ''),
                'question': hit.payload.get('question', ''),
                'answer': hit.payload.get('answer', ''),
                'course': hit.payload.get('course', ''),
                'section': hit.payload.get('section', ''),
                'score': hit.score,
            }
            for hit in results.points
        ]

    def batch_search(self, queries: List[str], size: int = 5,
                     course_filter: Optional[str] = None) -> List[List[Dict]]:
        """Search for multiple queries."""
        return [self.search(q, size, course_filter) for q in queries]

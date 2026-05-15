from typing import List, Dict, Any, Optional
from .bm25 import BM25Retriever
from .vector import VectorRetriever
from src.logger_config import logger

class HybridRetriever:
    def __init__(self, bm25_retriever: BM25Retriever, vector_retriever: Optional[VectorRetriever], settings: Dict[str, Any]):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.settings = settings
    
    def search(self, query: str, size: int, course_context: Optional[str]) -> List[Dict]:
        bm25_hits = self.bm25.search(query, size * 2, course_context)
        if not self.vector:
            return bm25_hits[:size]
        vector_hits = self.vector.search(query, size * 2, course_context)
        # Reciprocal Rank Fusion
        scores = {}
        hit_map = {}
        for idx, hit in enumerate(bm25_hits):
            hit_map[hit['_id']] = hit
            scores[hit['_id']] = scores.get(hit['_id'], 0) + 1.0 / (60 + idx + 1)
        for idx, hit in enumerate(vector_hits):
            if hit['_id'] not in hit_map:
                hit_map[hit['_id']] = hit
            scores[hit['_id']] = scores.get(hit['_id'], 0) + 1.0 / (60 + idx + 1)
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:size]
        return [hit_map[doc_id] for doc_id in sorted_ids if doc_id in hit_map]
    
    def batch_search(self, queries: List[str], k: int, course_context: Optional[str]) -> List[List[Dict]]:
        # Batch BM25 and vector individually, then fuse per query
        bm25_results = self.bm25.batch_search(queries, k * 2, course_context)
        if not self.vector:
            return [res[:k] for res in bm25_results]
        vector_results = self.vector.batch_search(queries, k * 2, course_context)
        final_results = []
        for i in range(len(queries)):
            scores = {}
            hit_map = {}
            for idx, hit in enumerate(bm25_results[i]):
                hit_map[hit['_id']] = hit
                scores[hit['_id']] = scores.get(hit['_id'], 0) + 1.0 / (60 + idx + 1)
            for idx, hit in enumerate(vector_results[i]):
                if hit['_id'] not in hit_map:
                    hit_map[hit['_id']] = hit
                scores[hit['_id']] = scores.get(hit['_id'], 0) + 1.0 / (60 + idx + 1)
            sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
            fused = [hit_map[doc_id] for doc_id in sorted_ids if doc_id in hit_map]
            final_results.append(fused)
        return final_results
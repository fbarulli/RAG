"""Pure Vector retrieval via Qdrant."""
import time
from typing import Optional, Any
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams, SparseVector, NamedSparseVector
from ..benchmark_types import SearchResult

def build_qdrant_filter(course: Optional[str], must_configs: list = None, should_configs: list = None) -> Optional[Filter]:
    must = []
    if course:
        must.append(FieldCondition(key='course', match=MatchValue(value=course)))
    if must_configs:
        for cond in must_configs:
            if 'key' in cond and 'value' in cond.get('match', {}):
                must.append(FieldCondition(key=cond['key'], match=MatchValue(value=cond['match']['value'])))
    
    should = []
    if should_configs:
        for cond in should_configs:
            if 'key' in cond and 'value' in cond.get('match', {}):
                should.append(FieldCondition(key=cond['key'], match=MatchValue(value=cond['match']['value'])))
                
    return Filter(must=must or None, should=should or None) if (must or should) else None

def parse_qdrant_points(points: Any, latency_ms: float) -> SearchResult:
    return SearchResult(
        hit_ids=tuple(str(p.payload.get('es_id', '')) for p in points),  # coerce legacy int es_ids
        hit_courses=tuple(p.payload.get('course', '') for p in points),
        hit_scores=tuple(float(p.score) if p.score is not None else 0.0 for p in points),
        top_answer=points[0].payload.get('answer') if points else None,
        latency_ms=latency_ms,
        hit_answers=tuple(p.payload.get('answer', '') for p in points),
        hit_questions=tuple(p.payload.get('question', '') for p in points)
    )

def run_vector_retrieval(client, collection: str, query_vector: list, course_filter: Optional[str], config: dict, top_k: int) -> SearchResult:
    start = time.perf_counter()
    
    must_configs = config.get('filters', {}).get('must', [])
    query_filter = build_qdrant_filter(course_filter, must_configs=must_configs)
    
    search_params = SearchParams(hnsw_ef=config['hnsw_ef']) if config.get('hnsw_ef') else None
    
    result = client.query_points(
        collection_name=collection, 
        query=query_vector, 
        limit=config.get('limit', top_k), 
        query_filter=query_filter, 
        score_threshold=config.get('score_threshold'), 
        search_params=search_params
    )
    
    return parse_qdrant_points(result.points, (time.perf_counter() - start) * 1000)



def run_sparse_retrieval(client, collection: str, query_indices: list, query_values: list, course_filter: Optional[str], config: dict, top_k: int) -> SearchResult:
    start = time.perf_counter()
    query_filter = build_qdrant_filter(course_filter)
    sparse_vector_name = config.get("sparse_vector_name", "sparse")
    result = client.query_points(
        collection_name=collection,
        query=SparseVector(indices=query_indices, values=query_values),
        using=sparse_vector_name,
        limit=config.get("limit", top_k),
        query_filter=query_filter,
    )
    return parse_qdrant_points(result.points, (time.perf_counter() - start) * 1000)

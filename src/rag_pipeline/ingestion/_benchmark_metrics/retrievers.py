"""
production_pipeline/p04_ingestion/_benchmark_metrics/retrievers.py
Retrieval implementations for different search types.
All functions return SearchResult.
"""
import time
from typing import Optional, TYPE_CHECKING
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams
from .._benchmark_types import SearchResult
if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

def run_es_retrieval(es: 'Elasticsearch', index: str, query_text: str, course_filter: str, config: dict, top_k: int) -> SearchResult:
    """BM25 retrieval via Elasticsearch."""
    start = time.perf_counter()
    boost_q = config.get('boost_question', 1.0)
    boost_t = config.get('boost_text', 1.0)
    if boost_q > 0 and boost_t > 0:
        must_clause = [{'multi_match': {'query': query_text, 'fields': [f'question^{boost_q}', f'answer^{boost_t}']}}]
    elif boost_q > 0:
        must_clause = [{'match': {'question': {'query': query_text, 'boost': boost_q}}}]
    else:
        must_clause = [{'match': {'answer': {'query': query_text, 'boost': boost_t}}}]
    body = {'query': {'bool': {'must': must_clause, 'filter': [{'term': {'course': course_filter}}] if course_filter else []}}, 'size': top_k}
    resp = es.search(index=index, body=body)
    latency_ms = (time.perf_counter() - start) * 1000
    hits = resp['hits']['hits']
    hit_ids = tuple((h['_source'].get('es_id', '') for h in hits))
    hit_courses = tuple((h['_source'].get('course', '') for h in hits))
    hit_scores = tuple((float(h['_score']) for h in hits))
    top_answer = hits[0]['_source'].get('answer') if hits else None
    return SearchResult(hit_ids=hit_ids, hit_scores=hit_scores, hit_courses=hit_courses, top_answer=top_answer, latency_ms=latency_ms)

def run_vector_retrieval(client, collection: str, query_vector: list, course_filter: str, config: dict, top_k: int) -> SearchResult:
    """Pure vector retrieval via Qdrant."""
    start = time.perf_counter()
    must_conditions: list = []
    if course_filter:
        must_conditions.append(FieldCondition(key='course', match=MatchValue(value=course_filter)))
    for cond in config.get('filters', {}).get('must', []):
        key = cond.get('key')
        match = cond.get('match', {})
        if key and 'value' in match:
            must_conditions.append(FieldCondition(key=key, match=MatchValue(value=match['value'])))
    query_filter = Filter(must=must_conditions) if must_conditions else None
    effective_limit = config.get('limit', top_k)
    search_params = None
    if config.get('hnsw_ef'):
        search_params = SearchParams(hnsw_ef=config['hnsw_ef'])
    result = client.query_points(collection_name=collection, query=query_vector, limit=effective_limit, query_filter=query_filter, score_threshold=config.get('score_threshold'), search_params=search_params, with_payload=True, with_vectors=False)
    latency_ms = (time.perf_counter() - start) * 1000
    points = result.points
    hit_ids = tuple((p.payload.get('es_id', '') for p in points))
    hit_courses = tuple((p.payload.get('course', '') for p in points))
    hit_scores = tuple((float(p.score) if p.score is not None else 0.0 for p in points))
    hit_answers = tuple((p.payload.get('answer', '') for p in points))
    top_answer = points[0].payload.get('answer') if points else None
    return SearchResult(hit_ids=hit_ids, hit_scores=hit_scores, hit_courses=hit_courses, top_answer=top_answer, latency_ms=latency_ms, hit_answers=hit_answers)

def run_hybrid_rrf_retrieval(client, collection: str, query_vector: list, es: 'Elasticsearch', es_index: str, query_text: str, course_filter: str, config: dict, top_k: int) -> SearchResult:
    """Reciprocal Rank Fusion over Qdrant (vector) + Elasticsearch (BM25)."""
    start = time.perf_counter()
    k_rrf = config.get('rrf_k', 60)
    must_conditions = []
    if course_filter:
        must_conditions.append(FieldCondition(key='course', match=MatchValue(value=course_filter)))
    query_filter = Filter(must=must_conditions) if must_conditions else None
    vector_result = client.query_points(collection_name=collection, query=query_vector, limit=top_k, query_filter=query_filter, with_payload=True, with_vectors=False)
    vector_items = [(p.payload.get('es_id', ''), p.payload.get('course', ''), p.payload.get('answer', '')) for p in vector_result.points]
    bm25_result = run_es_retrieval(es=es, index=es_index, query_text=query_text, course_filter=course_filter, config=config, top_k=top_k)
    bm25_items = list(zip(bm25_result.hit_ids, bm25_result.hit_courses))
    rrf_scores: dict[str, float] = {}
    id_to_course: dict[str, str] = {}
    id_to_answer: dict[str, str] = {}
    for rank, (doc_id, course, answer) in enumerate(vector_items, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
        if doc_id not in id_to_course:
            id_to_course[doc_id] = course
        if doc_id not in id_to_answer and answer:
            id_to_answer[doc_id] = answer
    for rank, (doc_id, course) in enumerate(bm25_items, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
        if doc_id not in id_to_course:
            id_to_course[doc_id] = course
    sorted_ids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])[:top_k]
    scores = tuple((rrf_scores[doc_id] for doc_id in sorted_ids))
    courses = tuple((id_to_course.get(doc_id, '') for doc_id in sorted_ids))
    top_answer = None
    if sorted_ids:
        top_answer = id_to_answer.get(sorted_ids[0])
        if not top_answer and bm25_result.hit_ids and (sorted_ids[0] == bm25_result.hit_ids[0]):
            top_answer = bm25_result.top_answer
    latency_ms = (time.perf_counter() - start) * 1000
    return SearchResult(hit_ids=tuple(sorted_ids), hit_scores=scores, hit_courses=courses, top_answer=top_answer, latency_ms=latency_ms)

def run_entity_boosted_retrieval(client, collection: str, query_vector: list, course_filter: str, config: dict, top_k: int, ner_category: Optional[str]=None, ner_primary_entity: Optional[str]=None) -> SearchResult:
    """Vector search with soft entity boosting via Qdrant should clauses."""
    start = time.perf_counter()
    must_conditions = []
    if course_filter:
        must_conditions.append(FieldCondition(key='course', match=MatchValue(value=course_filter)))
    should_conditions = []
    if ner_primary_entity:
        should_conditions.append(FieldCondition(key='ner_primary_entity', match=MatchValue(value=ner_primary_entity)))
    if ner_category and ner_category not in ('OTHER', 'UNKNOWN'):
        should_conditions.append(FieldCondition(key='ner_category', match=MatchValue(value=ner_category)))
    query_filter = Filter(must=must_conditions, should=should_conditions or None) if must_conditions or should_conditions else None
    result = client.query_points(collection_name=collection, query=query_vector, limit=top_k, query_filter=query_filter, with_payload=True, with_vectors=False)
    latency_ms = (time.perf_counter() - start) * 1000
    points = result.points
    hit_ids = tuple((p.payload.get('es_id', '') for p in points))
    hit_courses = tuple((p.payload.get('course', '') for p in points))
    hit_scores = tuple((float(p.score) if p.score is not None else 0.0 for p in points))
    hit_answers = tuple((p.payload.get('answer', '') for p in points))
    top_answer = points[0].payload.get('answer') if points else None
    return SearchResult(hit_ids=hit_ids, hit_scores=hit_scores, hit_courses=hit_courses, hit_answers=hit_answers, top_answer=top_answer, latency_ms=latency_ms)

def run_vector_retrieval_with_reranker(client, collection: str, query_vector: list, query_text: str, course_filter: str, config: dict, top_k: int, reranker_name: Optional[str]=None) -> SearchResult:
    """
    Vector retrieval + Reranking
    """
    from .._benchmark_reranker import evaluate_with_reranker
    import time
    start_total = time.perf_counter()
    fetch_k = max(top_k * 3, 30)
    initial_result = run_vector_retrieval(client=client, collection=collection, query_vector=query_vector, course_filter=course_filter, config=config, top_k=fetch_k)
    candidates = []
    for i, doc_id in enumerate(initial_result.hit_ids):
        candidates.append({'es_id': doc_id, 'payload': {'es_id': doc_id, 'question': '', 'answer': initial_result.hit_answers[i] if hasattr(initial_result, 'hit_answers') else ''}})
    reranked_ids, metrics = evaluate_with_reranker(query=query_text, retrieved_candidates=candidates, reranker_name=reranker_name, top_k=top_k)
    id_to_score = dict(zip(initial_result.hit_ids, initial_result.hit_scores))
    id_to_answer = dict(zip(initial_result.hit_ids, getattr(initial_result, 'hit_answers', [])))
    final_hit_ids = tuple(reranked_ids)
    final_scores = tuple((id_to_score.get(doc_id, 0.0) for doc_id in final_hit_ids))
    final_answers = tuple((id_to_answer.get(doc_id, '') for doc_id in final_hit_ids))
    total_latency_ms = (time.perf_counter() - start_total) * 1000
    return SearchResult(hit_ids=final_hit_ids, hit_scores=final_scores, hit_courses=initial_result.hit_courses[:len(final_hit_ids)], top_answer=final_answers[0] if final_answers else None, latency_ms=total_latency_ms, hit_answers=final_answers)
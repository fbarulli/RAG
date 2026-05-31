"""Hybrid, Boosted, and Reranking retrieval logic."""
import time
import re
from typing import Optional
from .qdrant_retrievers import build_qdrant_filter, parse_qdrant_points, run_vector_retrieval
from .es_retrievers import run_es_retrieval
from ..benchmark_types import SearchResult

def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores: return []
    s_min, s_max = min(scores), max(scores)
    s_range = s_max - s_min
    return [(s - s_min) / s_range if s_range > 0 else 0.0 for s in scores]

def run_entity_category_boosted_retrieval(client, collection, query_vector, course_filter, config, top_k, ner_category=None, ner_primary_entity=None, ner_entities=None, topic=None, section=None, **kwargs) -> SearchResult:
    start = time.perf_counter()
    should = []
    
    if config.get("use_entity", True):
        for _e in (ner_entities or ([ner_primary_entity] if ner_primary_entity else [])):
            should.append({'key': 'ner_primary_entity', 'match': {'value': _e}})
    if config.get("use_category", True) and ner_category and ner_category != "OTHER":
        should.append({'key': 'ner_category', 'match': {'value': ner_category}})
    if config.get("use_topic", False) and topic is not None and topic != -1:
        should.append({'key': 'topic', 'match': {'value': topic}})
        
    if config.get("use_section", False) and section:
        # Normalize section (module-N)
        norm = re.match(r"(module-\d+)", section.lower().strip())
        if norm: should.append({'key': 'section', 'match': {'value': norm.group(1)}})

    query_filter = build_qdrant_filter(course_filter, should_configs=should)
    result = client.query_points(collection_name=collection, query=query_vector, limit=top_k, query_filter=query_filter)
    if not result.points and should:
        fallback_filter = build_qdrant_filter(course_filter)
        result = client.query_points(collection_name=collection, query=query_vector, limit=top_k, query_filter=fallback_filter)
    return parse_qdrant_points(result.points, (time.perf_counter() - start) * 1000)

def run_entity_boosted_retrieval(*args, **kwargs) -> SearchResult:
    """Legacy wrapper for entity_boosted config."""
    # Force disable category/topic for this specific legacy config
    kwargs['config'] = {**kwargs.get('config', {}), 'use_category': False, 'use_topic': False}
    return run_entity_category_boosted_retrieval(*args, **kwargs)

def run_hybrid_rrf_retrieval(client, collection, query_vector, es, es_index, query_text, course_filter, config, top_k) -> SearchResult:
    start = time.perf_counter()
    k_rrf = config.get('rrf_k', 60)
    
    # Vector part
    q_filter = build_qdrant_filter(course_filter)
    v_res = client.query_points(collection_name=collection, query=query_vector, limit=top_k, query_filter=q_filter)
    
    # ES part
    e_res = run_es_retrieval(es, es_index, query_text, course_filter, config, top_k)
    
    # RRF Fusion
    scores = {}
    id_to_data = {}
    for rank, p in enumerate(v_res.points, 1):
        did = p.payload.get('es_id', '')
        scores[did] = scores.get(did, 0.0) + 1.0 / (k_rrf + rank)
        id_to_data[did] = (p.payload.get('course', ''), p.payload.get('answer', ''))
        
    for rank, did in enumerate(e_res.hit_ids, 1):
        scores[did] = scores.get(did, 0.0) + 1.0 / (k_rrf + rank)
        if did not in id_to_data:
            id_to_data[did] = (e_res.hit_courses[rank-1] if rank <= len(e_res.hit_courses) else '', '')

    sorted_ids = sorted(scores, key=lambda x: -scores[x])[:top_k]
    
    return SearchResult(
        hit_ids=tuple(sorted_ids),
        hit_scores=tuple(scores[did] for did in sorted_ids),
        hit_courses=tuple(id_to_data[did][0] for did in sorted_ids),
        top_answer=id_to_data[sorted_ids[0]][1] if sorted_ids else None,
        latency_ms=(time.perf_counter() - start) * 1000,
        hit_answers=tuple(id_to_data[did][1] for did in sorted_ids)
    )

def run_hybrid_dbsf_retrieval(client, collection, query_vector, es, es_index, query_text, course_filter, config, top_k) -> SearchResult:
    start = time.perf_counter()
    v_weight = config.get("vector_weight", 0.7)
    b_weight = config.get("bm25_weight", 0.3)
    
    q_filter = build_qdrant_filter(course_filter)
    v_res = client.query_points(collection_name=collection, query=query_vector, limit=top_k*2, query_filter=q_filter)
    e_res = run_es_retrieval(es, es_index, query_text, course_filter, config, top_k*2)
    
    v_norm = _normalize_scores([float(p.score) for p in v_res.points])
    b_norm = _normalize_scores(list(e_res.hit_scores))
    
    combined = {}
    id_info = {}
    for p, score in zip(v_res.points, v_norm):
        did = p.payload.get('es_id', '')
        combined[did] = combined.get(did, 0.0) + v_weight * score
        id_info[did] = (p.payload.get('course', ''), p.payload.get('answer', ''))
        
    for did, score in zip(e_res.hit_ids, b_norm):
        combined[did] = combined.get(did, 0.0) + b_weight * score
        if did not in id_info: id_info[did] = (e_res.hit_courses[rank-1] if rank <= len(e_res.hit_courses) else '', '')
        
    sorted_ids = sorted(combined, key=lambda x: -combined[x])[:top_k]
    return SearchResult(
        hit_ids=tuple(sorted_ids),
        hit_scores=tuple(combined[did] for did in sorted_ids),
        hit_courses=tuple(id_info[did][0] for did in sorted_ids),
        top_answer=id_info[sorted_ids[0]][1] if sorted_ids else None,
        latency_ms=(time.perf_counter() - start) * 1000,
        hit_answers=tuple(id_info[did][1] for did in sorted_ids)
    )

def run_vector_retrieval_with_reranker(client, collection, query_vector, query_text, course_filter, config, top_k, reranker_name=None) -> SearchResult:
    from ..benchmark_reranker import evaluate_with_reranker
    start = time.perf_counter()
    
    # Initial fetch (larger k for reranking)
    initial = run_vector_retrieval(client, collection, query_vector, course_filter, config, top_k * 3)
    
    candidates = [{'es_id': hit_id, 'question': query_text, 'answer': ans} 
                  for hit_id, ans in zip(initial.hit_ids, initial.hit_answers)]
    
    reranked_ids, rerank_metrics = evaluate_with_reranker(
        query=query_text, retrieved_candidates=candidates, reranker_name=reranker_name, top_k=top_k
    )
    
    # Map reranked IDs back to scores/answers
    id_map = {did: (s, ans) for did, s, ans in zip(initial.hit_ids, initial.hit_scores, initial.hit_answers)}
    
    return SearchResult(
        hit_ids=tuple(reranked_ids),
        hit_scores=tuple(id_map[did][0] for did in reranked_ids),
        hit_courses=initial.hit_courses[:len(reranked_ids)],
        top_answer=id_map[reranked_ids[0]][1] if reranked_ids else None,
        latency_ms=(time.perf_counter() - start) * 1000,
        hit_answers=tuple(id_map[did][1] for did in reranked_ids),
        reranker_latency_ms=rerank_metrics.get('reranker_latency_ms', 0.0)
    )


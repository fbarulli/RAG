"""
Public Functions for Retrieval Metrics Evaluation and Database Query Execution:

def check_code_integrity(text: str) -> float:
    Check if code blocks in text are complete and well-formed.
    I/O: text (str) -> float

def compute_latency_percentiles(latencies: list[float]) -> dict[str, float]:
    Compute p50, p95, p99 latency using proper percentile calculation.
    I/O: latencies (list[float]) -> dict[str, float]

def safe_mean(values: list[float]) -> float:
    Compute mean, returning 0.0 for empty lists.
    I/O: values (list[float]) -> float

def compute_hit_rate(hits: tuple[str, ...], expected_id: str, k: int) -> bool:
    Check if expected_id appears in top-k results.
    I/O: hits (tuple[str, ...]), expected_id (str), k (int) -> bool

def compute_reciprocal_rank(hits: tuple[str, ...], expected_id: str) -> float:
    Compute 1/rank if found, else 0.
    I/O: hits (tuple[str, ...]), expected_id (str) -> float

def compute_ndcg_at_k(hits: tuple[str, ...], expected_id: str, k: int) -> float:
    Compute NDCG@k for binary relevance (1 if correct, 0 otherwise).
    I/O: hits (tuple[str, ...]), expected_id (str), k (int) -> float

def run_hybrid_rrf_query(client: Any, collection: str, query_vector: list, es: Elasticsearch, es_index: str, query_text: str, course_filter: str, config: dict, top_k: int) -> tuple[tuple[str, ...], Optional[str], tuple[float, ...], float]:
    Execute a hybrid Reciprocal Rank Fusion (RRF) search combining dense vector hits with Elasticsearch BM25 results.
    I/O: client (Any), collection (str), query_vector (list), es (Elasticsearch), es_index (str), query_text (str), course_filter (str), config (dict), top_k (int) -> tuple[tuple[str, ...], Optional[str], tuple[float, ...], float]

def run_es_retrieval_query(es: Elasticsearch, index: str, query_text: str, course_filter: str, config: dict, top_k: int) -> tuple[tuple[str, ...], Optional[str], tuple[float, ...], float]:
    Execute a text-search query against Elasticsearch applying query fields and term boost configurations.
    I/O: es (Elasticsearch), index (str), query_text (str), course_filter (str), config (dict), top_k (int) -> tuple[tuple[str, ...], Optional[str], tuple[float, ...], float]

def run_retrieval_query(client: Any, collection: str, query_vector: list, course_filter: str, config: dict, top_k: int) -> tuple[tuple[str, ...], Optional[str], tuple[float, ...], float]:
    Execute a single retrieval query against Qdrant, applying config options.
    I/O: client (Any), collection (str), query_vector (list), course_filter (str), config (dict), top_k (int) -> tuple[tuple[str, ...], Optional[str], tuple[float, ...], float]

rag_pipeline/p04_ingestion/_benchmark_metrics.py
=====================
Compute retrieval metrics for benchmark evaluation.

Single responsibility: metric computation logic.
No I/O, no reporting, no model loading, no heuristic guessing.
Stratification relies exclusively on factual topic assignments.

Functions:
    evaluate_config(...) -> list[QueryResult]
    aggregate_metrics(...) -> MetricSummary
    aggregate_metrics_by_topic(...) -> list[MetricSummary]
"""
import math
import re
import time
from collections import defaultdict
from statistics import mean, quantiles, stdev, mean
from typing import Optional
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams
from elasticsearch import Elasticsearch
from ._benchmark_types import MetricSummary, QueryResult
from ._benchmark_reranker import evaluate_with_reranker

def check_code_integrity(text: str) -> float:
    """
    Check if code blocks in text are complete and well-formed.
    
    Returns 1.0 if no code blocks present (neutral), or ratio of complete blocks.
    """
    blocks = re.findall('```(?:\\w+)?\\s*(.*?)```', text, re.DOTALL)
    if not blocks:
        return 1.0
    complete = sum((1 for b in blocks if b.strip() and '\n' in b))
    return complete / len(blocks)

def compute_latency_percentiles(latencies: list[float]) -> dict[str, float]:
    """Compute p50, p95, p99 latency using proper percentile calculation."""
    if not latencies:
        return {'p50': 0.0, 'p95': 0.0, 'p99': 0.0}
    try:
        if len(latencies) >= 2:
            q = quantiles(latencies, n=100)
            return {'p50': q[49], 'p95': q[94], 'p99': q[98]}
        val = latencies[0]
        return {'p50': val, 'p95': val, 'p99': val}
    except Exception:
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        return {'p50': sorted_lat[math.ceil(n * 0.5) - 1], 'p95': sorted_lat[math.ceil(n * 0.95) - 1], 'p99': sorted_lat[math.ceil(n * 0.99) - 1]}

def safe_mean(values: list[float]) -> float:
    """Compute mean, returning 0.0 for empty lists."""
    return mean(values) if values else 0.0

def compute_hit_rate(hits: tuple[str, ...], expected_id: str, k: int) -> bool:
    """Check if expected_id appears in top-k results."""
    return expected_id in hits[:k]

def compute_reciprocal_rank(hits: tuple[str, ...], expected_id: str) -> float:
    """Compute 1/rank if found, else 0."""
    if expected_id in hits:
        return 1.0 / (hits.index(expected_id) + 1)
    return 0.0

def compute_ndcg_at_k(hits: tuple[str, ...], expected_id: str, k: int) -> float:
    """Compute NDCG@k for binary relevance (1 if correct, 0 otherwise)."""
    if expected_id not in hits:
        return 0.0
    rank = hits.index(expected_id) + 1
    if rank > k:
        return 0.0
    dcg = 1.0 / math.log2(rank + 1)
    idcg = 1.0
    return dcg / idcg

def run_hybrid_rrf_query(client, collection: str, query_vector: list, es: 'Elasticsearch', es_index: str, query_text: str, course_filter: str, config: dict, top_k: int) -> tuple[tuple[str, ...], Optional[str], tuple[float, ...], float]:
    start = time.perf_counter()
    k = config.get('rrf_k', 60)
    must_conditions = []
    if course_filter:
        must_conditions.append(FieldCondition(key='course', match=MatchValue(value=course_filter)))
    query_filter = Filter(must=must_conditions) if must_conditions else None
    vector_result = client.query_points(collection_name=collection, query=query_vector, limit=top_k, query_filter=query_filter, with_payload=True, with_vectors=False)
    vector_ids = [p.payload.get('es_id', '') for p in vector_result.points]
    bm25_result = run_es_retrieval_query(es=es, index=es_index, query_text=query_text, course_filter=course_filter, config=config, top_k=top_k)
    bm25_ids = list(bm25_result[0])
    rrf_scores: dict[str, float] = {}
    for rank, doc_id in enumerate(vector_ids, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, doc_id in enumerate(bm25_ids, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    sorted_ids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])[:top_k]
    scores = tuple((rrf_scores[doc_id] for doc_id in sorted_ids))
    id_to_answer = {p.payload.get('es_id', ''): p.payload.get('answer') for p in vector_result.points}
    top_answer = id_to_answer.get(sorted_ids[0]) if sorted_ids else None
    latency_ms = (time.perf_counter() - start) * 1000
    return (tuple(sorted_ids), top_answer, scores, latency_ms)

def run_es_retrieval_query(es: Elasticsearch, index: str, query_text: str, course_filter: str, config: dict, top_k: int) -> tuple[tuple[str, ...], Optional[str], tuple[float, ...], float]:
    start = time.perf_counter()
    boost_q = config.get('boost_question', 1.0)
    boost_t = config.get('boost_text', 1.0)
    must = []
    if boost_q > 0 and boost_t > 0:
        must.append({'multi_match': {'query': query_text, 'fields': [f'question^{boost_q}', f'answer^{boost_t}']}})
    elif boost_q > 0:
        must.append({'match': {'question': {'query': query_text, 'boost': boost_q}}})
    else:
        must.append({'match': {'answer': {'query': query_text, 'boost': boost_t}}})
    body = {'query': {'bool': {'must': must, 'filter': [{'term': {'course': course_filter}}] if course_filter else []}}, 'size': top_k}
    resp = es.search(index=index, body=body)
    latency_ms = (time.perf_counter() - start) * 1000
    hits = resp['hits']['hits']
    hit_ids = tuple((h['_source'].get('es_id', '') for h in hits))
    scores = tuple((float(h['_score']) for h in hits))
    top_answer = hits[0]['_source'].get('answer') if hits else None
    return (hit_ids, top_answer, scores, latency_ms)

def run_retrieval_query(client, collection: str, query_vector: list, course_filter: str, config: dict, top_k: int) -> tuple[tuple[str, ...], Optional[str], tuple[float, ...], float]:
    """
    Execute a single retrieval query against Qdrant, applying config options.
    
    Returns:
        hit_ids: Tuple of document IDs
        top_answer: The full answer text of the #1 result (or None)
        scores: Tuple of relevance scores
        latency_ms: Execution time in milliseconds
    """
    start = time.perf_counter()
    must_conditions = []
    if course_filter:
        must_conditions.append(FieldCondition(key='course', match=MatchValue(value=course_filter)))
    if config.get('filters'):
        for cond in config['filters'].get('must', []):
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
    scores = tuple((float(p.score) if p.score is not None else 0.0 for p in points))
    top_answer = points[0].payload.get('answer', None) if points else None
    return (hit_ids, top_answer, scores, latency_ms)

def run_entity_boosted_query(client, collection: str, query_vector: list, course_filter: str, config: dict, top_k: int, ner_category: str | None=None, ner_primary_entity: str | None=None) -> tuple[tuple[str, ...], Optional[str], tuple[float, ...], float]:
    """
    Vector search with soft entity boosting via should clauses.
    Falls back gracefully — never returns 0 results due to entity mismatch.
    """
    start = time.perf_counter()
    must_conditions = []
    if course_filter:
        must_conditions.append(FieldCondition(key='course', match=MatchValue(value=course_filter)))
    should_conditions = []
    if ner_primary_entity:
        should_conditions.append(FieldCondition(key='ner_primary_entity', match=MatchValue(value=ner_primary_entity)))
    if ner_category and ner_category not in ('OTHER', 'UNKNOWN'):
        should_conditions.append(FieldCondition(key='ner_category', match=MatchValue(value=ner_category)))
    query_filter = Filter(must=must_conditions, should=should_conditions if should_conditions else None) if must_conditions or should_conditions else None
    result = client.query_points(collection_name=collection, query=query_vector, limit=top_k, query_filter=query_filter, with_payload=True, with_vectors=False)
    latency_ms = (time.perf_counter() - start) * 1000
    points = result.points
    hit_ids = tuple((p.payload.get('es_id', '') for p in points))
    scores = tuple((float(p.score) if p.score is not None else 0.0 for p in points))
    top_answer = points[0].payload.get('answer', None) if points else None
    return (hit_ids, top_answer, scores, latency_ms)

def evaluate_config(client, collection: str, model, test_set: list[dict], topic_map: dict, config: dict, top_k: int, es: 'Elasticsearch | None'=None, es_index: str='faqs', encode_batch_size: int=32) -> list[QueryResult]:
    """
    Evaluate a single config/model combination against the test set.
    Uses factual topic/subtopic assignments from topic_map.
    Applies reranking if config specifies reranker=true and reranker_name.
    """
    use_reranker = config.get('reranker', False)
    reranker_name = config.get('reranker_name') if use_reranker else None
    retrieval_k = top_k * 4 if use_reranker else top_k
    results = []
    for test in test_set:
        query = test['query']
        expected_id = test['expected_id']
        course = test['course']
        ref_answer = test['answer']
        topic_info = topic_map.get(expected_id, {})
        topic = topic_info.get('topic', -1)
        subtopic = topic_info.get('subtopic')
        ner_category = topic_info.get('ner_category')
        ner_primary_entity = topic_info.get('ner_primary_entity')
        search_type = config.get('search_type', 'vector')
        if search_type == 'bm25' and es is not None:
            hit_ids, top_answer, scores, latency_ms = run_es_retrieval_query(es=es, index=es_index, query_text=query, course_filter=course, config=config, top_k=retrieval_k)
        elif search_type == 'hybrid_rrf' and es is not None:
            query_vector = model.encode(query, convert_to_numpy=True).tolist()
            hit_ids, top_answer, scores, latency_ms = run_hybrid_rrf_query(client=client, collection=collection, query_vector=query_vector, es=es, es_index=es_index, query_text=query, course_filter=course, config=config, top_k=retrieval_k)
        elif search_type == 'entity_boosted':
            query_vector = model.encode(query, convert_to_numpy=True).tolist()
            hit_ids, top_answer, scores, latency_ms = run_entity_boosted_query(client=client, collection=collection, query_vector=query_vector, course_filter=course, config=config, top_k=retrieval_k, ner_category=ner_category, ner_primary_entity=ner_primary_entity)
        else:
            query_vector = model.encode(query, convert_to_numpy=True).tolist()
            hit_ids, top_answer, scores, latency_ms = run_retrieval_query(client=client, collection=collection, query_vector=query_vector, course_filter=course, config=config, top_k=retrieval_k)
        reranker_latency_ms = 0.0
        if use_reranker and reranker_name and hit_ids:
            candidates = [{'es_id': id_} for id_ in hit_ids]
            hit_ids, rerank_metrics = evaluate_with_reranker(query=query, retrieved_candidates=candidates, reranker_name=reranker_name, top_k=top_k)
            hit_ids = tuple(hit_ids)
            reranker_latency_ms = rerank_metrics.get('reranker_latency_ms', 0.0)
            scores = tuple((0.0 for _ in hit_ids))
        code_int_ref = check_code_integrity(ref_answer)
        code_int_ret = check_code_integrity(top_answer) if top_answer else None
        results.append(QueryResult(query_id=test['query_id'], query_text=query, expected_id=expected_id, course=course, topic=topic, subtopic=subtopic, query_type=test.get('query_type', 'unknown'), hit_ids=hit_ids, hit_scores=scores, latency_ms=latency_ms, reranker_latency_ms=reranker_latency_ms, code_integrity_ref=code_int_ref, code_integrity_retrieved=code_int_ret))
    return results

def aggregate_metrics(results: list[QueryResult], config_name: str, model_name: str, topic: Optional[int]=None, subtopic: Optional[int]=None) -> MetricSummary:
    """Aggregate per-query results into summary metrics with all new diagnostics."""
    if not results:
        return MetricSummary(config_name=config_name, model_name=model_name, topic=topic, subtopic=subtopic, num_queries=0)
    n = len(results)
    hit_1 = sum((1 for r in results if compute_hit_rate(r.hit_ids, r.expected_id, 1)))
    hit_3 = sum((1 for r in results if compute_hit_rate(r.hit_ids, r.expected_id, 3)))
    hit_5 = sum((1 for r in results if compute_hit_rate(r.hit_ids, r.expected_id, 5)))
    hit_10 = sum((1 for r in results if compute_hit_rate(r.hit_ids, r.expected_id, 10)))
    mrr = safe_mean([compute_reciprocal_rank(r.hit_ids, r.expected_id) for r in results])
    ndcg = safe_mean([compute_ndcg_at_k(r.hit_ids, r.expected_id, 10) for r in results])
    latencies = [r.latency_ms for r in results]
    lat_pcts = compute_latency_percentiles(latencies)
    code_int_ref = safe_mean([r.code_integrity_ref for r in results])
    code_int_ret_vals = [r.code_integrity_retrieved for r in results if r.code_integrity_retrieved is not None]
    code_int_ret = safe_mean(code_int_ret_vals) if code_int_ret_vals else None
    ranks = []
    failures = []
    failure_sims = []
    cross_course_errors = 0
    for r in results:
        if r.expected_id in r.hit_ids:
            rank = r.hit_ids.index(r.expected_id) + 1
            ranks.append(rank)
        else:
            ranks.append(11)
            failures.append(r)
            if r.hit_scores and len(r.hit_scores) > 0:
                failure_sims.append(r.hit_scores[0])
        if r.hit_ids:
            pass
    cross_course_contamination = cross_course_errors / n if n > 0 else 0.0
    rank_std = stdev(ranks) if len(ranks) >= 2 else 0.0
    failure_count = len(failures)
    avg_failure_similarity = mean(failure_sims) if failure_sims else None
    return MetricSummary(config_name=config_name, model_name=model_name, topic=topic, subtopic=subtopic, num_queries=n, hit_rate_1=hit_1 / n, hit_rate_3=hit_3 / n, hit_rate_5=hit_5 / n, hit_rate_10=hit_10 / n, mrr=mrr, ndcg_10=ndcg, latency_p50=lat_pcts['p50'], latency_p95=lat_pcts['p95'], latency_p99=lat_pcts.get('p99', 0.0), avg_code_integrity_ref=code_int_ref, avg_code_integrity_retrieved=code_int_ret, cross_course_contamination=cross_course_contamination, rank_std=rank_std, failure_count=failure_count, avg_failure_similarity=avg_failure_similarity)
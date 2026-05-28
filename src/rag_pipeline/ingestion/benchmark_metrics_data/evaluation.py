"""
rag_pipeline/p04_ingestion/_benchmark_metrics/evaluation.py
Evaluation orchestration - ties retrievers to test data.
"""
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional, TypedDict, TYPE_CHECKING
from ..benchmark_types import QueryResult
from .core import check_code_integrity
from .retrievers import run_es_retrieval, run_vector_retrieval, run_vector_retrieval_with_reranker, run_hybrid_rrf_retrieval, run_entity_boosted_retrieval, run_hybrid_dbsf_retrieval, run_entity_category_boosted_retrieval
if TYPE_CHECKING:
    from elasticsearch import Elasticsearch
logger = logging.getLogger(__name__)

class QueryContext(TypedDict):
    query: str
    expected_id: str
    course: str
    topic: Optional[int]
    subtopic: Optional[int]
    ner_category: Optional[str]
    ner_primary_entity: Optional[str]
    query_vector: Optional[list]

@dataclass(frozen=True)
class RetrievalConfig:
    search_type: str
    use_reranker: bool
    reranker_name: Optional[str]
    retrieval_k: int
    top_k: int
    client: Any
    collection: str
    config: dict
    es: Optional[Any]
    es_index: str

def _encode_queries(model, test_set: list[dict], encode_batch_size: int) -> list[list[float]]:
    queries = [test['query'] for test in test_set]
    vectors = model.encode(queries, batch_size=encode_batch_size, convert_to_numpy=True, show_progress_bar=True)
    return vectors.tolist()

def _build_integrity_cache(test_set: list[dict]) -> dict[str, float]:
    """
    Pre-compute code integrity scores for all reference answers.
    Multiple queries can share the same expected_id/answer — computing
    once avoids redundant work in the per-query loop.
    """
    return {test['expected_id']: check_code_integrity(test['answer']) for test in test_set}


def _load_retrieval_configs() -> dict:
    """Load retrieval_configs.json via Paths — single source of truth."""
    from rag_pipeline.core.paths import Paths
    path = Paths.retrieval_configs()
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _search_types_requiring_es() -> frozenset:
    """Search types that require an Elasticsearch connection, derived from retrieval_configs.json."""
    return frozenset(
        v['search_type'] for v in _load_retrieval_configs().values()
        if 'hybrid' in v.get('search_type', '') or v.get('search_type') == 'bm25'
    )


def _search_types_requiring_vector() -> frozenset:
    """Search types that require an embedding model, derived from retrieval_configs.json."""
    return frozenset(
        v['search_type'] for v in _load_retrieval_configs().values()
        if v.get('search_type') != 'bm25'
    )


def _validate_config(rc: RetrievalConfig, model) -> None:
    """Raise early if required dependencies are missing for the chosen search type."""
    es_types = _search_types_requiring_es()
    vec_types = _search_types_requiring_vector()
    if rc.search_type in es_types and rc.es is None:
        raise ValueError(f"Search type '{rc.search_type}' requires Elasticsearch. ES-requiring types: {sorted(es_types)}")
    if rc.search_type in vec_types and model is None:
        raise ValueError(f"Search type '{rc.search_type}' requires an embedding model. Vector types: {sorted(vec_types)}")
    if rc.use_reranker and rc.search_type == 'bm25' and model is None:
        raise ValueError("Reranking with bm25 requires an embedding model for the reranker. Provide 'model' parameter.")

def _run_retrieval(rc: RetrievalConfig, query: str, query_vector: Optional[list], course: str, ner_category: Optional[str], ner_primary_entity: Optional[str], topic: Optional[int] = None, section: Optional[str] = None, query_type: Optional[str] = None):
    """Dispatch to the correct retriever based on search_type."""
    if rc.search_type == 'bm25' and rc.es is not None:
        return run_es_retrieval(es=rc.es, index=rc.es_index, query_text=query, course_filter=course, config=rc.config, top_k=rc.retrieval_k)
    if query_vector is None:
        raise ValueError(f"query_vector is None for search_type={rc.search_type!r}")
    if rc.search_type == 'hybrid_rrf' and rc.es is not None:
        return run_hybrid_rrf_retrieval(client=rc.client, collection=rc.collection, query_vector=query_vector, es=rc.es, es_index=rc.es_index, query_text=query, course_filter=course, config=rc.config, top_k=rc.retrieval_k)
    if rc.search_type == 'hybrid_dbsf' and rc.es is not None:
        return run_hybrid_dbsf_retrieval(client=rc.client, collection=rc.collection, query_vector=query_vector, es=rc.es, es_index=rc.es_index, query_text=query, course_filter=course, config=rc.config, top_k=rc.retrieval_k)
    if rc.search_type == 'entity_boosted':
        return run_entity_boosted_retrieval(client=rc.client, collection=rc.collection, query_vector=query_vector, course_filter=course, config=rc.config, top_k=rc.retrieval_k, ner_category=ner_category, ner_primary_entity=ner_primary_entity, query_type=query_type)
    if rc.search_type == 'entity_category_boosted':
        return run_entity_category_boosted_retrieval(client=rc.client, collection=rc.collection, query_vector=query_vector, course_filter=course, config=rc.config, top_k=rc.retrieval_k, ner_category=ner_category, ner_primary_entity=ner_primary_entity, topic=topic, section=section)
    if rc.use_reranker and rc.reranker_name:
        return run_vector_retrieval_with_reranker(client=rc.client, collection=rc.collection, query_vector=query_vector, query_text=query, course_filter=course, config=rc.config, top_k=rc.top_k, reranker_name=rc.reranker_name)
    return run_vector_retrieval(client=rc.client, collection=rc.collection, query_vector=query_vector, course_filter=course, config=rc.config, top_k=rc.top_k)


def _apply_reranking(search_result, rc: RetrievalConfig, query: str):
    """
    Apply reranking post-retrieval for non-vector search types.
    Vector search handles reranking internally via run_vector_retrieval_with_reranker.
    Returns (hit_ids, hit_scores, reranker_latency_ms).

    Post-rerank scores use descending reciprocal rank (1/rank) as a proxy
    so rank-weighted metrics like NDCG remain meaningful.
    """
    from ..benchmark_reranker import evaluate_with_reranker
    hit_ids = search_result.hit_ids
    hit_scores = search_result.hit_scores
    reranker_latency_ms = search_result.reranker_latency_ms
    if not (rc.use_reranker and rc.reranker_name and (rc.search_type != 'vector')):
        return (hit_ids, hit_scores, reranker_latency_ms)
    if not hit_ids:
        logger.warning(f"Reranker '{rc.reranker_name}' skipped — retrieval returned no candidates for search_type='{rc.search_type}'.")
        return (hit_ids, hit_scores, reranker_latency_ms)
    hit_answers = getattr(search_result, 'hit_answers', None) or [''] * len(hit_ids)
    candidates = [{'es_id': hit_ids[i], 'question': query, 'answer': hit_answers[i]} for i in range(len(hit_ids))]
    reranked_ids, rerank_metrics = evaluate_with_reranker(query=query, retrieved_candidates=candidates, reranker_name=rc.reranker_name, top_k=rc.top_k)
    hit_ids = tuple(reranked_ids)
    hit_scores = tuple((1.0 / (rank + 1) for rank in range(len(hit_ids))))
    reranker_latency_ms = rerank_metrics.get('reranker_latency_ms', 0.0)
    return (hit_ids, hit_scores, reranker_latency_ms)

def _build_query_context(idx: int, test: dict, topic_map: dict, query_vectors: Optional[list]) -> QueryContext:
    """Extract and return all per-query context fields."""
    expected_id = test['expected_id']
    topic_info = topic_map.get(expected_id, {})
    return {'query': test['query'], 'expected_id': expected_id, 'course': test['course'], 'topic': topic_info.get('topic'), 'subtopic': topic_info.get('subtopic'), 'ner_category': topic_info.get('ner_category'), 'ner_primary_entity': topic_info.get('ner_primary_entity'), 'section': topic_info.get('section'), 'query_vector': query_vectors[idx] if query_vectors is not None else None}


def _build_query_result(test: dict, ctx: QueryContext, search_result, hit_ids: tuple, hit_scores: tuple, reranker_latency_ms: float, integrity_cache: dict) -> QueryResult:
    """Assemble the final QueryResult from retrieval outputs."""
    return QueryResult(query_id=test['query_id'], query_text=ctx['query'], expected_id=ctx['expected_id'], course=ctx['course'], topic=ctx['topic'], subtopic=ctx['subtopic'], query_type=test.get('query_type', 'unknown'), hit_ids=hit_ids, hit_scores=hit_scores, hit_courses=search_result.hit_courses, latency_ms=search_result.latency_ms, reranker_latency_ms=reranker_latency_ms, code_integrity_ref=float(integrity_cache.get(ctx['expected_id']) or 0.0), code_integrity_retrieved=check_code_integrity(search_result.top_answer) if search_result.top_answer else None)

def _evaluate_single(idx: int, test: dict, topic_map: dict, query_vectors: Optional[list], integrity_cache: dict[str, float], rc: RetrievalConfig) -> QueryResult:
    """
    Evaluate a single test item: retrieve, optionally rerank, build QueryResult.
    Extracted for unit testability and to keep evaluate_config readable.
    """
    ctx = _build_query_context(idx, test, topic_map, query_vectors)
    search_result, hit_ids, hit_scores, reranker_latency_ms = _retrieve_and_rerank(ctx, rc)
    return _build_query_result(test, ctx, search_result, hit_ids, hit_scores, reranker_latency_ms, integrity_cache)

def _run_evaluation_loop(test_set: list[dict], topic_map: dict, query_vectors: Optional[list], integrity_cache: dict, rc: RetrievalConfig) -> list[QueryResult]:
    results: list[QueryResult] = []
    total = len(test_set)
    for idx, test in enumerate(test_set):
        if idx % 10 == 0:
            logger.info(f'Progress: {idx}/{total} queries evaluated')
        try:
            ctx = _build_query_context(idx, test, topic_map, query_vectors)
            search_result = _run_retrieval(rc=rc, query=ctx['query'], query_vector=ctx['query_vector'], course=ctx['course'], ner_category=ctx['ner_category'], ner_primary_entity=ctx['ner_primary_entity'], topic=ctx.get('topic'), section=ctx.get('section'), query_type=test.get("query_type", "unknown"))
            hit_ids, hit_scores, reranker_latency_ms = _apply_reranking(search_result=search_result, rc=rc, query=ctx['query'])
            results.append(_build_query_result(test, ctx, search_result, hit_ids, hit_scores, reranker_latency_ms, integrity_cache))
        except Exception as e:
            logger.error(f"Query '{test.get('query_id', idx)}' failed: {e}", exc_info=True)
    logger.info(f'Evaluation complete — {len(results)}/{total} queries succeeded.')
    return results

def evaluate_config(client, collection: str, model, test_set: list[dict], topic_map: dict, config: dict, top_k: int, es: 'Elasticsearch | None'=None, es_index: str='faqs', encode_batch_size: int=32) -> list[QueryResult]:
    """
    Evaluate one (config, model) pair against the full test set.
    Applies reranking post-retrieval for any search_type when config
    specifies reranker=true and reranker_name.
    """
    use_reranker = config.get('reranker', False)
    reranker_name = config.get('reranker_name') if use_reranker else None
    search_type = config.get('search_type', 'vector')
    rc = RetrievalConfig(search_type=search_type, use_reranker=use_reranker, reranker_name=reranker_name, retrieval_k=top_k * 4 if use_reranker else top_k, top_k=top_k, client=client, collection=collection, config=config, es=es, es_index=es_index)
    _validate_config(rc, model)
    query_vectors = None
    if search_type in _search_types_requiring_vector():
        queries = [test['query'] for test in test_set]
        query_vectors = _encode_queries(model, test_set, encode_batch_size)
    integrity_cache = _build_integrity_cache(test_set)
    if query_vectors is not None:
        logger.info(f'Vector check — dim={len(query_vectors[0])}, sample={query_vectors[0][:3]}')
        from ..benchmark_metrics_data.retrievers import run_entity_boosted_retrieval
        test_result = run_entity_boosted_retrieval(client=rc.client, collection=rc.collection, query_vector=query_vectors[0], course_filter=test_set[0]['course'], config=rc.config, top_k=5, ner_category=None, ner_primary_entity=None)
        logger.info(f'Test retrieval OK — hits={len(test_result.hit_ids)}')
    logger.info(f'Integrity cache built — {len(integrity_cache)} entries.')
    return _run_evaluation_loop(test_set=test_set, topic_map=topic_map, query_vectors=query_vectors, integrity_cache=integrity_cache, rc=rc)

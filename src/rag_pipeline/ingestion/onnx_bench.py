"""
rag_pipeline/p04_ingestion/_onnx_bench.py

Core evaluation processing functions for the ONNX Cross-Encoder matrix.
RESPONSIBILITY: Manages individual test query iteration execution flows.
"""
import logging
import hashlib
import json
from pathlib import Path
import traceback
import time
from typing import Any, Dict, List, Optional, Tuple
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from .onnx_bench_config import extract_active_environment
from .reranker_runner import RerankerRunner
from .benchmark_metrics_data.evaluation import run_entity_boosted_retrieval
from .benchmark_metrics_data.aggregation import aggregate_metrics
from .benchmark_types import QueryResult
logger = logging.getLogger(__name__)

def _extract_query_text(query_item: Dict[str, Any]) -> str:
    """RESPONSIBILITY: Extracts the raw text query string with appropriate field fallbacks."""
    return query_item.get('question') or query_item.get('query', '')

def _generate_query_embedding(query_text: str, embedding_model: SentenceTransformer) -> List[float]:
    """RESPONSIBILITY: Transforms raw string text dynamically into a vector array using the bi-encoder."""
    try:
        return embedding_model.encode(query_text, convert_to_numpy=True).tolist()
    except Exception as e:
        logger.error('Failed to dynamically encode query text: %s', e)
        return []

def _execute_vector_search(client: QdrantClient, collection: str, query_vector: List[float], config: Dict[str, Any]) -> Optional[Any]:
    """RESPONSIBILITY: Executes the network search pass against the Qdrant backend."""
    return run_entity_boosted_retrieval(client=client, collection=collection, query_vector=query_vector, course_filter=str(config.get('course_filter')) if config.get('course_filter') else None, config=config, top_k=40, ner_category=None, ner_primary_entity=None)

def _extract_hit_ids_safely(retrieval_result: Any) -> List[Any]:
    """RESPONSIBILITY: Safe schema interface extractor handling both custom objects and raw dict formats."""
    if not retrieval_result:
        return []
    if isinstance(retrieval_result, dict):
        return retrieval_result.get('hit_ids', [])
    return getattr(retrieval_result, 'hit_ids', None) or getattr(retrieval_result, 'ids', [])

def _retrieve_vector_candidates(client: QdrantClient, collection: str, query_item: Dict[str, Any], config: Dict[str, Any], embedding_model: SentenceTransformer) -> List[Dict[str, Any]]:
    """RESPONSIBILITY: Retrieves candidates directly from SearchResult — no secondary Qdrant lookup needed."""
    query_text = _extract_query_text(query_item)
    if not query_text:
        return []
    query_vector = _generate_query_embedding(query_text, embedding_model)
    if not query_vector:
        return []
    retrieval_result = _execute_vector_search(client, collection, query_vector, config)
    if not retrieval_result:
        return []
    hit_ids = getattr(retrieval_result, 'hit_ids', ())
    hit_answers = getattr(retrieval_result, 'hit_answers', ())
    hit_courses = getattr(retrieval_result, 'hit_courses', ())
    hit_scores = getattr(retrieval_result, 'hit_scores', ())
    if not hit_ids:
        return []
    return [
        {
            'es_id': hit_ids[i],
            'answer': hit_answers[i] if i < len(hit_answers) else '',
            'course': hit_courses[i] if i < len(hit_courses) else '',
            'score': hit_scores[i] if i < len(hit_scores) else 0.0,
        }
        for i in range(len(hit_ids))
    ]

def _score_and_sort_candidates(encoder: Any, query_text: str, candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    """RESPONSIBILITY: Scores candidates via RerankerRunner and sorts by score descending."""
    documents = [c["answer"] for c in candidates]
    t0 = time.perf_counter()
    doc_score_pairs = encoder.rerank(query=query_text, documents=documents)
    latency_ms = (time.perf_counter() - t0) * 1000
    score_map = {doc: score for doc, score in doc_score_pairs}
    for c in candidates:
        c["score"] = float(score_map.get(c["answer"], 0.0))
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return (candidates, latency_ms)

def _evaluate_single_query(client: QdrantClient, query_item: Dict[str, Any], idx: int, collection: str, config: Dict[str, Any], encoder: Any, embedding_model: SentenceTransformer, topic_map: Optional[Dict[str, Any]]=None) -> Optional[QueryResult]:
    """RESPONSIBILITY: Orchestrates a single query through retrieval and ranking evaluation steps."""
    query_text = _extract_query_text(query_item)
    candidates = _retrieve_vector_candidates(client, collection, query_item, config, embedding_model)
    if not candidates:
        return None
    reranked_hits, latency_ms = _score_and_sort_candidates(encoder, query_text, candidates)
    hit_ids = tuple(c.get('es_id', '') or c.get('payload', {}).get('es_id', '') for c in reranked_hits)
    hit_scores = tuple(float(c.get('score', 0.0)) for c in reranked_hits)
    hit_courses = tuple(c.get('payload', {}).get('course', '') for c in reranked_hits)
    return QueryResult(query_id=str(query_item.get('query_id') or query_item.get('id', idx)), query_text=query_text, expected_id=str(query_item.get('expected_id') or query_item.get('expected_doc_id') or query_item.get('document_id', '')), course=str(query_item.get('course', '')), topic=int(topic_map.get(query_text, {}).get('topic')) if topic_map and topic_map.get(query_text, {}).get('topic') is not None else None, subtopic=None, hit_ids=hit_ids, hit_scores=hit_scores, hit_courses=hit_courses, latency_ms=latency_ms, code_integrity_ref=0.0)

def _build_cache_key(collection: str, model_name: str, test_set: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> str:
    """RESPONSIBILITY: Stable cache key from collection + model + sorted query ids + retrieval config."""
    query_ids = sorted(str(q.get('query_id', i)) for i, q in enumerate(test_set))
    config_str = ''
    if config:
        config_str = str(config.get('boost_question', '')) + str(config.get('boost_text', '')) + str(config.get('rrf_k', '')) + str(config.get('course_filter', ''))
    key_str = collection + model_name + ''.join(query_ids) + config_str
    return hashlib.md5(key_str.encode()).hexdigest()


def _build_retrieved_cache(
    client: QdrantClient,
    test_set: List[Dict[str, Any]],
    collection: str,
    config: Dict[str, Any],
    embedding_model: SentenceTransformer,
    model_name: str = '',
    cache_dir: Optional[Path] = None,
    reset: bool = False,
) -> Dict[str, Dict]:
    """RESPONSIBILITY: Retrieve and embed all queries once, cache results for reuse across rerankers."""
    if cache_dir and model_name and not reset:
        cache_key = _build_cache_key(collection, model_name, test_set, config)
        cache_file = Path(cache_dir) / f"retrieved_{cache_key}.json"
        if cache_file.exists():
            logger.info("Disk cache hit — loading candidates from %s", cache_file)
            return json.loads(cache_file.read_text())

    cache = {}
    logger.info("Pre-fetching candidates for %d queries (shared across all rerankers)...", len(test_set))
    for idx, query_item in enumerate(test_set, start=1):
        query_text = _extract_query_text(query_item)
        if not query_text:
            continue
        query_vector = _generate_query_embedding(query_text, embedding_model)
        if not query_vector:
            continue
        retrieval_result = _execute_vector_search(client, collection, query_vector, config)
        if not retrieval_result:
            continue
        hit_ids = getattr(retrieval_result, 'hit_ids', ())
        hit_answers = getattr(retrieval_result, 'hit_answers', ())
        hit_courses = getattr(retrieval_result, 'hit_courses', ())
        hit_scores = getattr(retrieval_result, 'hit_scores', ())
        candidates = [
            {
                'es_id': hit_ids[i],
                'answer': hit_answers[i] if i < len(hit_answers) else '',
                'course': hit_courses[i] if i < len(hit_courses) else '',
                'score': hit_scores[i] if i < len(hit_scores) else 0.0,
            }
            for i in range(len(hit_ids))
        ]
        # Sort candidates by answer length to minimise padding waste in reranker batches
        candidates.sort(key=lambda c: len(c['answer']))
        cache[query_item.get('query_id', idx)] = {
            'query_item': query_item,
            'query_text': query_text,
            'candidates': candidates,
        }
        if idx % 50 == 0:
            logger.info("Pre-fetch progress: %d/%d queries", idx, len(test_set))
    logger.info("Pre-fetch complete: %d queries cached", len(cache))

    if cache_dir and model_name:
        cache_key = _build_cache_key(collection, model_name, test_set, config)
        cache_file = Path(cache_dir) / f"retrieved_{cache_key}.json"
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache))
        logger.info("Disk cache saved to %s", cache_file)

    return cache


def run_benchmark_loop(
    retrieved_cache: Dict[str, Dict],
    model_entry: Dict[str, Any],
    encoder: Any,
    topic_map: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """RESPONSIBILITY: Runs reranking over pre-fetched candidates — no redundant retrieval or embedding."""
    results = []
    model_name = encoder.model_name
    logger.info("Starting reranking loop for %d cached queries with %s...", len(retrieved_cache), model_name)
    for idx, (query_id, entry) in enumerate(retrieved_cache.items(), start=1):
        try:
            query_item = entry['query_item']
            query_text = entry['query_text']
            candidates = [c.copy() for c in entry['candidates']]  # copy to avoid mutation across rerankers
            if not candidates:
                continue
            reranked_hits, latency_ms = _score_and_sort_candidates(encoder, query_text, candidates)
            hit_ids = tuple(c.get('es_id', '') for c in reranked_hits)
            hit_scores = tuple(float(c.get('score', 0.0)) for c in reranked_hits)
            hit_courses = tuple(c.get('course', '') for c in reranked_hits)
            result = QueryResult(
                query_id=str(query_item.get('query_id') or query_item.get('id', idx)),
                query_text=query_text,
                expected_id=str(query_item.get('expected_id') or query_item.get('expected_doc_id') or query_item.get('document_id', '')),
                course=str(query_item.get('course', '')),
                topic=int(topic_map.get(query_text, {}).get('topic')) if topic_map and topic_map.get(query_text, {}).get('topic') is not None else None,
                subtopic=None,
                hit_ids=hit_ids,
                hit_scores=hit_scores,
                hit_courses=hit_courses,
                latency_ms=latency_ms,
                code_integrity_ref=0.0,
            )
            results.append(result)
            if idx % 50 == 0:
                logger.info("Reranking progress: %d/%d queries", idx, len(retrieved_cache))
        except Exception as e:
            logger.error("Failed reranking query %s: %s", query_id, e, exc_info=True)
            continue
    if not results:
        logger.warning("Zero results for reranker: %s", model_name)
        return None
    cfg_name = f"onnx_{model_name.replace('/', '_')}"
    summary = aggregate_metrics(results, cfg_name, model_entry['name'])
    logger.info("🏆 %s | Hit@5=%s MRR=%s", model_name, f"{summary.hit_rate_5:.1%}", f"{summary.mrr:.4f}")
    return (summary, results)

def setup_bi_encoder_context(config: Any) -> Dict[str, Any]:
    """RESPONSIBILITY: Matches active environment models to structural metadata stored in models.json."""
    model_entries = config.get_model_entries()
    if not model_entries:
        raise ValueError('No base embedding models mapped inside models.json')
    target_model, collection_name = extract_active_environment()
    model_entry = next((m for m in model_entries if m['name'] == target_model), model_entries)
    model_entry['collection'] = collection_name
    return model_entry

def prepare_sliced_test_set(config: Any, args: Any) -> List[Dict[str, Any]]:
    """RESPONSIBILITY: Loads evaluation datasets and applies requested sample constraints."""
    test_set = config.get_test_set()
    sample_size = getattr(args, 'sample_size', 0)
    if sample_size > 0:
        test_set = test_set[:sample_size]
    print(f'[INFO] Running on {len(test_set)} queries (full dataset: {sample_size == 0})')
    return test_set

def parse_runtime_hyperparameters(args: Any) -> Dict[str, Any]:
    """RESPONSIBILITY: Organizes unvarying extraction configurations for retriever components."""
    return {'boost_question': 5.0, 'boost_text': 5.0, 'rrf_k': 60, 'course_filter': getattr(args, 'course_filter', None)}

def execute_matrix_evaluation(
    client: QdrantClient,
    test_set: List[Any],
    embedding_entry: Dict[str, Any],
    reranker_entries: List[Dict[str, Any]],
    retrieval_config: Dict[str, Any],
    embedding_model: SentenceTransformer,
    topic_map: Optional[Dict[str, Any]] = None,
    target_override: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    reset: bool = False,
) -> List[Any]:
    """RESPONSIBILITY: Retrieve once, rerank with all models in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    active_rerankers = [
        r for r in reranker_entries
        if not target_override or r['name'] == target_override
    ]

    # Step 1: retrieve and embed all queries once
    collection = embedding_entry['collection']
    retrieved_cache = _build_retrieved_cache(
        client=client,
        test_set=test_set,
        collection=collection,
        config=retrieval_config,
        embedding_model=embedding_model,
        model_name=embedding_entry.get('name', ''),
        cache_dir=cache_dir,
        reset=reset,
    )

    if not retrieved_cache:
        logger.warning("No candidates retrieved — aborting matrix evaluation.")
        return []

    # Step 2: load all reranker models
    encoders = {}
    for reranker_entry in active_rerankers:
        try:
            encoders[reranker_entry['name']] = RerankerRunner(model_key=reranker_entry['name'])
        except Exception as e:
            logger.error("Failed to load reranker '%s': %s", reranker_entry['name'], e)

    # Step 3: run all rerankers in parallel
    summaries = []

    def _run_single(reranker_entry):
        name = reranker_entry['name']
        encoder = encoders.get(name)
        if not encoder:
            return None
        logger.info("🚀 Starting reranker: %s (%s)", name, reranker_entry['model'])
        return run_benchmark_loop(
            retrieved_cache=retrieved_cache,
            model_entry=reranker_entry,
            encoder=encoder,
            topic_map=topic_map,
        )

    with ThreadPoolExecutor(max_workers=len(active_rerankers)) as executor:
        futures = {executor.submit(_run_single, r): r for r in active_rerankers}
        for future in as_completed(futures):
            reranker_entry = futures[future]
            try:
                result = future.result()
                if result:
                    summary, results = result
                    summaries.append((summary, results, reranker_entry['name']))
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error("Reranker '%s' failed: %s", reranker_entry['name'], e)

    return summaries
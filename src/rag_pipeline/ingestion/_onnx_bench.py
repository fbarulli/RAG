"""
rag_pipeline/p04_ingestion/_onnx_bench.py

Core evaluation processing functions for the ONNX Cross-Encoder matrix.
RESPONSIBILITY: Manages individual test query iteration execution flows.
"""
import logging
import traceback
import time
from typing import Any, Dict, List, Optional, Tuple
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from ._onnx_bench_config import extract_active_environment
from ._onnx_bench_engine import compile_onnx_runtime_node, prepare_candidates_from_hits
from ._benchmark_metrics.evaluation import run_entity_boosted_retrieval
from ._benchmark_metrics.aggregation import aggregate_metrics
from ._benchmark_types import QueryResult
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
    return run_entity_boosted_retrieval(client=client, collection=collection, query_vector=query_vector, course_filter=config.get('course_filter', 'all'), config=config, top_k=40, ner_category=None, ner_primary_entity=None)

def _extract_hit_ids_safely(retrieval_result: Any) -> List[Any]:
    """RESPONSIBILITY: Safe schema interface extractor handling both custom objects and raw dict formats."""
    if not retrieval_result:
        return []
    if isinstance(retrieval_result, dict):
        return retrieval_result.get('hit_ids', [])
    return getattr(retrieval_result, 'hit_ids', None) or getattr(retrieval_result, 'ids', [])

def _retrieve_vector_candidates(client: QdrantClient, collection: str, query_item: Dict[str, Any], config: Dict[str, Any], embedding_model: SentenceTransformer) -> List[Dict[str, Any]]:
    """RESPONSIBILITY: High-level orchestrator organizing synchronous textual, embedding, and vector search stages."""
    query_text = _extract_query_text(query_item)
    if not query_text:
        return []
    query_vector = _generate_query_embedding(query_text, embedding_model)
    if not query_vector:
        return []
    retrieval_result = _execute_vector_search(client, collection, query_vector, config)
    hit_ids = _extract_hit_ids_safely(retrieval_result)
    if not hit_ids:
        return []
    return prepare_candidates_from_hits(client=client, collection=collection, hit_ids=hit_ids, top_k=40)

def _score_and_sort_candidates(encoder: Any, query_text: str, candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    """RESPONSIBILITY: Tokenizes, scores via ONNX, and sorts document matches."""
    pairs = [(query_text, c['answer']) for c in candidates]
    t0 = time.perf_counter()
    scores = encoder.predict(pairs, batch_size=32, convert_to_numpy=True)
    latency_ms = (time.perf_counter() - t0) * 1000
    for i, score in enumerate(scores):
        candidates[i]['score'] = float(score)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return (candidates, latency_ms)

def _evaluate_single_query(client: QdrantClient, query_item: Dict[str, Any], idx: int, collection: str, config: Dict[str, Any], encoder: Any, embedding_model: SentenceTransformer, topic_map: Optional[Dict[str, Any]]=None) -> Optional[Dict[str, Any]]:
    """RESPONSIBILITY: Orchestrates a single query through retrieval and ranking evaluation steps."""
    query_text = _extract_query_text(query_item)
    candidates = _retrieve_vector_candidates(client, collection, query_item, config, embedding_model)
    if not candidates:
        return None
    reranked_hits, latency_ms = _score_and_sort_candidates(encoder, query_text, candidates)
    hit_ids = tuple(c.get('es_id', '') or c.get('payload', {}).get('es_id', '') for c in reranked_hits)
    hit_scores = tuple(float(c.get('score', 0.0)) for c in reranked_hits)
    hit_courses = tuple(c.get('payload', {}).get('course', '') for c in reranked_hits)
    return QueryResult(query_id=str(query_item.get('id', idx)), query_text=query_text, expected_id=str(query_item.get('expected_doc_id') or query_item.get('document_id', '')), course=str(query_item.get('course', '')), topic=topic_map.get(query_text) if topic_map else None, subtopic=None, hit_ids=hit_ids, hit_scores=hit_scores, hit_courses=hit_courses, latency_ms=latency_ms, code_integrity_ref=0.0)

def run_benchmark_loop(client: QdrantClient, test_set: List[Dict[str, Any]], model_entry: Dict[str, Any], config: Dict[str, Any], encoder: Any, embedding_model: SentenceTransformer, topic_map: Optional[Dict[str, Any]]=None) -> Optional[Any]:
    """RESPONSIBILITY: Manages outer tracking metrics loop iteration boundaries."""
    results = []
    collection = model_entry['collection']
    model_name = encoder.model_name
    logger.info('Starting processing strategy loop for %d evaluation queries...', len(test_set))
    for idx, query_item in enumerate(test_set, start=1):
        try:
            evaluated_item = _evaluate_single_query(client, query_item, idx, collection, config, encoder, embedding_model, topic_map)
            if evaluated_item:
                results.append(evaluated_item)
            if idx % 50 == 0:
                logger.info('Progress Check | Evaluated %d/%d queries successfully.', idx, len(test_set))
        except Exception as e:
            logger.error("Failed executing evaluation task block on query index %d: %s", idx, e, exc_info=True)
            continue
    if not results:
        logger.warning('Zero execution nodes completed for target model matrix: %s', model_name)
        return None
    cfg_name = f"onnx_{model_name.replace('/', '_')}"
    summary = aggregate_metrics(results, cfg_name, model_entry['name'])
    logger.info('🏆 Final Performance Matrix for %s: Hit@5=%s MRR=%s', model_name, f'{summary.hit_rate_5:.1%}', f'{summary.mrr:.4f}')
    return summary

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
    return {'boost_question': 5.0, 'boost_text': 5.0, 'rrf_k': 60, 'course_filter': getattr(args, 'course_filter', 'machine-learning-zoomcamp')}

def execute_matrix_evaluation(client: QdrantClient, test_set: List[Any], embedding_entry: Dict[str, Any], reranker_entries: List[Dict[str, Any]], retrieval_config: Dict[str, Any], embedding_model: SentenceTransformer, topic_map: Optional[Dict[str, Any]]=None, target_override: str=None) -> List[Any]:
    """RESPONSIBILITY: Orchestrates the sequential profiling loops over cross-encoder matrices."""
    summaries = []
    for reranker_entry in reranker_entries:
        if target_override and reranker_entry['name'] != target_override:
            continue
        try:
            logger.info('🚀 Starting Benchmark Matrix Node: %s (%s)', reranker_entry['name'], reranker_entry['model'])
            encoder = compile_onnx_runtime_node(model_key=reranker_entry['model'], max_length=reranker_entry.get('max_length', 512), provider='CPUExecutionProvider')
            summary = run_benchmark_loop(client, test_set, embedding_entry, retrieval_config, encoder, embedding_model, topic_map)
            if summary:
                summaries.append(summary)
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error("Skipping node failure on cross-encoder '%s': %s", reranker_entry['name'], e)
            continue
    return summaries
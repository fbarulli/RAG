"""
production_pipeline/p04_ingestion/_benchmark_metrics/evaluation.py
Evaluation orchestration - ties retrievers to test data.
"""

import time
import logging
from typing import Optional, TYPE_CHECKING

from ._benchmark_types import QueryResult
from ._benchmark_reranker import evaluate_with_reranker
from _benchmark_metrics.core import check_code_integrity
from _benchmark_metrics.retrievers import (
    run_es_retrieval,
    run_vector_retrieval,
    run_vector_retrieval_with_reranker,
    run_hybrid_rrf_retrieval,
    run_entity_boosted_retrieval,
)

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode_queries(model, test_set: list[dict], encode_batch_size: int) -> list[list[float]]:
    """
    Batch-encode all queries upfront and convert to plain Python lists.
    Pre-converting avoids repeated .tolist() calls inside the main loop.
    """
    queries = [test["query"] for test in test_set]
    vectors = model.encode(
        queries,
        batch_size=encode_batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return vectors.tolist()


def _validate_config(search_type: str, use_reranker: bool, es, model) -> None:
    """Raise early if required dependencies are missing for the chosen search type."""
    if search_type in ("bm25", "hybrid_rrf") and es is None:
        raise ValueError(
            f"Search type '{search_type}' requires Elasticsearch connection. "
            "Provide 'es' parameter or use a different config."
        )
    if search_type in ("vector", "hybrid_rrf", "entity_boosted") and model is None:
        raise ValueError(
            f"Search type '{search_type}' requires an embedding model. "
            "Provide 'model' parameter or use a different config."
        )
    if use_reranker and search_type == "bm25" and model is None:
        raise ValueError(
            "Reranking with bm25 requires an embedding model for the reranker. "
            "Provide 'model' parameter."
        )


def _run_retrieval(
    search_type: str,
    use_reranker: bool,
    reranker_name: Optional[str],
    retrieval_k: int,
    top_k: int,
    client,
    collection: str,
    query_vector: Optional[list],
    query: str,
    course: str,
    config: dict,
    ner_category: Optional[str],
    ner_primary_entity: Optional[str],
    es,
    es_index: str,
):
    """Dispatch to the correct retriever based on search_type."""
    if search_type == "bm25" and es is not None:
        return run_es_retrieval(
            es=es, index=es_index,
            query_text=query, course_filter=course,
            config=config, top_k=retrieval_k,
        )

    if search_type == "hybrid_rrf" and es is not None:
        return run_hybrid_rrf_retrieval(
            client=client, collection=collection,
            query_vector=query_vector,
            es=es, es_index=es_index,
            query_text=query, course_filter=course,
            config=config, top_k=retrieval_k,
        )

    if search_type == "entity_boosted":
        logger.debug(
            f"Calling entity_boosted retrieval — "
            f"ner_category={ner_category!r}, "
            f"ner_primary_entity={ner_primary_entity!r}, "
            f"top_k={retrieval_k}"
        )
        t0 = time.perf_counter()
        result = run_entity_boosted_retrieval(
            client=client, collection=collection,
            query_vector=query_vector, course_filter=course,
            config=config, top_k=retrieval_k,
            ner_category=ner_category,
            ner_primary_entity=ner_primary_entity,
        )
        logger.debug(
            f"entity_boosted retrieval returned in {(time.perf_counter() - t0) * 1000:.1f}ms"
        )
        return result

    # vector (default)
    if use_reranker and reranker_name:
        return run_vector_retrieval_with_reranker(
            client=client, collection=collection,
            query_vector=query_vector, query_text=query,
            course_filter=course, config=config,
            top_k=top_k, reranker_name=reranker_name,
        )

    return run_vector_retrieval(
        client=client, collection=collection,
        query_vector=query_vector, course_filter=course,
        config=config, top_k=top_k,
    )


def _apply_reranking(
    search_result,
    search_type: str,
    use_reranker: bool,
    reranker_name: Optional[str],
    query: str,
    top_k: int,
):
    """
    Apply reranking post-retrieval for non-vector search types.
    Vector search handles reranking internally via run_vector_retrieval_with_reranker.
    Returns (hit_ids, hit_scores, reranker_latency_ms).

    Post-rerank scores use descending reciprocal rank (1/rank) as a proxy
    so rank-weighted metrics like NDCG remain meaningful.
    """
    hit_ids             = search_result.hit_ids
    hit_scores          = search_result.hit_scores
    reranker_latency_ms = search_result.reranker_latency_ms

    if not (use_reranker and reranker_name and search_type != "vector"):
        logger.debug(
            f"Reranking skipped — use_reranker={use_reranker}, "
            f"reranker_name={reranker_name!r}, search_type={search_type!r}"
        )
        return hit_ids, hit_scores, reranker_latency_ms

    if not hit_ids:
        logger.warning(
            f"Reranker '{reranker_name}' skipped — retrieval returned no candidates "
            f"for search_type='{search_type}'."
        )
        return hit_ids, hit_scores, reranker_latency_ms

    logger.debug(
        f"Calling evaluate_with_reranker — "
        f"reranker_name={reranker_name!r}, "
        f"candidates={len(hit_ids)}, "
        f"top_k={top_k}"
    )

    hit_answers = getattr(search_result, "hit_answers", None) or [""] * len(hit_ids)
    candidates = [
        {
            "es_id":    hit_ids[i],
            "question": query,
            "answer":   hit_answers[i],
        }
        for i in range(len(hit_ids))
    ]

    reranked_ids, rerank_metrics = evaluate_with_reranker(
        query=query,
        retrieved_candidates=candidates,
        reranker_name=reranker_name,
        top_k=top_k,
    )

    logger.debug(
        f"evaluate_with_reranker returned — "
        f"reranked_ids={len(reranked_ids)}, "
        f"reranker_latency_ms={rerank_metrics.get('reranker_latency_ms', 0.0):.1f}ms"
    )

    hit_ids             = tuple(reranked_ids)
    hit_scores          = tuple(1.0 / (rank + 1) for rank in range(len(hit_ids)))
    reranker_latency_ms = rerank_metrics.get("reranker_latency_ms", 0.0)

    return hit_ids, hit_scores, reranker_latency_ms


def _evaluate_single(
    idx: int,
    test: dict,
    topic_map: dict,
    query_vectors: Optional[list],
    search_type: str,
    use_reranker: bool,
    reranker_name: Optional[str],
    retrieval_k: int,
    top_k: int,
    client,
    collection: str,
    config: dict,
    es,
    es_index: str,
) -> QueryResult:
    """
    Evaluate a single test item: retrieve, optionally rerank, build QueryResult.
    Extracted for unit testability and to keep evaluate_config readable.
    """
    query              = test["query"]
    expected_id        = test["expected_id"]
    course             = test["course"]
    ref_answer         = test["answer"]
    topic_info         = topic_map.get(expected_id, {})
    topic              = topic_info.get("topic")
    subtopic           = topic_info.get("subtopic")
    ner_category       = topic_info.get("ner_category")
    ner_primary_entity = topic_info.get("ner_primary_entity")
    query_vector       = query_vectors[idx] if query_vectors is not None else None

    logger.debug(f"[{idx + 1}] Retrieving: {query!r} (course={course!r})")

    search_result = _run_retrieval(
        search_type=search_type,
        use_reranker=use_reranker,
        reranker_name=reranker_name,
        retrieval_k=retrieval_k,
        top_k=top_k,
        client=client,
        collection=collection,
        query_vector=query_vector,
        query=query,
        course=course,
        config=config,
        ner_category=ner_category,
        ner_primary_entity=ner_primary_entity,
        es=es,
        es_index=es_index,
    )

    logger.debug(
        f"[{idx + 1}] Retrieval done — "
        f"hits={len(search_result.hit_ids)}, "
        f"latency={search_result.latency_ms:.1f}ms"
    )

    hit_ids, hit_scores, reranker_latency_ms = _apply_reranking(
        search_result=search_result,
        search_type=search_type,
        use_reranker=use_reranker,
        reranker_name=reranker_name,
        query=query,
        top_k=top_k,
    )

    logger.debug(
        f"[{idx + 1}] Reranking done — "
        f"reranked_hits={len(hit_ids)}, "
        f"reranker_latency={reranker_latency_ms:.1f}ms"
    )

    return QueryResult(
        query_id=test["query_id"],
        query_text=query,
        expected_id=expected_id,
        course=course,
        topic=topic,
        subtopic=subtopic,
        query_type=test.get("query_type", "unknown"),
        hit_ids=hit_ids,
        hit_scores=hit_scores,
        hit_courses=search_result.hit_courses,
        latency_ms=search_result.latency_ms,
        reranker_latency_ms=reranker_latency_ms,
        code_integrity_ref=check_code_integrity(ref_answer),
        code_integrity_retrieved=(
            check_code_integrity(search_result.top_answer)
            if search_result.top_answer else None
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_config(
    client,
    collection: str,
    model,
    test_set: list[dict],
    topic_map: dict,
    config: dict,
    top_k: int,
    es: "Elasticsearch | None" = None,
    es_index: str = "faqs",
    encode_batch_size: int = 32,
) -> list[QueryResult]:
    """
    Evaluate one (config, model) pair against the full test set.
    Applies reranking post-retrieval for any search_type when config
    specifies reranker=true and reranker_name.
    """
    search_type   = config.get("search_type", "vector")
    use_reranker  = config.get("reranker", False)
    reranker_name = config.get("reranker_name") if use_reranker else None

    retrieval_k = top_k * 4 if use_reranker else top_k

    _validate_config(search_type, use_reranker, es, model)

    query_vectors = None
    if search_type in ("vector", "hybrid_rrf", "entity_boosted"):
        query_vectors = _encode_queries(model, test_set, encode_batch_size)
        logger.info(
            f"Encoding complete — {len(query_vectors)} query vectors ready. "
            f"Starting retrieval loop over {len(test_set)} items."
        )

    total   = len(test_set)
    results = []

    for idx, test in enumerate(test_set):
        if idx % 10 == 0:
            logger.info(f"Progress: {idx}/{total} queries evaluated")
        try:
            results.append(_evaluate_single(
                idx=idx,
                test=test,
                topic_map=topic_map,
                query_vectors=query_vectors,
                search_type=search_type,
                use_reranker=use_reranker,
                reranker_name=reranker_name,
                retrieval_k=retrieval_k,
                top_k=top_k,
                client=client,
                collection=collection,
                config=config,
                es=es,
                es_index=es_index,
            ))
        except Exception as e:
            logger.error(
                f"[{idx}] Query '{test.get('query_id')}' failed: {e}",
                exc_info=True,
            )

    logger.info(f"Evaluation complete — {len(results)}/{total} queries succeeded.")
    return results
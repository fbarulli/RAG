"""
production_pipeline/p04_ingestion/_benchmark_metrics/evaluation.py
Evaluation orchestration - ties retrievers to test data.
"""

from typing import Optional, TYPE_CHECKING

from .._benchmark_types import QueryResult
from .._benchmark_reranker import evaluate_with_reranker
from .core import check_code_integrity
from .retrievers import (
    run_es_retrieval,
    run_vector_retrieval,
    run_vector_retrieval_with_reranker,
    run_hybrid_rrf_retrieval,
    run_entity_boosted_retrieval,
)

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch


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

    # Fetch a larger candidate pool when reranking so the reranker has more to work with
    retrieval_k = top_k * 4 if use_reranker else top_k

    # Validate requirements before doing any work
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

    # Pre-encode all queries in one batched pass for vector-based search types
    needs_vectors = search_type in ("vector", "hybrid_rrf", "entity_boosted")
    query_vectors = None

    if needs_vectors:
        queries = [test["query"] for test in test_set]
        query_vectors = model.encode(
            queries,
            batch_size=encode_batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    results: list[QueryResult] = []

    for idx, test in enumerate(test_set):
        query              = test["query"]
        expected_id        = test["expected_id"]
        course             = test["course"]
        ref_answer         = test["answer"]
        topic_info         = topic_map.get(expected_id, {})
        topic              = topic_info.get("topic")
        subtopic           = topic_info.get("subtopic")
        ner_category       = topic_info.get("ner_category")
        ner_primary_entity = topic_info.get("ner_primary_entity")
        query_vector       = query_vectors[idx].tolist() if query_vectors is not None else None

        # --- Retrieval ---
        if search_type == "bm25" and es is not None:
            search_result = run_es_retrieval(
                es=es, index=es_index,
                query_text=query, course_filter=course,
                config=config, top_k=retrieval_k,
            )
        elif search_type == "hybrid_rrf" and es is not None:
            search_result = run_hybrid_rrf_retrieval(
                client=client, collection=collection,
                query_vector=query_vector,
                es=es, es_index=es_index,
                query_text=query, course_filter=course,
                config=config, top_k=retrieval_k,
            )
        elif search_type == "entity_boosted":
            search_result = run_entity_boosted_retrieval(
                client=client, collection=collection,
                query_vector=query_vector, course_filter=course,
                config=config, top_k=retrieval_k,
                ner_category=ner_category,
                ner_primary_entity=ner_primary_entity,
            )
        else:  # vector
            if use_reranker and reranker_name:
                # Legacy path: vector retrieval with reranker handled inside retriever
                search_result = run_vector_retrieval_with_reranker(
                    client=client, collection=collection,
                    query_vector=query_vector, query_text=query,
                    course_filter=course, config=config,
                    top_k=top_k, reranker_name=reranker_name,
                )
            else:
                search_result = run_vector_retrieval(
                    client=client, collection=collection,
                    query_vector=query_vector, course_filter=course,
                    config=config, top_k=top_k,
                )

        # --- Reranking (applies to all search_types except vector which handles it internally) ---
        hit_ids            = search_result.hit_ids
        hit_scores         = search_result.hit_scores
        reranker_latency_ms = search_result.reranker_latency_ms

        if use_reranker and reranker_name and search_type != "vector":
            candidates = [{"es_id": id_} for id_ in hit_ids]
            reranked_ids, rerank_metrics = evaluate_with_reranker(
                query=query,
                retrieved_candidates=candidates,
                reranker_name=reranker_name,
                top_k=top_k,
            )
            hit_ids             = tuple(reranked_ids)
            hit_scores          = tuple(0.0 for _ in hit_ids)  # scores not meaningful post-rerank
            reranker_latency_ms = rerank_metrics.get("reranker_latency_ms", 0.0)

        results.append(QueryResult(
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
        ))

    return results
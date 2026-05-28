"""
Facade module for retrievers.
Exports all retrieval functions to maintain compatibility with evaluation.py.
"""
from .es_retrievers import run_es_retrieval
from .qdrant_retrievers import run_vector_retrieval
from .composite_retrievers import (
    run_hybrid_rrf_retrieval,
    run_hybrid_dbsf_retrieval,
    run_entity_boosted_retrieval,
    run_entity_category_boosted_retrieval,
    run_vector_retrieval_with_reranker
)

__all__ = [
    "run_es_retrieval",
    "run_vector_retrieval",
    "run_hybrid_rrf_retrieval",
    "run_hybrid_dbsf_retrieval",
    "run_entity_boosted_retrieval",
    "run_entity_category_boosted_retrieval",
    "run_vector_retrieval_with_reranker",
]


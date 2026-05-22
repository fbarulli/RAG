"""
rag_pipeline/ingestion/_onnx_bench_engine.py
ONNX Benchmark Engine - using the cleaned reranker
"""

import logging
from typing import Dict, Any
from .reranker_runner import RerankerRunner

logger = logging.getLogger(__name__)

class ONNXBenchEngine:
    """Simplified benchmark engine using the new reranker."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.reranker = RerankerRunner(model_name=model_name)

    def rerank_results(self, query: str, candidates: list, top_k: int = 10) -> list:
        """Rerank retrieved candidates."""
        if not candidates:
            return []

        # Extract text from candidates (handle both str and dicts)
        docs = []
        for c in candidates:
            if isinstance(c, dict) and 'text' in c:
                docs.append(c['text'])
            elif isinstance(c, str):
                docs.append(c)
            else:
                docs.append(str(c))

        reranked = self.reranker.rerank(
            query=query,
            documents=docs,
            show_progress=False
        )

        # Return top_k with original candidate metadata if available
        final_results = []
        for doc, score in reranked[:top_k]:
            final_results.append({
                'text': doc,
                'score': float(score),
                'reranker': self.model_name
            })
        
        return final_results

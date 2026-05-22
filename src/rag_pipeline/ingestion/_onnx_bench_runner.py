"""
rag_pipeline/ingestion/_onnx_bench_runner.py
Updated benchmark runner using the clean reranker
"""

import logging
from ._reranker_runner import RerankerRunner
from ._onnx_bench_engine import ONNXBenchEngine

logger = logging.getLogger(__name__)

class ONNXBenchRunner:
    """Convenience wrapper for benchmarking."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.reranker = RerankerRunner(model_name=model_name)
        self.engine = ONNXBenchEngine(model_name=model_name)

    def run_reranking(self, query: str, candidates: list, top_k: int = 10):
        """Main entry point used by benchmarks."""
        logger.debug("Reranking %d candidates with %s", len(candidates), self.model_name)
        
        return self.engine.rerank_results(
            query=query,
            candidates=candidates,
            top_k=top_k
        )

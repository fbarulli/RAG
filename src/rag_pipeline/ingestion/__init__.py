"""
rag_pipeline/ingestion package init
"""

from ._reranker_runner import RerankerRunner
from ._onnx_cross_encoder import ONNXCrossEncoder

__all__ = ["RerankerRunner", "ONNXCrossEncoder"]

# Optional: convenient default instance
default_reranker = None

def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Easy access to reranker."""
    global default_reranker
    if default_reranker is None or default_reranker.model_name != model_name:
        default_reranker = RerankerRunner(model_name=model_name)
    return default_reranker

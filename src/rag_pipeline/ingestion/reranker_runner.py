"""
rag_pipeline/ingestion/_reranker_runner.py
Main reranker orchestration supporting config
"""

import logging
from typing import List, Tuple, Optional
import numpy as np

from .onnx_cross_encoder import ONNXCrossEncoder
from .reranker_config import get_model_config

logger = logging.getLogger(__name__)


class RerankerRunner:
    """High-level reranker supporting the config JSON."""

    def __init__(self, model_key: str = "MiniLM-L6"):
        self.model_config = get_model_config(model_key)
        self.model_name = self.model_config.model
        self.max_length = self.model_config.max_length
        self.model_key = model_key
        self._onnx_reranker = None

    @property
    def onnx_reranker(self):
        if self._onnx_reranker is None:
            logger.info("Loading ONNX reranker → %s (%s)", self.model_key, self.model_name)
            self._onnx_reranker = ONNXCrossEncoder(
                model_name=self.model_name,
                max_length=self.max_length,
                provider="CPUExecutionProvider",
                quantize=self.model_config.quantization
            )
        return self._onnx_reranker

    def rerank(self, query: str, documents: List[str], batch_size: Optional[int] = None, show_progress: bool = False):
        if not documents:
            return []

        pairs = [(query, doc) for doc in documents]

        scores = self.onnx_reranker.predict(
            pairs=pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress
        )

        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        return doc_score_pairs

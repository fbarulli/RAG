"""
rag_pipeline/p04_ingestion/_onnx_inference.py
ONNX Cross-Encoder Inference Engine

Provides optimized ONNX-based cross-encoder inference with batch processing,
numerical stability improvements, and memory-efficient operations.
"""
import logging
import numpy as np
from typing import Dict, List, Tuple, Union
from pathlib import Path
from rag_pipeline.ingestion._onnx_model_loader import ONNXModelLoader
logger = logging.getLogger(__name__)

class ONNXCrossEncoder:
    """
    Optimized ONNX cross-encoder with pure NumPy inference pipeline.
    """

    def __init__(self, model_name: str, max_length: int, provider: str='CPUExecutionProvider'):
        """I/O: model_name (str), max_length (int), provider (str) -> None"""
        logger.debug('ONNXCrossEncoder.__init__ | model_name=%r max_length=%d', model_name, max_length)
        self.model_name = model_name
        self.max_length = max_length
        self.provider = provider
        self.model_loader = ONNXModelLoader(model_name, provider)
        self.tokenizer = self.model_loader.tokenizer
        self.model = self.model_loader.model

    def predict(self, pairs: List[Tuple[str, str]], batch_size: int=32, show_progress_bar: bool=False, convert_to_numpy: bool=True) -> Union[np.ndarray, list]:
        """I/O: pairs (List[Tuple[str, str]]), batch_size (int), show_progress_bar (bool), 
             convert_to_numpy (bool) -> Union[np.ndarray, list]"""
        n_pairs = len(pairs)
        if n_pairs == 0:
            return np.array([], dtype=np.float32) if convert_to_numpy else []
        all_scores = np.empty(n_pairs, dtype=np.float32)
        current_idx = 0
        try:
            for batch_start in range(0, n_pairs, batch_size):
                batch_end = min(batch_start + batch_size, n_pairs)
                batch = pairs[batch_start:batch_end]
                batch_idx = batch_start // batch_size
                inputs = self._tokenize_batch(batch, batch_idx)
                logits = self._forward_batch(inputs, batch_idx)
                scores = self._extract_scores(logits, batch_idx)
                end_idx = current_idx + len(scores)
                all_scores[current_idx:end_idx] = scores
                current_idx = end_idx
            if convert_to_numpy:
                return all_scores
            return all_scores.tolist()
        except Exception as e:
            logger.error('ONNXCrossEncoder.predict | Failed\n%s', str(e))
            raise

    def _tokenize_batch(self, batch: List[Tuple[str, str]], batch_index: int) -> Dict[str, np.ndarray]:
        """I/O: batch (List[Tuple[str, str]]), batch_index (int) -> Dict[str, np.ndarray]"""
        try:
            return self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors='np')
        except Exception as e:
            logger.error('ONNXCrossEncoder._tokenize_batch | Failed at batch %d\n%s', batch_index, str(e))
            raise

    def _forward_batch(self, inputs: Dict[str, np.ndarray], batch_index: int) -> np.ndarray:
        """I/O: inputs (Dict), batch_index (int) -> np.ndarray"""
        try:
            outputs = self.model(**inputs)
            return outputs.logits
        except Exception as e:
            logger.error('ONNXCrossEncoder._forward_batch | Failed at batch %d\n%s', batch_index, str(e))
            raise

    def _extract_scores(self, logits: np.ndarray, batch_index: int) -> np.ndarray:
        """I/O: logits (np.ndarray), batch_index (int) -> np.ndarray"""
        try:
            num_labels = logits.shape[-1]
            if num_labels >= 2:
                clipped_logits = np.clip(logits[:, 1], -500, 500)
                return 1.0 / (1.0 + np.exp(-clipped_logits))
            else:
                return logits.flatten()
        except Exception as e:
            logger.error('ONNXCrossEncoder._extract_scores | Failed at batch %d\n%s', batch_index, str(e))
            raise
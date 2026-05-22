"""
rag_pipeline/ingestion/_onnx_cross_encoder.py
ONNX Cross-Encoder - CPU optimized version
"""

import logging
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import os
from rag_pipeline.ingestion._onnx_model_loader import ONNXModelLoader

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None


class ONNXCrossEncoder:
    """
    High-performance ONNX Cross-Encoder optimized for CPU.
    """

    def __init__(self, model_name: str, max_length: int = 512, 
                 provider: str = 'CPUExecutionProvider', 
                 cache_dir: str = 'experiments/onnx_cache'):
        self.model_name = model_name
        self.max_length = max_length
        self.provider = provider
        
        # CPU-friendly default batch size
        self.default_batch_size = self._get_cpu_batch_size()
        
        logger.info('ONNXCrossEncoder initialized | model=%s | provider=%s | default_batch=%d', 
                   model_name, provider, self.default_batch_size)
        
        self.model_loader = ONNXModelLoader(
            model_name=model_name, 
            provider=provider, 
            cache_dir=cache_dir
        )
        self.model = self.model_loader.model
        self.tokenizer = self.model_loader.tokenizer
        
        self._verify_model_structure()

    def _get_cpu_batch_size(self) -> int:
        """Return sensible batch size based on CPU cores."""
        cpu_count = os.cpu_count() or 4
        if cpu_count <= 4:
            return 8
        elif cpu_count <= 8:
            return 16
        else:
            return 24

    def _verify_model_structure(self) -> None:
        try:
            dummy_input = self.tokenizer(['test query'], ['test document'], 
                                       return_tensors='np', truncation=True, max_length=self.max_length)
            _ = self.model(**dummy_input)
            logger.debug('Model verification successful')
        except Exception as e:
            logger.error('Model verification failed: %s', e)
            raise

    def predict(self, pairs: List[Tuple[str, str]], 
                batch_size: Optional[int] = None, 
                show_progress_bar: bool = False,
                convert_to_numpy: bool = True) -> Union[np.ndarray, List[float]]:
        
        if batch_size is None:
            batch_size = self.default_batch_size

        n_pairs = len(pairs)
        if n_pairs == 0:
            return np.array([], dtype=np.float32) if convert_to_numpy else []

        all_scores = np.empty(n_pairs, dtype=np.float32)
        iterator = range(0, n_pairs, batch_size)
        
        if show_progress_bar and HAS_TQDM:
            iterator = tqdm(iterator, desc=f'ONNX Rerank ({self.model_name})', 
                          total=(n_pairs + batch_size - 1) // batch_size)

        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, n_pairs)
            batch_pairs = pairs[batch_start:batch_end]
            
            inputs = self.tokenizer(
                batch_pairs, 
                padding=True, 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors='np'
            )
            
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Extract scores
            if len(logits.shape) == 1:
                logits = logits.reshape(-1, 1)
            scores = logits[:, 1] if logits.shape[-1] >= 2 else logits.flatten()
            scores = self._sigmoid(scores)
            
            all_scores[batch_start:batch_end] = scores.astype(np.float32)

        return all_scores if convert_to_numpy else all_scores.tolist()

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Stable sigmoid."""
        try:
            from scipy.special import expit
            return expit(x)
        except ImportError:
            x = np.clip(x, -500, 500)
            return 1.0 / (1.0 + np.exp(-x))

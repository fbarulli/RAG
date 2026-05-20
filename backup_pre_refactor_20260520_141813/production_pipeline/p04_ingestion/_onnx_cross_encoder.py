"""
production_pipeline/p04_ingestion/_onnx_cross_encoder.py
ONNX Cross-Encoder Inference Engine

Handles batched inference for cross-encoder models using ONNX Runtime via Optimum.
Optimized for benchmarking workflows with minimal overhead.
"""
import logging
from typing import Dict, List, Tuple, Union, Optional

import numpy as np

from production_pipeline.p04_ingestion._onnx_model_loader import ONNXModelLoader

logger = logging.getLogger(__name__)

# Try to import tqdm for progress bars, but don't fail if missing
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None


class ONNXCrossEncoder:
    """
    High-performance Cross-Encoder wrapper for ONNX models.
    
    Utilizes ONNXModelLoader for caching and initialization, then provides
    a streamlined interface for batch scoring of text pairs.
    """

    def __init__(
        self, 
        model_name: str, 
        max_length: int = 512, 
        provider: str = "CPUExecutionProvider",
        cache_dir: str = "experiments/onnx_cache"
    ):
        """
        Initializes the Cross-Encoder.

        Args:
            model_name: Hugging Face model identifier.
            max_length: Maximum sequence length for tokenization.
            provider: ONNX Runtime provider (e.g., CPUExecutionProvider).
            cache_dir: Directory for caching exported ONNX models.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.provider = provider
        
        logger.debug("Initializing ONNXCrossEncoder | model=%s provider=%s", model_name, provider)
        
        # Initialize model loader (handles caching logic)
        self.model_loader = ONNXModelLoader(
            model_name=model_name, 
            provider=provider, 
            cache_dir=cache_dir
        )
        
        # Bind model and tokenizer for quick access
        self.model = self.model_loader.model
        self.tokenizer = self.model_loader.tokenizer
        
        # Verify model output structure once during init
        self._verify_model_structure()

    def _verify_model_structure(self) -> None:
        """
        Performs a dry-run inference with dummy data to ensure the model 
        is correctly loaded and outputs expected logits shape.
        """
        try:
            dummy_input = self.tokenizer(
                ["test"], ["test"], 
                return_tensors="np", 
                truncation=True, 
                max_length=self.max_length
            )
            # Optimum models accept kwargs
            _ = self.model(**dummy_input)
            logger.debug("Model structure verification successful.")
        except Exception as e:
            logger.error("Model structure verification failed: %s", e)
            raise RuntimeError(f"Model {self.model_name} failed initialization check.") from e

    def predict(
        self, 
        pairs: List[Tuple[str, str]], 
        batch_size: int = 32, 
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True
    ) -> Union[np.ndarray, List[float]]:
        """
        Computes similarity scores for a list of text pairs.

        Args:
            pairs: List of (query, document) tuples.
            batch_size: Inference batch size.
            show_progress_bar: Display progress bar (requires tqdm).
            convert_to_numpy: If False, returns a list of floats.

        Returns:
            Array or list of similarity scores (0.0 to 1.0).
        """
        n_pairs = len(pairs)
        if n_pairs == 0:
            return np.array([], dtype=np.float32) if convert_to_numpy else []

        all_scores = np.empty(n_pairs, dtype=np.float32)
        
        # Setup progress bar
        iterator = range(0, n_pairs, batch_size)
        if show_progress_bar and HAS_TQDM:
            iterator = tqdm(iterator, desc="Scoring", total=(n_pairs + batch_size - 1) // batch_size)
        elif show_progress_bar and not HAS_TQDM:
            logger.warning("Progress bar requested but 'tqdm' is not installed.")

        try:
            for batch_start in iterator:
                batch_end = min(batch_start + batch_size, n_pairs)
                batch_pairs = pairs[batch_start:batch_end]
                
                # 1. Tokenize
                inputs = self._tokenize_batch(batch_pairs)
                
                # 2. Inference
                logits = self._forward_batch(inputs)
                
                # 3. Post-process (Sigmoid + Extract)
                scores = self._extract_scores(logits)
                
                # 4. Store
                all_scores[batch_start:batch_end] = scores

            return all_scores if convert_to_numpy else all_scores.tolist()
            
        except Exception as e:
            logger.error("ONNXCrossEncoder.predict | Failed during inference: %s", e)
            raise

    def _tokenize_batch(self, batch: List[Tuple[str, str]]) -> Dict[str, np.ndarray]:
        """
        Tokenizes a batch of text pairs.
        Optimized to remove unnecessary arguments from the hot path.
        """
        return self.tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="np"
        )

    def _forward_batch(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Runs the ONNX model on provided inputs.
        Uses Optimum's __call__ interface which handles the ONNX session internally.
        """
        try:
            # Optimum ORTModel expects inputs as kwargs
            # outputs is a SequenceClassifierOutput object
            outputs = self.model(**inputs)
            return outputs.logits
        except Exception as e:
            logger.error("ONNXCrossEncoder._forward_batch | Inference failed: %s", e)
            raise

    def _extract_scores(self, logits: np.ndarray) -> np.ndarray:
        """
        Converts raw logits to similarity scores (0-1).
        Handles both single-label (regression) and dual-label (classification) outputs.
        """
        try:
            # Shape: (batch_size, num_labels)
            if len(logits.shape) == 1:
                logits = logits.reshape(-1, 1)
                
            num_labels = logits.shape[-1]
            
            if num_labels >= 2:
                # Classification mode: Use logits for the positive class (index 1)
                # Apply sigmoid to convert logit to probability
                raw_scores = logits[:, 1]
                scores = self._sigmoid(raw_scores)
            else:
                # Regression mode: Logits are already scores, apply sigmoid just in case
                raw_scores = logits.flatten()
                scores = self._sigmoid(raw_scores)
                
            return scores.astype(np.float32)
            
        except Exception as e:
            logger.error("ONNXCrossEncoder._extract_scores | Failed: %s", e)
            raise

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid function.
        Uses scipy if available, otherwise numpy with clipping.
        """
        try:
            from scipy.special import expit
            return expit(x)
        except ImportError:
            # Fallback to numpy with clipping to prevent overflow
            x = np.clip(x, -500, 500)
            return 1.0 / (1.0 + np.exp(-x))

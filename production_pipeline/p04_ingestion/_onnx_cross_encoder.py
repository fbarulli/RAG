""" ONNX Cross-Encoder Inference Engine """
import logging
import numpy as np
from typing import Dict, List, Tuple, Union
from production_pipeline.p04_ingestion._onnx_model_loader import ONNXModelLoader

logger = logging.getLogger(__name__)

class ONNXCrossEncoder:
    def __init__(self, model_name: str, max_length: int, provider: str = "CPUExecutionProvider"):
        logger.debug("ONNXCrossEncoder.__init__ | model_name=%r max_length=%d", model_name, max_length)
        self.model_name = model_name
        self.max_length = max_length
        self.provider = provider
        
        # Initialize model loader
        self.model_loader = ONNXModelLoader(model_name, provider)
        self.tokenizer = self.model_loader.tokenizer
        self.model = self.model_loader.model # This should be the ONNX Session

    def predict(
        self, 
        pairs: List[Tuple[str, str]], 
        batch_size: int = 32, 
        show_progress_bar: bool = False, 
        convert_to_numpy: bool = True, 
    ) -> Union[np.ndarray, list]:
        n_pairs = len(pairs)
        if n_pairs == 0:
            return np.array([], dtype=np.float32) if convert_to_numpy else []

        all_scores = np.empty(n_pairs, dtype=np.float32)
        current_idx = 0

        try:
            for batch_start in range(0, n_pairs, batch_size):
                batch_end = min(batch_start + batch_size, n_pairs)
                batch = pairs[batch_start:batch_end]
                
                inputs = self._tokenize_batch(batch, batch_start)
                logits = self._forward_batch(inputs, batch_start)
                scores = self._extract_scores(logits, batch_start)
                
                end_idx = min(current_idx + len(scores), n_pairs)
                all_scores[current_idx:end_idx] = scores[:end_idx-current_idx]
                current_idx = end_idx

            return all_scores if convert_to_numpy else all_scores.tolist()
        except Exception as e:
            logger.error("ONNXCrossEncoder.predict | Failed\n%s", str(e))
            raise

    def _tokenize_batch(self, batch: List[Tuple[str, str]], batch_index: int) -> Dict[str, np.ndarray]:
        return self.tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="np" 
        )

    def _forward_batch(self, inputs: Dict, batch_index: int) -> np.ndarray:
        try:
            # FIX: Actually run the ONNX session
            ort_inputs = {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
            }
            # self.model is the InferenceSession
            outputs = self.model.run(None, ort_inputs)
            return outputs[0] # Return logits
        except Exception as e:
            logger.error("ONNXCrossEncoder._forward_batch | Failed at batch %d\n%s", batch_index, str(e))
            raise

    def _extract_scores(self, logits: np.ndarray, batch_index: int) -> List[float]:
        try:
            num_labels = logits.shape[-1]
            if num_labels >= 2:
                clipped_logits = np.clip(logits[:, 1], -500, 500)
                scores = 1.0 / (1.0 + np.exp(-clipped_logits))
            else:
                scores = logits.flatten()
            return scores.tolist()
        except Exception as e:
            logger.error("ONNXCrossEncoder._extract_scores | Failed at batch %d\n%s", batch_index, str(e))
            raise

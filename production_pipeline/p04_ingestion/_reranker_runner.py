"""
production_pipeline/p04_ingestion/_reranker_runner.py
Reranker with optional dynamic quantization for better CPU performance.
"""

import time
from typing import List, Tuple, Dict
from sentence_transformers import CrossEncoder
import logging
import torch

logger = logging.getLogger(__name__)


def run_reranking(
    reranker_config: Dict,
    query: str,
    candidates: List[Dict],
    top_k: int = 5,
    use_quantization: bool = True
) -> Tuple[List[str], float]:
    """
    Rerank with optional dynamic quantization.
    """
    start = time.perf_counter()
    
    model_name = reranker_config["model"]
    logger.info(f"Running reranker: {reranker_config['name']} {'(quantized)' if use_quantization else ''}")

    try:
        model = CrossEncoder(
            model_name, 
            max_length=reranker_config.get("max_length", 512),
            device="cpu"
        )
        
        # Dynamic quantization (int8)
        if use_quantization and hasattr(model.model, 'quantize'):
            logger.info("Applying dynamic quantization (int8)...")
            model.model = torch.quantization.quantize_dynamic(
                model.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        
        # Prepare pairs
        pairs = []
        id_map = []
        for cand in candidates:
            text = f"{cand.get('question', '')} {cand.get('answer', '')}".strip() or cand.get('answer', '')
            pairs.append((query, text))
            id_map.append(cand.get('es_id', ''))
        
        # Inference
        scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
        
        # Sort
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        reranked_ids = [id_map[i] for i in sorted_indices[:top_k]]
        
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(f"Done in {latency_ms:.1f}ms")
        
        return reranked_ids, latency_ms
        
    except Exception as e:
        logger.error(f"Reranker failed: {e}")
        fallback = [c.get('es_id', '') for c in candidates[:top_k]]
        return fallback, 0.0
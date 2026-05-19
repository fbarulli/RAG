"""
production_pipeline/p04_ingestion/_reranker_runner.py
Reranker with model caching and dynamic INT8 quantization for CPU performance.
"""

import os
import time
import threading
import logging
from typing import List, Tuple, Dict, Optional, TypedDict

import numpy as np
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Global CPU performance settings — set once at import time, not per model load
torch.set_flush_denormal(True)
torch.set_num_threads(os.cpu_count() or 1)


class RerankerConfig(TypedDict, total=False):
    model:      str    # required
    name:       str
    max_length: int
    batch_size: int


_MODEL_CACHE: Dict[str, CrossEncoder] = {}
_CACHE_LOCK = threading.Lock()


def _load_model(model_name: str, max_length: int, use_quantization: bool) -> CrossEncoder:
    logger.info(f"Loading reranker model: {model_name} "
                f"(max_length={max_length}, quantized={use_quantization})")
    model = CrossEncoder(model_name, max_length=max_length, device="cpu")
    if use_quantization:
        logger.info(f"Applying INT8 dynamic quantization to {model_name}...")
        model.model = torch.quantization.quantize_dynamic(
            model.model, {torch.nn.Linear}, dtype=torch.qint8,
        )
        logger.info("Quantization applied.")
    return model


def _get_model(model_name: str, max_length: int, use_quantization: bool) -> CrossEncoder:
    cache_key = f"{model_name}:q{int(use_quantization)}:ml{max_length}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    with _CACHE_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]
        model = _load_model(model_name, max_length, use_quantization)
        _MODEL_CACHE[cache_key] = model
        return model


def run_reranking(
    reranker_config: RerankerConfig,
    query: str,
    candidates: List[Dict],
    top_k: int = 5,
    use_quantization: bool = False,
) -> Tuple[List[str], float]:
    model_name     = reranker_config["model"]
    max_length     = reranker_config.get("max_length", 512)
    reranker_label = reranker_config.get("name", model_name)
    batch_size     = reranker_config.get("batch_size", 32 if use_quantization else 16)

    if not candidates:
        logger.warning(f"Reranker '{reranker_label}' received empty candidate list.")
        return [], 0.0

    # Filter out candidates missing es_id before inference — avoids wasting compute
    valid_candidates = [(c, c.get("es_id")) for c in candidates if c.get("es_id")]
    n_filtered = len(candidates) - len(valid_candidates)
    if n_filtered:
        logger.warning(
            f"Reranker '{reranker_label}': {n_filtered} candidate(s) missing 'es_id' "
            "filtered out before inference."
        )
    if not valid_candidates:
        logger.warning(f"Reranker '{reranker_label}': no valid candidates after filtering.")
        return [], 0.0

    candidates_clean, id_map = zip(*valid_candidates)

    start = time.perf_counter()

    try:
        model = _get_model(model_name, max_length, use_quantization)

        pairs: List[Tuple[str, str]] = []
        for cand in candidates_clean:
            question = cand.get("question", "")
            answer   = cand.get("answer", "")
            text     = f"{question} {answer}".strip() or answer
            pairs.append((query, text))

        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        # O(N) top_k selection via argpartition instead of O(NlogN) full sort
        scores_np = np.array(scores)
        if top_k >= len(scores_np):
            top_indices = np.argsort(scores_np)[::-1]
        else:
            top_indices = np.argpartition(scores_np, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores_np[top_indices])[::-1]]

        reranked_ids = [id_map[i] for i in top_indices]

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"Reranker '{reranker_label}' done in {latency_ms:.1f}ms "
            f"({'quantized' if use_quantization else 'fp32'}, "
            f"{len(candidates_clean)} candidates → top {top_k})"
        )
        return reranked_ids, latency_ms

    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.error(
            f"Reranker '{reranker_label}' failed after {latency_ms:.1f}ms: {e}",
            exc_info=True,
        )
        fallback = [c.get("es_id", "") for c in candidates[:top_k]]
        return fallback, latency_ms


def evict_model(
    model_name: str,
    max_length: int = 512,
    use_quantization: bool = False,
) -> None:
    """Evict a specific model from the cache to free memory."""
    cache_key = f"{model_name}:q{int(use_quantization)}:ml{max_length}"
    with _CACHE_LOCK:
        removed = _MODEL_CACHE.pop(cache_key, None)
    if removed:
        logger.info(f"Evicted model '{cache_key}' from cache.")
    else:
        logger.debug(f"evict_model: '{cache_key}' was not in cache.")


def clear_model_cache() -> None:
    """Evict all cached reranker models, e.g. to free memory between benchmark runs."""
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
    logger.info("Reranker model cache cleared.")
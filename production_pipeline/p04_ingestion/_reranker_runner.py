"""
production_pipeline/p04_ingestion/_reranker_runner.py

Handles model loading, caching, and reranking inference.
"""

import time
import threading
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_MODEL_CACHE: Dict[str, CrossEncoder] = {}
_CACHE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model_pytorch(model_name: str, max_length: int, use_quantization: bool) -> CrossEncoder:
    """Load a CrossEncoder in fp32 or with INT8 dynamic quantization."""
    logger.info(
        f"Loading reranker model: {model_name} "
        f"(max_length={max_length}, quantized={use_quantization})"
    )
    model = CrossEncoder(model_name, max_length=max_length, device="cpu")
    if use_quantization:
        logger.info(f"Applying INT8 dynamic quantization to {model_name}...")
        model.model = torch.quantization.quantize_dynamic(
            model.model, {torch.nn.Linear}, dtype=torch.qint8,
        )
        logger.info("Quantization applied.")
    return model


def _load_model_onnx(model_name: str, max_length: int) -> CrossEncoder:
    """
    Load a CrossEncoder backed by ONNX Runtime via Hugging Face Optimum.
    Falls back to PyTorch fp32 if optimum is unavailable or export fails.
    """
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        logger.info(f"Loading ONNX reranker: {model_name} (max_length={max_length})")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            model_name,
            export=True,          # exports to ONNX on first load, cached after
            provider="CPUExecutionProvider",
        )

        # Wrap in CrossEncoder shell so the rest of the pipeline is unchanged
        model = CrossEncoder(model_name, max_length=max_length, device="cpu")
        model.tokenizer = tokenizer
        model.model = ort_model
        logger.info(f"ONNX model ready: {model_name}")
        return model

    except Exception as e:
        logger.warning(
            f"ONNX load failed for '{model_name}': {e}. "
            "Falling back to PyTorch fp32."
        )
        return _load_model_pytorch(model_name, max_length, use_quantization=False)


def _load_model(
    model_name: str,
    max_length: int,
    use_quantization: bool,
    use_onnx: bool,
) -> CrossEncoder:
    """Route to the correct loader based on flags."""
    if use_onnx:
        return _load_model_onnx(model_name, max_length)
    return _load_model_pytorch(model_name, max_length, use_quantization)


def _get_model(
    model_name: str,
    max_length: int,
    use_quantization: bool,
    use_onnx: bool,
) -> CrossEncoder:
    """
    Fetch a model from the thread-safe cache, loading it on first access.
    Cache key includes all flags that affect the loaded artefact.
    """
    cache_key = f"{model_name}:q{int(use_quantization)}:onnx{int(use_onnx)}:ml{max_length}"

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    with _CACHE_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]
        model = _load_model(model_name, max_length, use_quantization, use_onnx)
        _MODEL_CACHE[cache_key] = model
        return model


# ---------------------------------------------------------------------------
# Candidate validation
# ---------------------------------------------------------------------------

def _filter_valid_candidates(
    candidates: List[Dict],
    reranker_label: str,
) -> Tuple[tuple, tuple]:
    """
    Filter candidates missing es_id.
    Returns (candidates_clean, id_map) as tuples.
    Raises ValueError if no valid candidates remain.
    """
    clean, ids = [], []
    for c in candidates:
        es_id = c.get("es_id")
        if es_id:
            clean.append(c)
            ids.append(es_id)

    n_filtered = len(candidates) - len(clean)
    if n_filtered:
        logger.warning(
            f"Reranker '{reranker_label}': {n_filtered} candidate(s) missing "
            "'es_id' filtered out before inference."
        )
    if not clean:
        raise ValueError(f"Reranker '{reranker_label}': no valid candidates after filtering.")

    return tuple(clean), tuple(ids)


# ---------------------------------------------------------------------------
# Pair building
# ---------------------------------------------------------------------------

def _build_pairs(query: str, candidates: tuple) -> List[Tuple[str, str]]:
    """Build (query, candidate_text) pairs for cross-encoder input."""
    pairs = []
    for cand in candidates:
        question = cand.get("question", "")
        answer   = cand.get("answer", "")
        text     = f"{question} {answer}".strip() or answer
        pairs.append((query, text))
    return pairs


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference(
    model: CrossEncoder,
    pairs: List[Tuple[str, str]],
    batch_size: int,
) -> np.ndarray:
    """Run cross-encoder inference and return scores as a numpy array."""
    return model.predict(pairs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)


# ---------------------------------------------------------------------------
# Top-k selection
# ---------------------------------------------------------------------------

def _topk_indices(scores_np: np.ndarray, top_k: int) -> np.ndarray:
    """
    Return indices of top_k scores in descending order.
    Uses O(N) argpartition when top_k < N, O(N log N) argsort otherwise.
    """
    if top_k >= len(scores_np):
        return np.argsort(scores_np)[::-1]
    indices = np.argpartition(scores_np, -top_k)[-top_k:]
    return indices[np.argsort(scores_np[indices])[::-1]]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_reranking(
    reranker_config: dict,
    query: str,
    candidates: List[Dict],
    top_k: int = 5,
    use_quantization: bool = False,
    use_onnx: bool = False,
) -> Tuple[List[str], float]:
    """
    Rerank candidates for a query and return (reranked_es_ids, latency_ms).
    Falls back to original candidate order on any failure.

    Args:
        reranker_config:  Config dict from rerankers.json.
        query:            The search query string.
        candidates:       Retrieved candidate dicts, each must have 'es_id'.
        top_k:            Number of results to return.
        use_quantization: Apply INT8 dynamic quantization (PyTorch path only).
                          Ignored when use_onnx=True.
                          Skipped automatically if config sets "quantization": false.
        use_onnx:         Export and run model via ONNX Runtime instead of PyTorch.
    """
    model_name     = reranker_config["model"]
    max_length     = reranker_config.get("max_length", 512)
    reranker_label = reranker_config.get("name", model_name)

    # Per-model quantization opt-out via config (e.g. ColBERT)
    use_quantization = use_quantization and reranker_config.get("quantization", True)

    batch_size = reranker_config.get("batch_size", 32 if use_quantization else 16)

    if not candidates:
        logger.warning(f"Reranker '{reranker_label}' received empty candidate list.")
        return [], 0.0

    start = time.perf_counter()

    try:
        candidates_clean, id_map = _filter_valid_candidates(candidates, reranker_label)
        model       = _get_model(model_name, max_length, use_quantization, use_onnx)
        pairs       = _build_pairs(query, candidates_clean)
        scores_np   = _run_inference(model, pairs, batch_size)
        top_indices = _topk_indices(scores_np, top_k)

        reranked_ids = [id_map[i] for i in top_indices]
        latency_ms   = (time.perf_counter() - start) * 1000

        mode = "onnx" if use_onnx else ("quantized" if use_quantization else "fp32")
        logger.info(
            f"Reranker '{reranker_label}' done in {latency_ms:.1f}ms "
            f"({mode}, {len(candidates_clean)} candidates → top {top_k})"
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
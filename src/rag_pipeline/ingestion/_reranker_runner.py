"""
production_pipeline/p04_ingestion/_reranker_runner.py

Handles model loading, caching, and reranking inference.
"""
import time
import threading
import logging
import traceback
import numpy as np
import torch
from typing import Any, Dict, List, Tuple, Optional
from sentence_transformers import CrossEncoder
from ._onnx_cross_encoder import ONNXCrossEncoder
logger = logging.getLogger(__name__)
_MODEL_CACHE: Dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()

def _load_model_pytorch(model_name: str, max_length: int, use_quantization: bool) -> CrossEncoder:
    """Load a CrossEncoder in fp32 or with INT8 dynamic quantization."""
    logger.info('Loading reranker model: %r (max_length=%d, quantized=%s)', model_name, max_length, use_quantization)
    model = CrossEncoder(model_name, max_length=max_length, device='cpu')
    if use_quantization:
        logger.info('Applying INT8 dynamic quantization to %r...', model_name)
        model.model = torch.quantization.quantize_dynamic(model.model, {torch.nn.Linear}, dtype=torch.qint8)
        logger.info('Quantization applied to %r.', model_name)
    return model

def _load_model_onnx(model_name: str, max_length: int) -> ONNXCrossEncoder:
    """
    Load a model backed by ONNX Runtime via ONNXCrossEncoder.
    Falls back to PyTorch fp32 if loading fails.
    """
    try:
        return ONNXCrossEncoder(model_name, max_length)
    except Exception as e:
        logger.warning('ONNX load failed for %r: %s. Falling back to PyTorch fp32.\n%s', model_name, e, traceback.format_exc())
        return _load_model_pytorch(model_name, max_length, use_quantization=False)

def _load_model(model_name: str, max_length: int, use_quantization: bool, use_onnx: bool) -> Any:
    """Route to the correct loader based on flags."""
    if use_onnx:
        return _load_model_onnx(model_name, max_length)
    return _load_model_pytorch(model_name, max_length, use_quantization)

def _get_model(model_name: str, max_length: int, use_quantization: bool, use_onnx: bool) -> Any:
    """
    Fetch a model from the thread-safe cache, loading it on first access.
    Cache key encodes every flag that affects the loaded artefact.
    """
    cache_key = f'{model_name}:q{int(use_quantization)}:onnx{int(use_onnx)}:ml{max_length}'
    if cache_key in _MODEL_CACHE:
        logger.debug('_get_model | Cache hit | cache_key=%r', cache_key)
        return _MODEL_CACHE[cache_key]
    with _CACHE_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]
        logger.debug('_get_model | Cache miss — loading | cache_key=%r', cache_key)
        model = _load_model(model_name, max_length, use_quantization, use_onnx)
        _MODEL_CACHE[cache_key] = model
        return model

def _filter_valid_candidates(candidates: List[Dict], reranker_label: str) -> Tuple[tuple, tuple]:
    """
    Filter candidates missing es_id.
    Returns (candidates_clean, id_map) as tuples.
    Raises ValueError if no valid candidates remain.
    """
    clean, ids = ([], [])
    for c in candidates:
        es_id = c.get('es_id')
        if es_id:
            clean.append(c)
            ids.append(es_id)
    n_filtered = len(candidates) - len(clean)
    if n_filtered:
        logger.warning("Reranker %r: %d candidate(s) missing 'es_id' filtered out before inference.", reranker_label, n_filtered)
    if not clean:
        raise ValueError(f"Reranker '{reranker_label}': no valid candidates after filtering.")
    return (tuple(clean), tuple(ids))

def _build_pairs(query: str, candidates: tuple) -> List[Tuple[str, str]]:
    """Build (query, candidate_text) pairs for cross-encoder input."""
    pairs = []
    for cand in candidates:
        question = cand.get('question', '')
        answer = cand.get('answer', '')
        text = f'{question} {answer}'.strip() or answer
        pairs.append((query, text))
    return pairs

def _run_inference(model: Any, pairs: List[Tuple[str, str]], batch_size: int) -> np.ndarray:
    """Run cross-encoder inference and return scores as a numpy array."""
    return model.predict(pairs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)

def _topk_indices(scores_np: np.ndarray, top_k: int) -> np.ndarray:
    """
    Return indices of top_k scores in descending order.
    Uses O(N) argpartition when top_k < N, O(N log N) argsort otherwise.
    """
    if top_k >= len(scores_np):
        return np.argsort(scores_np)[::-1]
    indices = np.argpartition(scores_np, -top_k)[-top_k:]
    return indices[np.argsort(scores_np[indices])[::-1]]

def _resolve_reranker_params(reranker_config: dict, use_quantization: bool, use_onnx: bool) -> Tuple[str, str, int, bool, int]:
    """
    Extract and normalise all scalar parameters from config.

    Returns:
        (model_name, reranker_label, max_length, use_quantization, batch_size)
    """
    model_name = reranker_config['model']
    reranker_label = reranker_config.get('name', model_name)
    max_length = reranker_config.get('max_length', 512)
    use_quantization = use_quantization and reranker_config.get('quantization', True)
    batch_size = reranker_config.get('batch_size', 32 if use_quantization else 16)
    logger.debug('_resolve_reranker_params | label=%r model=%r max_length=%d use_quantization=%s use_onnx=%s batch_size=%d', reranker_label, model_name, max_length, use_quantization, use_onnx, batch_size)
    return (model_name, reranker_label, max_length, use_quantization, batch_size)

def _validate_candidates(candidates: List[Dict], reranker_label: str) -> Optional[Tuple[tuple, tuple]]:
    """
    Guard-clause wrapper around _filter_valid_candidates.

    Returns None (and logs) if the input list is empty; raises ValueError
    if filtering leaves nothing valid.
    """
    if not candidates:
        logger.warning('_validate_candidates | Reranker %r received an empty candidate list.', reranker_label)
        return None
    return _filter_valid_candidates(candidates, reranker_label)

def _score_candidates(model: Any, query: str, candidates_clean: tuple, batch_size: int, reranker_label: str) -> np.ndarray:
    """
    Build input pairs, run inference, and return raw scores.
    """
    pairs = _build_pairs(query, candidates_clean)
    logger.debug('_score_candidates | reranker=%r n_pairs=%d batch_size=%d', reranker_label, len(pairs), batch_size)
    scores = _run_inference(model, pairs, batch_size)
    logger.debug('_score_candidates | reranker=%r scores_shape=%s min=%.4f max=%.4f', reranker_label, scores.shape, float(scores.min()), float(scores.max()))
    return scores

def _select_top_ids(scores_np: np.ndarray, id_map: tuple, top_k: int, reranker_label: str) -> List[str]:
    """
    Apply top-k selection over scores and return the corresponding es_ids.
    """
    top_indices = _topk_indices(scores_np, top_k)
    reranked_ids = [id_map[i] for i in top_indices]
    logger.debug('_select_top_ids | reranker=%r top_k=%d selected_ids=%s', reranker_label, top_k, reranked_ids)
    return reranked_ids

def _build_fallback(candidates: List[Dict], top_k: int, reranker_label: str, exc: Exception, latency_ms: float) -> Tuple[List[str], float]:
    """
    Log the failure and return the first top_k candidates in original order.
    """
    logger.error('_build_fallback | Reranker %r failed after %.1fms — returning original order fallback: %s\n%s', reranker_label, latency_ms, exc, traceback.format_exc())
    fallback = [c.get('es_id', '') for c in candidates[:top_k]]
    return (fallback, latency_ms)

def run_reranking(reranker_config: dict, query: str, candidates: List[Dict], top_k: int=5, use_quantization: bool=False, use_onnx: bool=False) -> Tuple[List[str], float]:
    """
    Rerank candidates for a query and return (reranked_es_ids, latency_ms).
    Falls back to original candidate order on any failure.

    Args:
        reranker_config:  Config dict from rerankers.json.
        query:            The search query string.
        candidates:        Retrieved candidate dicts, each must have 'es_id'.
        top_k:            Number of results to return.
        use_quantization: Apply INT8 dynamic quantization (PyTorch path only).
                          Ignored when use_onnx=True.
                          Skipped automatically if config sets "quantization": false.
        use_onnx:         Export and run model via ONNX Runtime instead of PyTorch.
    """
    model_name, reranker_label, max_length, use_quantization, batch_size = _resolve_reranker_params(reranker_config, use_quantization, use_onnx)
    start = time.perf_counter()
    validated = _validate_candidates(candidates, reranker_label)
    if validated is None:
        return ([], 0.0)
    try:
        candidates_clean, id_map = validated
        model = _get_model(model_name, max_length, use_quantization, use_onnx)
        scores_np = _score_candidates(model, query, candidates_clean, batch_size, reranker_label)
        reranked_ids = _select_top_ids(scores_np, id_map, top_k, reranker_label)
        latency_ms = (time.perf_counter() - start) * 1000
        mode = 'onnx' if use_onnx else 'quantized' if use_quantization else 'fp32'
        logger.info('run_reranking | Reranker %r done in %.1fms (%s, %d candidates → top %d)', reranker_label, latency_ms, mode, len(candidates_clean), top_k)
        return (reranked_ids, latency_ms)
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return _build_fallback(candidates, top_k, reranker_label, exc, latency_ms)
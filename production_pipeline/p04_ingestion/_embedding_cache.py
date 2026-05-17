"""
_embedding_cache.py
===================
Utilities for persisting and loading sentence-embedding arrays to/from disk.

Responsibilities
----------------
- Derive a deterministic cache path from a model short name.
- Load a cached ``.npy`` file, validating shape against the expected corpus size.
- Save an embedding array atomically (write-to-temp + rename) to avoid
  partial files on crash or interrupt.

These functions are pure I/O with no model or Qdrant dependency, so they can
be imported by any pipeline stage that encodes and caches embeddings.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from rag_pipeline.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------

def get_cache_path(cache_dir: Path, model_short_name: str) -> Path:
    """Return the ``.npy`` path for *model_short_name* inside *cache_dir*."""
    return cache_dir / f"{model_short_name}.npy"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_cached_embeddings(
    cache_dir: Path,
    model_short_name: str,
    expected_count: int,
) -> Optional[np.ndarray]:
    """
    Return a cached embedding array, or ``None`` if the cache is absent or stale.

    Parameters
    ----------
    cache_dir:
        Directory that holds ``.npy`` cache files.
    model_short_name:
        Filename stem used when the cache was saved (e.g. ``"bge_base_en_v1_5"``).
    expected_count:
        Number of documents in the current corpus.  If the cache row-count
        differs, the file is considered stale and ``None`` is returned.

    Notes
    -----
    ``allow_pickle=False`` is intentional: pickle-based ``.npy`` files can
    execute arbitrary code on load, so we reject them outright.
    """
    path = get_cache_path(cache_dir, model_short_name)
    if not path.exists():
        return None

    try:
        arr = np.load(path, allow_pickle=False)
    except ValueError as e:
        logger.warning(f"Failed to load cache {path}: {e}")
        return None

    if arr.shape[0] != expected_count:
        logger.warning(
            f"Cache '{path}' has {arr.shape[0]} rows but corpus has "
            f"{expected_count} — ignoring stale cache"
        )
        return None

    logger.info(f"Loaded embeddings from cache: {path} shape={arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_embeddings_cache(
    cache_dir: Path,
    model_short_name: str,
    vectors: np.ndarray,
) -> None:
    """
    Persist *vectors* to ``<cache_dir>/<model_short_name>.npy`` atomically.

    The array is first written to a sibling temp file, then renamed into
    place.  This guarantees that a reader never sees a half-written file,
    even if the process is interrupted mid-write.

    Parameters
    ----------
    cache_dir:
        Directory in which to store the cache file (created if absent).
    model_short_name:
        Filename stem (e.g. ``"bge_base_en_v1_5"``).
    vectors:
        2-D float array of shape ``(n_docs, embedding_dim)``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = get_cache_path(cache_dir, model_short_name)

    with tempfile.NamedTemporaryFile(
        dir=cache_dir,
        prefix=f".tmp_{model_short_name}_",
        suffix=".npy",
        delete=False,
    ) as tmp_file:
        np.save(tmp_file, vectors)
        tmp_path = Path(tmp_file.name)

    # Atomic on POSIX; best-effort on Windows (rename over existing file).
    shutil.move(str(tmp_path), str(path))
    logger.info(f"Saved embeddings cache: {path} shape={vectors.shape}")
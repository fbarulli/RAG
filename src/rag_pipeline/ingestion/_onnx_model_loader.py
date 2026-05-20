""" production_pipeline/p04_ingestion/_onnx_model_loader.py
ONNX Model Loading and Configuration Module
Handles loading and initialization of ONNX models with the Optimum library.
"""
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any, Optional, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from optimum.onnxruntime import ORTModelForSequenceClassification
logger = logging.getLogger(__name__)
CACHE_REQUIRED_FILES = ['model.onnx', 'config.json']
CACHE_TOKENIZER_FILES = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']

class ONNXModelLoader:
    """
    Handles the loading and initialization of ONNX models and tokenizers.
    Provides smart local caching to bypass redundant on-the-fly export operations.
    Ensures consistency between the tokenizer and the exported ONNX graph.
    """

    def __init__(self, model_name: str, provider: str='CPUExecutionProvider', cache_dir: str='experiments/onnx_cache'):
        """
        Initializes the ONNX Model Loader.

        Args:
            model_name: Hugging Face model identifier (e.g., 'BAAI/bge-reranker-base').
            provider: ONNX Runtime provider (e.g., 'CPUExecutionProvider', 'CUDAExecutionProvider').
            cache_dir: Base directory for storing cached ONNX artifacts.
        """
        self.model_name = model_name
        self.provider = provider
        self.cache_dir = Path(cache_dir)
        model_hash = hashlib.md5(model_name.encode()).hexdigest()
        self.local_onnx_path = self.cache_dir / f"{model_hash}_{model_name.replace('/', '_')}"
        export_needed = not self._is_cache_valid()
        self._verify_dependencies(export_needed=export_needed)
        if self._is_cache_valid():
            logger.info('Cache valid. Loading assets from local path: %s', self.local_onnx_path)
            self.model, self.tokenizer = self._load_from_cache()
        else:
            logger.info('Cache miss or invalid. Exporting model from hub: %s', model_name)
            self.model, self.tokenizer = self._export_and_cache()

    def _verify_dependencies(self, export_needed: bool=False) -> None:
        """Assert required packages are present in the environment."""
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import AutoTokenizer
            if export_needed:
                import torch
                logger.debug('Dependencies verified including torch for export.')
            else:
                logger.debug('Dependencies verified for local loading.')
        except ImportError as e:
            logger.error('ONNXModelLoader | Missing dependencies: %s', str(e))
            if export_needed:
                raise ImportError('Export requires PyTorch. Please install: pip install torch optimum[onnxruntime] transformers') from e
            raise ImportError('Missing dependencies. Please install required packages.') from e

    def _is_cache_valid(self) -> bool:
        """Checks if the local cache contains all necessary files."""
        path = self.local_onnx_path
        if not path.exists():
            return False
        for file in CACHE_REQUIRED_FILES:
            if not (path / file).exists():
                logger.debug('Cache missing file: %s', file)
                return False
        has_tokenizer = any(((path / f).exists() for f in CACHE_TOKENIZER_FILES))
        if not has_tokenizer:
            logger.debug('Cache missing tokenizer files')
            return False
        return True

    def _load_from_cache(self) -> Tuple[Any, Any]:
        """Loads model and tokenizer from the local cache directory."""
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
        cache_path_str = str(self.local_onnx_path)
        try:
            model = ORTModelForSequenceClassification.from_pretrained(cache_path_str, provider=self.provider)
            tokenizer = AutoTokenizer.from_pretrained(cache_path_str)
            return (model, tokenizer)
        except Exception as e:
            logger.error('Failed to load from cache %s: %s', cache_path_str, e)
            raise

    def _export_and_cache(self) -> Tuple[Any, Any]:
        """Exports the model from Hugging Face Hub to ONNX and saves to local cache."""
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info('Exporting %s to ONNX format...', self.model_name)
            model = ORTModelForSequenceClassification.from_pretrained(self.model_name, export=True, provider=self.provider)
            self.local_onnx_path.mkdir(parents=True, exist_ok=True)
            logger.info('Saving artifacts to cache: %s', self.local_onnx_path)
            model.save_pretrained(str(self.local_onnx_path))
            tokenizer.save_pretrained(str(self.local_onnx_path))
            return (model, tokenizer)
        except Exception as e:
            logger.error('Export failed for %s: %s', self.model_name, e)
            if self.local_onnx_path.exists():
                logger.warning('Cleaning up partial export directory...')
                shutil.rmtree(self.local_onnx_path, ignore_errors=True)
            raise
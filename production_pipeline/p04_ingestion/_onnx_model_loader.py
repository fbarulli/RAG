"""
production_pipeline/p04_ingestion/_onnx_model_loader.py
ONNX Model Loading and Configuration Module

Handles loading and initialization of ONNX models with the Optimum library.
Provides smart local caching to bypass redundant on-the-fly export operations.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional, Tuple, TYPE_CHECKING

# Use TYPE_CHECKING to avoid importing heavy libraries at module load time
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from optimum.onnxruntime import ORTModelForSequenceClassification

logger = logging.getLogger(__name__)

# Required files to consider a cache entry valid
CACHE_REQUIRED_FILES = ["model.onnx", "config.json", "tokenizer.json"]


class ONNXModelLoader:
    """
    Handles the loading and initialization of ONNX models and tokenizers.
    
    Provides smart local caching to bypass redundant on-the-fly export operations.
    Ensures consistency between the tokenizer and the exported ONNX graph.
    """

    def __init__(
        self, 
        model_name: str, 
        provider: str = "CPUExecutionProvider", 
        cache_dir: str = "experiments/onnx_cache"
    ):
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
        
        # Generate a safe, unique hash for the cache directory to avoid invalid chars
        model_hash = hashlib.md5(model_name.encode()).hexdigest()
        self.local_onnx_path = self.cache_dir / f"{model_hash}_{model_name.replace('/', '_')}"

        # Verify environment readiness
        self._verify_dependencies(export_needed=not self._is_cache_valid())

        # Determine load strategy based on cache validity
        if self._is_cache_valid():
            logger.info("Cache valid. Loading assets from local path: %s", self.local_onnx_path)
            self.model, self.tokenizer = self._load_from_cache()
        else:
            logger.info("Cache miss or invalid. Exporting model from hub: %s", model_name)
            self.model, self.tokenizer = self._export_and_cache()

    def _verify_dependencies(self, export_needed: bool = False) -> None:
        """
        Assert required packages are present in the environment.
        
        Args:
            export_needed: If True, verifies 'torch' is available for export operations.
        """
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import AutoTokenizer
            
            if export_needed:
                import torch  # noqa: F401
                logger.debug("Dependencies verified including torch for export.")
            else:
                logger.debug("Dependencies verified for local loading.")
                
        except ImportError as e:
            missing_pkg = "torch" if "torch" in str(e) else "optimum[onnxruntime] / transformers"
            logger.error("ONNXModelLoader | Missing dependencies: %s", missing_pkg)
            if export_needed:
                raise ImportError(
                    "Export requires PyTorch. Please install: pip install torch optimum[onnxruntime] transformers"
                ) from e
            raise ImportError("Missing dependencies. Please install required packages.") from e

    def _get_cache_path(self) -> Path:
        """Returns the standardized local path for the model cache."""
        return self.local_onnx_path

    def _is_cache_valid(self) -> bool:
        """
        Checks if the local cache contains all necessary files to load the model.
        
        Returns:
            True if all required files exist, False otherwise.
        """
        path = self._get_cache_path()
        if not path.exists():
            return False
        
        # Check for critical artifacts
        for file in CACHE_REQUIRED_FILES:
            if not (path / file).exists():
                logger.debug("Cache missing file: %s", file)
                return False
        return True

    def _load_from_cache(self) -> Tuple[Any, Any]:
        """
        Loads model and tokenizer from the local cache directory.
        
        Returns:
            Tuple of (ORTModel, Tokenizer)
        """
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        cache_path_str = str(self.local_onnx_path)
        
        try:
            model = ORTModelForSequenceClassification.from_pretrained(
                cache_path_str, 
                provider=self.provider
            )
            tokenizer = AutoTokenizer.from_pretrained(cache_path_str)
            return model, tokenizer
        except Exception as e:
            logger.error("Failed to load from cache %s: %s", cache_path_str, e)
            # Invalidate cache if loading fails to force re-export on next run
            logger.warning("Cache loading failed. Cache may be corrupted.")
            raise

    def _export_and_cache(self) -> Tuple[Any, Any]:
        """
        Exports the model from Hugging Face Hub to ONNX and saves to local cache.
        
        Returns:
            Tuple of (ORTModel, Tokenizer)
        """
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        try:
            # 1. Load Tokenizer first (from Hub)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # 2. Export Model (requires torch)
            logger.info("Exporting %s to ONNX format...", self.model_name)
            model = ORTModelForSequenceClassification.from_pretrained(
                self.model_name, 
                export=True, 
                provider=self.provider
            )

            # 3. Save both to local cache to ensure consistency
            self.local_onnx_path.mkdir(parents=True, exist_ok=True)
            
            logger.info("Saving artifacts to cache: %s", self.local_onnx_path)
            model.save_pretrained(str(self.local_onnx_path))
            tokenizer.save_pretrained(str(self.local_onnx_path))
            
            return model, tokenizer

        except Exception as e:
            logger.error("Export failed for %s: %s", self.model_name, e)
            # Clean up partial exports to prevent valid cache detection next time
            if self.local_onnx_path.exists():
                logger.warning("Cleaning up partial export directory...")
                import shutil
                shutil.rmtree(self.local_onnx_path, ignore_errors=True)
            raise

    @property
    def model(self) -> Any:
        """Returns the loaded ORTModel."""
        return self._model

    @model.setter
    def model(self, value: Any):
        self._model = value

    @property
    def tokenizer(self) -> Any:
        """Returns the loaded Tokenizer."""
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: Any):
        self._tokenizer = value

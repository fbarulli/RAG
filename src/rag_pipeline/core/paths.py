"""
RAG Pipeline Path Configuration

This module provides centralized access to all file and directory paths used in the RAG pipeline.
It serves as the single source of truth for path configuration, loading settings from JSON files
in the project's configs/ directory (paths.json, db.json, defaults.json, etc.).

All paths are resolved relative to the project root, which is automatically detected by locating
the pyproject.toml file in the directory hierarchy.

Basic Usage:
    from rag_pipeline.core.paths import Paths

    # Get directory paths
    raw_data_dir = Paths.raw_dir()
    processed_data_dir = Paths.processed_dir()

    # Get file paths
    clean_data = Paths.clean_jsonl()
    config = Paths.defaults()

    # Get derived values
    collection_name = Paths.collection_for_model("BAAI/bge-base-en-v1.5")
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any

class Paths:
    """
    Central path manager for the RAG pipeline.

    This class provides static access to all file and directory paths used throughout
    the project. Paths are loaded from configuration files and resolved relative to the
    project root (found via pyproject.toml).

    All methods are class methods and should be called directly on the class:
        Paths.raw_dir()  # Not Paths().raw_dir()

    Configuration Sources:
    - configs/paths.json: Main path mappings
    - configs/db.json: Database paths
    - configs/defaults.json: Default configuration values
    """

    # Internal state
    _base: Optional[Path] = None
    _config: Optional[Dict[str, Any]] = None

    # ==================== CORE PATH RESOLUTION ====================

    @classmethod
    def base(cls) -> Path:
        """
        Get the project root directory.

        The project root is identified by the presence of pyproject.toml in the
        directory hierarchy starting from this module's location.

        Returns:
            Path: Absolute path to the project root directory.

        Raises:
            RuntimeError: If pyproject.toml cannot be found in any parent directory.
        """
        if cls._base is None:
            current = Path(__file__).resolve()
            for parent in [current] + list(current.parents):
                if (parent / "pyproject.toml").exists():
                    cls._base = parent
                    break
            else:
                raise RuntimeError("Could not find project root (pyproject.toml)")
        return cls._base

    @classmethod
    def _load_config(cls) -> Dict[str, Any]:
        """Load the main paths configuration from configs/paths.json."""
        if cls._config is None:
            config_path = cls.base() / "configs" / "paths.json"
            try:
                with open(config_path, encoding="utf-8") as f:
                    cls._config = json.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load configs/paths.json: {e}")
            if not cls._config:
                raise RuntimeError("configs/paths.json is empty")
        return cls._config

    @classmethod
    def _resolve(cls, key: str) -> Path:
        """Resolve a path from configuration using the given key."""
        return cls.base() / cls._load_config()[key]

    @classmethod
    def _require(cls, path: "Path", hint: str = "") -> "Path":
        """Verify a path exists, raising FileNotFoundError with hint if missing."""
        if not path.exists():
            msg = f"Required path not found: {path}"
            if hint:
                msg += f"\n Hint: {hint}"
            raise FileNotFoundError(msg)
        return path

    # ==================== DATA DIRECTORIES ====================

    @classmethod
    def raw_dir(cls) -> Path:
        """Path to the raw data directory (from paths.json)."""
        return cls._resolve("raw_dir")

    @classmethod
    def processed_dir(cls) -> Path:
        """Path to the processed data directory (from paths.json)."""
        return cls._resolve("processed_dir")

    @classmethod
    def experiments_dir(cls) -> Path:
        """Path to the experiments directory (from paths.json)."""
        return cls._resolve("experiments_dir")

    @classmethod
    def embeddings_cache_dir(cls) -> Path:
        """Path to the embeddings cache directory (from paths.json)."""
        return cls._resolve("embeddings_cache_dir")

    @classmethod
    def ablation_results_dir(cls) -> Path:
        """Path to the ablation results directory (from paths.json)."""
        return cls._resolve("ablation_results_dir")

    # ==================== TOPIC-RELATED PATHS ====================

    @classmethod
    def topics_dir(cls) -> Path:
        """Path to the main topics directory."""
        return cls._resolve("topics_dir")

    @classmethod
    def topics_experiments_dir(cls) -> Path:
        """Path to the topics experiments directory."""
        return cls._resolve("topics_experiments_dir")

    @classmethod
    def topics_output_dir(cls) -> Path:
        """Path to the topics output directory."""
        return cls._resolve("topics_output_dir")

    @classmethod
    def topics_rules_dir(cls) -> Path:
        """Path to the topics rules directory."""
        return cls._resolve("topics_rules_dir")

    @classmethod
    def reranker_results_dir(cls) -> Path:
        """
        Path to the reranker results directory.

        Raises:
            FileNotFoundError: If directory doesn't exist. Run:
                uv run python -m rag_pipeline.eda.topics.core.topic_modeling
        """
        return cls._require(
            cls._resolve("reranker_results_dir"),
            "Run: uv run python -m rag_pipeline.eda.topics.core.topic_modeling"
        )

    @classmethod
    def topic_assignments(cls) -> Path:
        """
        Path to the topic assignments file.

        Raises:
            FileNotFoundError: If file doesn't exist. Run:
                uv run python -m rag_pipeline.eda.topics.core.topic_modeling
        """
        return cls._require(
            cls._resolve("topic_assignments"),
            "Run: uv run python -m rag_pipeline.eda.topics.core.topic_modeling",
        )

    @classmethod
    def topics_default_output(cls) -> Path:
        """Path to the default topics output file."""
        return cls._resolve("topics_default_output")

    @classmethod
    def stopwords_path(cls) -> Path:
        """Path to the stopwords file (used for TF-IDF analysis)."""
        return cls.topics_experiments_dir() / "tfidf_analysis" / "stopwords" / "stopwords_pass2.txt"

    # ==================== DATA FILES ====================

    @classmethod
    def clean_jsonl(cls) -> Path:
        """
        Path to the cleaned JSONL data file.

        Raises:
            FileNotFoundError: If file doesn't exist. Run the cleaning pipeline first.
        """
        p = cls._resolve("clean_jsonl")
        if not p.exists():
            raise FileNotFoundError(
                f"Clean JSONL not found at {p}. Run the cleaning pipeline first."
            )
        return p

    @classmethod
    def test_jsonl(cls) -> Path:
        """Path to the test JSONL file."""
        return cls._resolve("test_jsonl")

    # ==================== CONFIGURATION FILES ====================

    @classmethod
    def configs_dir(cls) -> Path:
        """Path to the main configs directory."""
        return cls._resolve("configs_dir")

    @classmethod
    def entity_patterns(cls) -> Path:
        """
        Path to the entity patterns configuration file.

        Raises:
            FileNotFoundError: If file doesn't exist at expected location.
        """
        return cls._require(
            cls._resolve("entity_patterns"),
            "Expected at configs/entity_patterns.json",
        )

    @classmethod
    def models_config(cls) -> Path:
        """Path to the models configuration file."""
        return cls._resolve("models_config")

    @classmethod
    def service_config(cls) -> Path:
        """Path to the service configuration file (configs/service.json)."""
        return cls.configs_dir() / "service.json"

    @classmethod
    def ablation_config(cls) -> Path:
        """
        Path to the ablation configuration file.

        Raises:
            FileNotFoundError: If configs/ablation_config.json doesn't exist.
        """
        return cls._require(
            cls.base() / "configs" / "ablation_config.json",
            "configs/ablation_config.json"
        )

    # ==================== CONFIGURATION VALUES ====================

    @classmethod
    def defaults(cls) -> dict:
        """
        Get all default configuration values from configs/defaults.json.

        Returns:
            dict: The complete defaults configuration.

        Raises:
            FileNotFoundError: If defaults.json doesn't exist.
        """
        cfg = cls.base() / "configs" / "defaults.json"
        if not cfg.exists():
            raise FileNotFoundError(f"defaults.json not found at {cfg}")
        with open(cfg, encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def topic_modeling_defaults(cls) -> dict:
        """
        Get topic modeling defaults from configs/defaults.json.

        Returns:
            dict: The topic_modeling section of the defaults.

        Raises:
            KeyError: If 'topic_modeling' section is missing from defaults.
        """
        data = cls.defaults()
        if "topic_modeling" not in data:
            raise KeyError("'topic_modeling' section missing from configs/defaults.json")
        return data["topic_modeling"]

    @classmethod
    def data_download_url(cls) -> str:
        """Get the data download URL from defaults.json."""
        return cls.defaults()["data_download_url"]

    @classmethod
    def download_courses(cls) -> list:
        """Get the list of courses to download from defaults.json."""
        return cls.defaults()["download_courses"]

    # ==================== DATABASE PATHS ====================

    @classmethod
    def _load_db_config(cls) -> dict:
        """Load database configuration from configs/db.json."""
        cfg = cls.base() / "configs" / "db.json"
        if not cfg.exists():
            raise FileNotFoundError(f"db.json not found at {cfg}")
        with open(cfg, encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def results_db(cls) -> Path:
        """
        Path to the results database file.

        Creates parent directories if they don't exist.
        """
        p = cls.base() / cls._load_db_config()["results_db"]
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def mlflow_db(cls) -> Path:
        """
        Path to the MLflow database file.

        Creates parent directories if they don't exist.
        """
        p = cls.base() / cls._load_db_config()["mlflow_db"]
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def mlflow_dir(cls) -> Path:
        """
        Path to the MLflow directory (parent of mlflow_db).

        Creates the directory if it doesn't exist.
        """
        p = cls.mlflow_db().parent
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ==================== PIPELINE FILES ====================

    @classmethod
    def input_file(cls, stage: str) -> Path:
        """
        Get the input file path for a pipeline stage.

        Args:
            stage: Name of the pipeline stage (must exist in input_mapping config).

        Returns:
            Path: The input file path for the stage.

        Raises:
            ValueError: If the stage is not configured in input_mapping.
        """
        mapping = cls._load_config().get("input_mapping", {})
        if stage not in mapping:
            raise ValueError(f"Unknown stage: {stage}. Available: {list(mapping)}")
        return cls.base() / mapping[stage]

    @classmethod
    def output_file(cls, stage: str) -> Path:
        """
        Get the output file path for a pipeline stage.

        Args:
            stage: Name of the pipeline stage (must exist in output_mapping config).

        Returns:
            Path: The output file path for the stage.

        Raises:
            ValueError: If the stage is not configured in output_mapping.
        """
        mapping = cls._load_config().get("output_mapping", {})
        if stage not in mapping:
            raise ValueError(f"Unknown stage: {stage}. Available: {list(mapping)}")
        return cls.base() / mapping[stage]

    # ==================== UTILITY METHODS ====================

    @classmethod
    def collection_for_model(cls, model_name: str, encode_mode="question") -> str:
        """
        Generate a Qdrant collection name from a model name.

        This is the single source of truth for collection naming in the project.

        Args:
            model_name: Full model name (e.g., 'BAAI/bge-base-en-v1.5')
            encode_mode: Encoding mode. Can be:
                - A string (default: 'question') - no suffix added
                - An object with a 'suffix' attribute - that suffix will be appended

        Returns:
            str: The collection name (e.g., 'faqs_bge_base_en_v1_5')

        Example:
            >>> Paths.collection_for_model("BAAI/bge-base-en-v1.5")
            'faqs_bge_base_en_v1_5'

            # To include a suffix:
            >>> class QA: suffix = "_qa"
            >>> Paths.collection_for_model("BAAI/bge-base-en-v1.5", QA())
            'faqs_bge_base_en_v1_5_qa'
        """
        short = model_name.split('/')[-1].replace('-', '_').replace('.', '_')
        suffix = encode_mode.suffix if hasattr(encode_mode, "suffix") else ""
        return f'faqs_{short}{suffix}'

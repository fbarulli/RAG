'''
src/rag_pipeline/core/paths.py
'''
import json
from pathlib import Path
from typing import Optional, Dict, Any


class Paths:
    @classmethod
    def _require(cls, path: "Path", hint: str = "") -> "Path":
        """Raise FileNotFoundError with a helpful message if path is missing."""
        if not path.exists():
            msg = f"Required path not found: {path}"
            if hint:
                msg += f"\n  Hint: {hint}"
            raise FileNotFoundError(msg)
        return path


    """Single source of truth — strictly reads from configs/paths.json"""
    _base: Optional[Path] = None
    _config: Optional[Dict[str, Any]] = None

    @classmethod
    def base(cls) -> Path:
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
        return cls.base() / cls._load_config()[key]

    @classmethod
    def raw_dir(cls) -> Path:
        return cls._resolve("raw_dir")

    @classmethod
    def processed_dir(cls) -> Path:
        return cls._resolve("processed_dir")

    @classmethod
    def experiments_dir(cls) -> Path:
        return cls._resolve("experiments_dir")

    @classmethod
    def clean_jsonl(cls) -> Path:
        p = cls._resolve("clean_jsonl")
        if not p.exists():
            raise FileNotFoundError(
                f"Clean JSONL not found at {p}. "
                "Run the cleaning pipeline first."
            )
        return p

    @classmethod
    def test_jsonl(cls) -> Path:
        return cls._resolve("test_jsonl")

    @classmethod
    def topic_assignments(cls) -> Path:
        return cls._require(
            cls._resolve("topic_assignments"),
            "Run: uv run python -m rag_pipeline.eda.topics.core.topic_modeling",
        )

    @classmethod
    def reranker_results_dir(cls) -> Path:
        return cls._require(cls._resolve("reranker_results_dir"), "Run: uv run python -m rag_pipeline.eda.topics.core.topic_modeling")
    
    @classmethod
    def retrieval_configs(cls) -> Path:
        return cls._resolve("retrieval_configs")

    @staticmethod
    def collection_for_model(model_name: str, encode_mode = "question") -> str:
        """Derive Qdrant collection name from model name — single source of truth.
        Single source of truth for collection naming.
        e.g. 'BAAI/bge-base-en-v1.5' -> 'faqs_bge_base_en_v1_5'
        e.g. encode_mode='qa' -> 'faqs_bge_base_en_v1_5_qa'
        """
        short = model_name.split('/')[-1].replace('-', '_').replace('.', '_')
        suffix = encode_mode.suffix if hasattr(encode_mode, "suffix") else ""
        return f'faqs_{short}{suffix}'

    @classmethod
    def input_file(cls, stage: str) -> Path:
        mapping = cls._load_config().get("input_mapping", {})
        if stage not in mapping:
            raise ValueError(f"Unknown stage: {stage}. Available: {list(mapping)}")
        return cls.base() / mapping[stage]

    @classmethod
    def output_file(cls, stage: str) -> Path:
        mapping = cls._load_config().get("output_mapping", {})
        if stage not in mapping:
            raise ValueError(f"Unknown stage: {stage}. Available: {list(mapping)}")
        return cls.base() / mapping[stage]

    @classmethod
    def topics_dir(cls) -> Path:
        return cls._resolve("topics_dir")

    @classmethod
    def topics_experiments_dir(cls) -> Path:
        return cls._resolve("topics_experiments_dir")

    @classmethod
    def topics_output_dir(cls) -> Path:
        return cls._resolve("topics_output_dir")

    @classmethod
    def topics_rules_dir(cls) -> Path:
        return cls._resolve("topics_rules_dir")

    @classmethod
    def entity_patterns(cls) -> Path:
        return cls._require(
            cls._resolve("entity_patterns"),
            "Expected at configs/entity_patterns.json",
        )
    
    @classmethod
    def topics_default_output(cls) -> Path:
        return cls._resolve("topics_default_output")
    
    @classmethod
    def defaults(cls) -> dict:
        """Full contents of configs/defaults.json."""
        cfg = cls.base() / "configs" / "defaults.json"
        if not cfg.exists():
            raise FileNotFoundError(f"defaults.json not found at {cfg}")
        with open(cfg, encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def topic_modeling_defaults(cls) -> dict:
        data = cls.defaults()
        if "topic_modeling" not in data:
            raise KeyError("'topic_modeling' section missing from configs/defaults.json")
        return data["topic_modeling"]

    @classmethod
    def ablation_results_dir(cls) -> Path:
        return cls._resolve("ablation_results_dir")

    @classmethod
    def ablation_config(cls) -> Path:
        return cls._require(cls.base() / "configs" / "ablation_config.json", "configs/ablation_config.json")

    @classmethod
    def stopwords_path(cls) -> Path:
        return cls.topics_experiments_dir() / "tfidf_analysis" / "stopwords" / "stopwords_pass2.txt"

    @classmethod
    def embeddings_cache_dir(cls) -> Path:
        return cls._resolve("embeddings_cache_dir")

    @classmethod
    def models_config(cls) -> Path:
        return cls._resolve("models_config")
    @classmethod
    def configs_dir(cls) -> Path:
        return cls._resolve("configs_dir")

    @classmethod
    def service_config(cls) -> Path:
        return cls.configs_dir() / "service.json"

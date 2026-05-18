"""src/rag_pipeline/paths.py
Paths Management Module.

This module provides centralized path resolution for the RAG pipeline.

Classes:
    Paths: Dynamic project root detection and standard directory helpers.

Methods (Classmethods):
    base(): Find and cache the project root (where pyproject.toml lives).
    _load_defaults(): Load defaults.json from configs directory.
    get(key, default): Get a path from defaults.json or return default.
    raw_dir(): Path to raw data directory.
    processed_dir(): Path to processed data directory.
    experiments_dir(): Path to experiments directory.
    configs_dir(): Path to configs directory.
    embeddings_cache_dir(): Path to embeddings cache directory.
    clean_jsonl(): Path to cleaned FAQ documents.
    test_jsonl(): Path to test set queries.
    models_config(): Path to models.json.
    retrieval_configs(): Path to retrieval_configs.json.
    defaults_json(): Path to defaults.json.
    topic_assignments(): Path to merged topic assignments.
    topic_assignments_for_model(model_name): Path to per-model topic assignments.
    benchmark_results(): Path to benchmark_results.json.
    benchmark_summary(): Path to benchmark_summary.txt.
    benchmark_performance(): Path to benchmark_performance.json.
    benchmark_comparison(): Path to benchmark_comparison.json.
    benchmark_query_results(): Path to benchmark_query_results.json.
    input_file(stage): Default input file for a pipeline stage.
    output_file(stage): Default output file for a pipeline stage.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any


class Paths:
    """Dynamic project root detection + standard directory helpers."""
    
    _base: Optional[Path] = None
    _defaults: Optional[Dict[str, Any]] = None
    
    @classmethod
    def base(cls) -> Path:
        """Find and cache the project root (where pyproject.toml lives)."""
        if cls._base is None:
            current = Path(__file__).resolve()
            for parent in [current] + list(current.parents):
                if (parent / "pyproject.toml").exists():
                    cls._base = parent
                    break
            else:
                raise RuntimeError(
                    "Could not find project root. "
                    "Ensure pyproject.toml exists in your project directory."
                )
        return cls._base
    
    @classmethod
    def _load_defaults(cls) -> Dict[str, Any]:
        """Load defaults.json from configs directory."""
        if cls._defaults is None:
            defaults_path = cls.base() / "configs" / "defaults.json"
            if defaults_path.exists():
                with open(defaults_path) as f:
                    cls._defaults = json.load(f)
            else:
                cls._defaults = {}
        return cls._defaults
    
    @classmethod
    def get(cls, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a path from defaults.json or return default.
        Keys use dot notation: "paths.clean_jsonl"
        """
        defaults = cls._load_defaults()
        
        # Support nested keys like "paths.clean_jsonl"
        if "." in key:
            parts = key.split(".")
            value = defaults
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = None
                    break
            if value:
                return str(cls.base() / value)
        
        # Simple key lookup
        value = defaults.get(key)
        if value:
            return str(cls.base() / value)
        
        return default
    
    # ========== Core Directories ==========
    
    @classmethod
    def raw_dir(cls) -> Path:
        return cls.base() / "production_pipeline" / "p01_data_cleaning" / "data" / "raw"
    
    @classmethod
    def processed_dir(cls) -> Path:
        return cls.base() / "production_pipeline" / "p01_data_cleaning" / "data" / "processed"
    
    @classmethod
    def experiments_dir(cls) -> Path:
        path = cls.get("paths.experiments_dir")
        if path:
            return Path(path)
        return cls.base() / "production_pipeline" / "experiments"
    
    @classmethod
    def configs_dir(cls) -> Path:
        return cls.base() / "configs"
    
    @classmethod
    def embeddings_cache_dir(cls) -> Path:
        path = cls.get("paths.embeddings_cache_dir")
        if path:
            return Path(path)
        return cls.experiments_dir() / "embeddings"
    
    # ========== Input Files ==========
    
    @classmethod
    def clean_jsonl(cls) -> Path:
        """Path to cleaned FAQ documents."""
        path = cls.get("paths.clean_jsonl")
        if path:
            return Path(path)
        return cls.processed_dir() / "clean.jsonl"
    
    @classmethod
    def test_jsonl(cls) -> Path:
        """Path to test set queries."""
        path = cls.get("paths.test_jsonl")
        if path:
            return Path(path)
        return cls.processed_dir() / "test.jsonl"
    
    # ========== Config Files ==========
    
    @classmethod
    def models_config(cls) -> Path:
        """Path to models.json."""
        path = cls.get("paths.models_config")
        if path:
            return Path(path)
        return cls.configs_dir() / "models.json"
    
    @classmethod
    def retrieval_configs(cls) -> Path:
        """Path to retrieval_configs.json."""
        path = cls.get("paths.retrieval_configs")
        if path:
            return Path(path)
        return cls.configs_dir() / "retrieval_configs.json"
    
    @classmethod
    def defaults_json(cls) -> Path:
        """Path to defaults.json."""
        return cls.configs_dir() / "defaults.json"
    
    # ========== EDA / Topic Files ==========
    
    @classmethod
    def topic_assignments(cls) -> Path:
        """Path to merged topic assignments."""
        path = cls.get("paths.topic_assignments")
        if path:
            return Path(path)
        return cls.base() / "production_pipeline" / "p02_eda" / "experiments" / "topic_assignments_all.json"
    
    @classmethod
    def topic_assignments_for_model(cls, model_name: str) -> Path:
        """Path to per-model topic assignments."""
        slug = model_name.replace("/", "_").replace("-", "_")
        return cls.base() / "production_pipeline" / "p02_eda" / "experiments" / f"topic_assignments_{slug}.json"
    
    # ========== Output Files ==========
    
    @classmethod
    def benchmark_results(cls) -> Path:
        """Path to benchmark_results.json."""
        return cls.experiments_dir() / "benchmark_results.json"
    
    @classmethod
    def benchmark_summary(cls) -> Path:
        """Path to benchmark_summary.txt."""
        return cls.experiments_dir() / "benchmark_summary.txt"
    
    @classmethod
    def benchmark_performance(cls) -> Path:
        """Path to benchmark_performance.json."""
        return cls.experiments_dir() / "benchmark_performance.json"
    
    @classmethod
    def benchmark_comparison(cls) -> Path:
        """Path to benchmark_comparison.json."""
        return cls.experiments_dir() / "benchmark_comparison.json"
    
    @classmethod
    def benchmark_query_results(cls) -> Path:
        """Path to benchmark_query_results.json."""
        return cls.experiments_dir() / "benchmark_query_results.json"
    
    # ========== Legacy / Stage-based ==========
    
    @classmethod
    def input_file(cls, stage: str) -> Path:
        """Default input file for a pipeline stage."""
        mapping = {
            "parse": cls.processed_dir() / "parsed.jsonl",
            "dedup": cls.processed_dir() / "parsed.jsonl",
            "eda": cls.clean_jsonl(),
        }
        if stage not in mapping:
            raise ValueError(f"Unknown stage for input: {stage}")
        return mapping[stage]
    
    @classmethod
    def output_file(cls, stage: str) -> Path:
        """Default output file for a pipeline stage."""
        mapping = {
            "parse": cls.processed_dir() / "parsed.jsonl",
            "dedup": cls.clean_jsonl(),
            "eda": cls.experiments_dir() / "eda_summary.json",
        }
        if stage not in mapping:
            raise ValueError(f"Unknown stage for output: {stage}")
        return mapping[stage]
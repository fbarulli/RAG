"""src/rag_pipeline/paths.py
Centralized path resolution — eliminates hardcoded BASE logic across all scripts.
"""
from pathlib import Path
from typing import Optional


class Paths:
    """Dynamic project root detection + standard directory helpers."""
    
    _base: Optional[Path] = None
    
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
    def raw_dir(cls) -> Path:
        return cls.base() / "production_pipeline" / "01_data_cleaning" / "data" / "raw"
    
    @classmethod
    def processed_dir(cls) -> Path:
        return cls.base() / "production_pipeline" / "01_data_cleaning" / "data" / "processed"
    
    @classmethod
    def experiments_dir(cls) -> Path:
        return cls.base() / "production_pipeline" / "experiments"
    
    @classmethod
    def input_file(cls, stage: str) -> Path:
        """Default input file for a pipeline stage."""
        mapping = {
            "parse": cls.processed_dir() / "parsed.jsonl",
            "dedup": cls.processed_dir() / "parsed.jsonl",
            "eda": cls.processed_dir() / "clean.jsonl",
        }
        if stage not in mapping:
            raise ValueError(f"Unknown stage for input: {stage}")
        return mapping[stage]
    
    @classmethod
    def output_file(cls, stage: str) -> Path:
        """Default output file for a pipeline stage."""
        mapping = {
            "parse": cls.processed_dir() / "parsed.jsonl",
            "dedup": cls.processed_dir() / "clean.jsonl",
            "eda": cls.experiments_dir() / "eda_summary.json",
        }
        if stage not in mapping:
            raise ValueError(f"Unknown stage for output: {stage}")
        return mapping[stage]
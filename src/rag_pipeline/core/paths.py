import json
from pathlib import Path
from typing import Optional, Dict, Any

class Paths:
    """Single source of truth — strictly reads from configs/defaults.json"""

    _base: Optional[Path] = None
    _config: Optional[Dict[str, Any]] = None

    @classmethod
    def base(cls) -> Path:
        if cls._base is None:
            current = Path(__file__).resolve()
            for parent in [current] + list(current.parents):
                if (parent / 'pyproject.toml').exists():
                    cls._base = parent
                    break
            else:
                raise RuntimeError("Could not find project root (pyproject.toml)")
        return cls._base

    @classmethod
    def _load_config(cls) -> Dict[str, Any]:
        if cls._config is None:
            config_path = cls.base() / 'configs' / 'defaults.json'
            try:
                with open(config_path, encoding='utf-8') as f:
                    full = json.load(f)
                cls._config = full.get("paths", {})
            except Exception as e:
                raise RuntimeError(f"Failed to load configs/defaults.json: {e}")
            
            if not cls._config:
                raise RuntimeError("No 'paths' section found in defaults.json")
        return cls._config

    @classmethod
    def raw_dir(cls) -> Path:
        return cls.base() / cls._load_config()["raw_dir"]

    @classmethod
    def processed_dir(cls) -> Path:
        return cls.base() / cls._load_config()["processed_dir"]

    @classmethod
    def experiments_dir(cls) -> Path:
        return cls.base() / cls._load_config()["experiments_dir"]

    @classmethod
    def clean_jsonl(cls) -> Path:
        return cls.base() / cls._load_config()["clean_jsonl"]

    @classmethod
    def test_jsonl(cls) -> Path:
        return cls.base() / cls._load_config()["test_jsonl"]

    @classmethod
    def topic_assignments(cls) -> Path:
        return cls.base() / cls._load_config()["topic_assignments"]

    @classmethod
    def input_file(cls, stage: str) -> Path:
        mapping = {
            "parse": cls.processed_dir() / "parsed.jsonl",
            "dedup": cls.processed_dir() / "parsed.jsonl",
        }
        if stage not in mapping:
            raise ValueError(f"Unknown stage: {stage}")
        return mapping[stage]

    @classmethod
    def output_file(cls, stage: str) -> Path:
        mapping = {
            "parse": cls.processed_dir() / "parsed.jsonl",
            "dedup": cls.clean_jsonl(),
        }
        if stage not in mapping:
            raise ValueError(f"Unknown stage: {stage}")
        return mapping[stage]

from pathlib import Path
from typing import List, Dict
import sys

# Ensure project root is in path
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    from rag_pipeline.core.paths import Paths
import json

class TopicsConfig:
    """Topics pipeline configuration — single source of truth via central configs."""
    
    # Paths (centralized)
    BASE_DIR = Paths.topics_dir()
    EXPERIMENTS_DIR = Paths.topics_experiments_dir()
    OUTPUT_DIR = Paths.topics_output_dir()
    RULES_DIR = Paths.topics_rules_dir()
    
    @classmethod
    def _load_models(cls) -> List[Dict]:
        """Load from configs/models.json"""
        models_path = Paths.base() / "configs" / "models.json"
        with open(models_path, encoding="utf-8") as f:
            data = json.load(f)
        return [m for m in data.get("models", []) if m.get("enabled", False)]
    
    @classmethod
    def get_embedding_models(cls) -> List[str]:
        models = cls._load_models()
        tier_order = {"balanced": 0, "fast": 1, "experimental": 2}
        sorted_models = sorted(models, key=lambda m: tier_order.get(m.get("tier"), 99))
        return [m["name"] for m in sorted_models]
    
    DEFAULT_MODEL = None
    
    CLUSTER_CONFIDENCE_THRESHOLD = 0.75
    RULE_OVERRIDE_THRESHOLD = 0.40
    
    CATEGORIES = ["CONCEPT", "ERROR", "ADMIN", "OTHER"]
    
    @classmethod
    def get_output_path(cls, filename: str) -> Path:
        return cls.OUTPUT_DIR / filename

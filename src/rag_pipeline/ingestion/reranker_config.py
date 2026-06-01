"""
Strict Pydantic config — no silent failures
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, ValidationError

class RerankerModelConfig(BaseModel):
    name: str
    model: str
    max_length: int = 512
    reranker: bool = True
    quantization: bool = False

class RerankerConfig(BaseModel):
    models: List[RerankerModelConfig]
    training: Dict[str, Any] = {}

def load_reranker_config() -> RerankerConfig:
    path = Path("configs/rerankers.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        
        config = RerankerConfig.model_validate(data)
        print(f"✅ Loaded {len(config.models)} reranker models")
        return config
    except ValidationError as e:
        print("❌ Validation Error in rerankers.json:")
        print(e)
        raise
    except Exception as e:
        print(f"❌ Failed to load rerankers.json: {e}")
        raise

def get_model_config(model_key: str) -> RerankerModelConfig:
    config = load_reranker_config()
    for m in config.models:
        if m.name == model_key:
            return m
    raise KeyError(f"Model '{model_key}' not found. Available: {[m.name for m in config.models]}")

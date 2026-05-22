"""
rag_pipeline/ingestion/_reranker_config.py
Uses your central Paths class from core
"""

import json
from typing import List, Dict
from ..core.paths import Paths   # Correct import path

def load_reranker_config() -> List[Dict]:
    """Load rerankers.json using the central Paths class"""
    try:
        # Use Paths to resolve the config
        rerankers_path = Paths.base() / "configs" / "rerankers.json"
        
        if rerankers_path.exists():
            with open(rerankers_path, encoding="utf-8") as f:
                data = json.load(f)
            models = data.get("models", [])
            print(f"✅ Loaded {len(models)} reranker models from configs/rerankers.json")
            return models
        else:
            print(f"⚠️ rerankers.json not found at {rerankers_path}")
    except Exception as e:
        print(f"Warning: Could not load rerankers config: {e}")
    
    # Fallback
    print("Using fallback reranker config")
    return [
        {"name": "MiniLM-L6", "model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "max_length": 512}
    ]


def get_model_config(model_key: str) -> Dict:
    """Get config by model name/key (e.g. 'MiniLM-L6', 'bge-reranker-base')"""
    models = load_reranker_config()
    
    for m in models:
        if m.get("name") == model_key or m.get("model") == model_key:
            return m.copy()  # return a copy to avoid accidental mutation
    
    print(f"Warning: Model key '{model_key}' not found → using fallback")
    return {
        "name": model_key,
        "model": model_key,
        "max_length": 512,
        "reranker": True
    }

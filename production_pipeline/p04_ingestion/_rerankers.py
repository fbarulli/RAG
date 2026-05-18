"""
Small dedicated module for reranker configuration.
Keeps _benchmark_config.py clean.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict
from rag_pipeline.paths import Paths
import logging

logger = logging.getLogger(__name__)

def load_rerankers(selected: Optional[str] = None) -> List[Dict]:
    """Load rerankers from configs/rerankers.json"""
    path = Paths.base() / "configs" / "rerankers.json"
    
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return []
    
    with open(path) as f:
        data = json.load(f)
    
    models = data.get("models", [])
    
    if selected:
        for m in models:
            if m.get("name") == selected or m.get("model") == selected:
                return [m]
        logger.warning(f"Reranker '{selected}' not found in config")
        return []
    
    return models


def get_reranker_model(reranker_config: Dict) -> str:
    """Get the actual model name from config entry"""
    return reranker_config.get("model")
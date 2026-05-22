"""
src/rag_pipeline/ingestion/reranking/_reranking_triples.py
Single responsibility: Load train data + manage triples generation/caching.
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from ...core.paths import Paths
from ...core.logging import get_logger
from ..create_training_triples import generate_training_triples

logger = get_logger(__name__)


def load_train_data(sample_size: int = 50) -> List[Dict[str, Any]]:
    train_path: Path = Paths.processed_dir() / "train.jsonl"
    if not train_path.exists():
        logger.error(f"train.jsonl not found: {train_path}")
        sys.exit(1)

    data: List[Dict[str, Any]] = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("question") and item.get("answer"):
                    data.append(item)
            if len(data) >= sample_size:
                break
    return data


def get_or_generate_triples(queries: List[str], corpus: List[Dict], sample_size: int) -> List[Dict]:
    triples_dir = Paths.experiments_dir() / "reranker_training"
    triples_dir.mkdir(parents=True, exist_ok=True)
    triples_path = triples_dir / f"triples_sample_{sample_size}.json"

    if triples_path.exists():
        logger.info(f"Loading cached triples: {triples_path}")
        with open(triples_path, encoding="utf-8") as f:
            return json.load(f)

    logger.info(f"Generating triples (sample={sample_size})...")
    triples = generate_training_triples(
        queries=queries,
        corpus=corpus,
        num_hard_negatives=5,
        model_key="MiniLM-L6",
        max_candidates=30,
        output_path=str(triples_path)
    )
    return triples
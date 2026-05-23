"""
src/rag_pipeline/ingestion/reranking/reranking_triples.py
Single responsibility: Load train data + manage triples generation/caching.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.paths import Paths
from ...core.logging import get_logger
from .create_training_triples import generate_training_triples
from ..reranker_config import load_reranker_config

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


def get_or_generate_triples(
    train_items: List[Dict[str, Any]],
    sample_size: int,
    client: Any,
    collection: str,
    embedding_model: Any,
    retrieval_config: Dict[str, Any],
    topic_map: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    config = load_reranker_config()["training"]
    triples_dir = Paths.experiments_dir() / "reranker_training"
    triples_dir.mkdir(parents=True, exist_ok=True)
    triples_path = triples_dir / f"triples_sample_{sample_size}.json"

    if triples_path.exists():
        logger.info(f"Loading cached triples: {triples_path}")
        with open(triples_path, encoding="utf-8") as f:
            return json.load(f)

    logger.info(f"Generating triples (sample={sample_size})...")
    triples = generate_training_triples(
        train_items=train_items,
        num_hard_negatives=config["num_hard_negatives"],
        max_candidates=config["max_candidates"],
        client=client,
        collection=collection,
        embedding_model=embedding_model,
        retrieval_config=retrieval_config,
        topic_map=topic_map,
        output_path=str(triples_path),
    )
    return triples

"""
rag_pipeline/ingestion/_create_training_triples.py
Generate training triples (query, positive, hard_negatives) for cross-encoder training.
Uses your existing clean reranker.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm

from ._reranker_runner import RerankerRunner
from ..core.paths import Paths

logger = logging.getLogger(__name__)


def generate_training_triples(
    queries: List[str],
    corpus: List[Dict],                    # list of {"id": , "text": , ...}
    num_hard_negatives: int = 5,
    model_key: str = "MiniLM-L6",
    max_candidates: int = 50,              # retrieve more candidates first
    output_path: str = None
) -> List[Dict]:
    """
    Generate training triples using your current best reranker.
    """
    runner = RerankerRunner(model_key=model_key)
    triples = []

    print(f"Generating triples using {model_key} | hard negatives = {num_hard_negatives}")

    for query in tqdm(queries, desc="Generating triples"):
        # Get all candidates
        candidates = [doc["text"] for doc in corpus if doc.get("text")]

        if not candidates:
            continue

        # Rerank all candidates (or top-N if too many)
        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]

        reranked = runner.rerank(query=query, documents=candidates, show_progress=False)

        # Assume first relevant document is positive (you can improve this logic later)
        if len(reranked) == 0:
            continue

        positive_doc = reranked[0][0]   # highest scored document
        hard_negatives = [doc for doc, score in reranked[1:1+num_hard_negatives]]

        triples.append({
            "query": query,
            "positive": positive_doc,
            "hard_negatives": hard_negatives,
            "model_used": model_key
        })

    # Save if path provided
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(triples, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(triples)} triples to {out_path}")

    return triples


# Simple test / CLI helper
if __name__ == "__main__":
    # Example usage
    print("This module is ready. Import and use generate_training_triples()")

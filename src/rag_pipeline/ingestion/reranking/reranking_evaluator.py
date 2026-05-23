"""
src/rag_pipeline/ingestion/reranking/reranking_evaluator.py
Single responsibility: Create evaluator for cross-encoder training.
"""
import random
from typing import Dict, List

from datasets import Dataset
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator


def create_proper_evaluator(
    train_examples: Dataset,
    holdout_fraction: float = 0.2,
):
    """Create a reranking evaluator from a Dataset with query/document/label columns."""
    random.seed(42)
    rows = [train_examples[i] for i in range(len(train_examples))]
    holdout_size = max(10, int(len(rows) * holdout_fraction))
    holdout = random.sample(rows, holdout_size)

    query_map: Dict[str, Dict[str, List[str]]] = {}
    for row in holdout:
        q = row["query"]
        doc = row["document"]
        label = row["label"]
        if q not in query_map:
            query_map[q] = {"positive": [], "negative": []}
        if label == 1.0:
            query_map[q]["positive"].append(doc)
        else:
            query_map[q]["negative"].append(doc)

    eval_data = [
        {"query": q, "positive": v["positive"], "negative": v["negative"]}
        for q, v in query_map.items()
        if v["positive"] and v["negative"]
    ]
    return CrossEncoderRerankingEvaluator(samples=eval_data, name="dev")

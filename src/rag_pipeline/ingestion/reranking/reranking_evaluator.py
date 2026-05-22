"""
src/rag_pipeline/ingestion/reranking/_reranking_evaluator.py
Single responsibility: Create proper reranking evaluator.
"""
import random
from typing import List, Dict

from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator


def create_proper_evaluator(examples: List[List], holdout_fraction: float = 0.2):
    random.seed(42)
    holdout_size = max(10, int(len(examples) * holdout_fraction))
    holdout = random.sample(examples, holdout_size)

    query_map: Dict[str, Dict[str, List[str]]] = {}
    for q, doc, label in holdout:
        if q not in query_map:
            query_map[q] = {"positive": [], "negative": []}
        if label == 1.0:
            query_map[q]["positive"].append(doc)
        else:
            query_map[q]["negative"].append(doc)

    eval_data = [
        {"query": q, "positive": v["positive"], "negative": v["negative"]}
        for q, v in query_map.items() if v["positive"] and v["negative"]
    ]
    return CrossEncoderRerankingEvaluator.from_input_examples(eval_data, name="dev")
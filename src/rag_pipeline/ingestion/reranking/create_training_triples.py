"""
src/rag_pipeline/ingestion/reranking/create_training_triples.py
Generate training triples using ground truth positives from train.jsonl
and entity-boosted Qdrant retrieval for hard negatives.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ..onnx_bench import _generate_query_embedding
from ..benchmark_metrics_data.retrievers import run_entity_boosted_retrieval

logger = logging.getLogger(__name__)


def generate_training_triples(
    train_items: List[Dict[str, Any]],
    num_hard_negatives: int,
    max_candidates: int,
    client: Any,
    collection: str,
    embedding_model: Any,
    retrieval_config: Dict[str, Any],
    topic_map: Optional[Dict[str, Any]] = None,
    output_path: str = None,
) -> List[Dict]:
    """
    Generate training triples using ground truth positives from train.jsonl.
    Uses entity-boosted retrieval if topic_map is provided.
    Hard negatives are top retrieved docs that are NOT the correct document.
    """
    triples = []
    logger.info(
        f"Generating triples | items={len(train_items)} | "
        f"hard_negatives={num_hard_negatives} | entity_boosted={topic_map is not None}"
    )

    for item in tqdm(train_items, desc="Generating triples"):
        query = item.get("question") or item.get("query", "")
        correct_answer = item.get("answer", "")
        correct_id = item.get("id", "")
        course = item.get("course", "")
        if not query or not correct_answer:
            continue

        query_vector = _generate_query_embedding(query, embedding_model)
        if not query_vector:
            continue

        ner_category = None
        ner_primary_entity = None
        topic = None
        if topic_map and correct_id in topic_map:
            entry = topic_map[correct_id]
            ner_category = entry.get("ner_category")
            ner_primary_entity = entry.get("ner_primary_entity")
            topic = entry.get("topic")

        retrieval_result = run_entity_boosted_retrieval(
            client=client,
            collection=collection,
            query_vector=query_vector,
            course_filter=course,
            config=retrieval_config,
            top_k=max_candidates,
            ner_category=ner_category,
            ner_primary_entity=ner_primary_entity,
        )
        if not retrieval_result:
            continue

        hit_ids = getattr(retrieval_result, "hit_ids", ())
        hit_answers = getattr(retrieval_result, "hit_answers", ())

        hard_negatives = [
            hit_answers[i]
            for i in range(min(len(hit_ids), max_candidates))
            if hit_ids[i] != correct_id and i < len(hit_answers) and hit_answers[i]
        ][:num_hard_negatives]

        if not hard_negatives:
            continue

        triples.append({
            "query": query,
            "positive": correct_answer,
            "hard_negatives": hard_negatives,
            "doc_id": correct_id,
            "course": course,
            "topic": topic,
            "ner_category": ner_category,
            "ner_primary_entity": ner_primary_entity,
        })

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(triples, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(triples)} triples to {out_path}")

    return triples

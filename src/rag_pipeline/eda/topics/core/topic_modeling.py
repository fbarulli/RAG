# src/rag_pipeline/eda/topics/core/topic_modeling.py
"""
Runs BERTopic to discover and label topics in cleaned FAQ data.
Supports batch processing of multiple embedding models and auto-skipping completed runs.

Run:
    uv run python -m rag_pipeline.eda.topics.core.topic_modeling --run-all
    uv run python -m rag_pipeline.eda.topics.core.topic_modeling --embedding-model "BAAI/bge-small-en-v1.5"
"""
import json
import gc
from pathlib import Path
from typing import Any, Optional

from rag_pipeline.logging import get_logger
from rag_pipeline.core.paths import Paths
from rag_pipeline.eda.topics.config import TopicsConfig
from rag_pipeline.eda.topics.core.topic_cluster import TopicCluster
from rag_pipeline.eda.topics.classification.tfidf_stopwords import load_stopwords
from rag_pipeline.eda.topics.classification.entity_pattern_learner import (
    build_base_nlp, extract_missed_terms, suggest_patterns, update_entity_ruler,
)
from configs.benchmark_cli import create_topic_modeling_parser
from rag_pipeline.eda.topics.classification.topic_rules import ClassificationRules

logger = get_logger(__name__)


# --- helpers ----------------------------------------------------------------

def _model_output_path(base: Path, model_name: str, run_all: bool) -> Path:
    if not run_all:
        return base
    slug = model_name.replace("/", "_").replace("-", "_")
    return base.parent / f"topic_assignments_{slug}.json"


def _load_documents(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        logger.error("Failed to load documents from %s: %s", path, e)
        raise


def _reassign_outliers(
    topic_model: Any,
    questions: list[str],
    topics: list[int],
    probs: list[float],
) -> tuple[list[int], list[float]]:
    outlier_indices = [i for i, t in enumerate(topics) if t == -1]
    if not outlier_indices:
        return topics, probs
        
    outlier_questions = [questions[i] for i in outlier_indices]
    # approximate_distribution returns (probs, topics)
    topic_distr, _ = topic_model.approximate_distribution(outlier_questions)
    
    for idx, dist in zip(outlier_indices, topic_distr):
        best_topic = int(dist.argmax())
        best_prob = float(dist.max())
        if best_prob >= 0.1:
            topics[idx] = best_topic
            probs[idx] = best_prob
    return topics, probs


def _tag_ner(questions: list[str]) -> dict[str, dict]:
    """Performs NER tagging. Called once per session to avoid redundant computation."""
    logger.info("Starting NER tagging for %d questions...", len(questions))
    nlp = build_base_nlp()
    missed = extract_missed_terms(questions, nlp)
    suggestions = suggest_patterns(missed, min_count=3)
    nlp = update_entity_ruler(nlp, suggestions)
    
    tagged: dict[str, dict] = {}
    for doc in nlp.pipe(questions, batch_size=64):
        ents = list(doc.ents)
        tagged[doc.text] = {
            "category": ents[0].label_ if ents else "OTHER",
            "primary_entity": ents[0].text.lower() if ents else None,
        }
    logger.info("NER tagging complete.")
    return tagged


def _build_assignments(
    docs: list[dict],
    topics: list[int],
    probs: list[float],
    topic_model: Any,
    ner_tagged: dict[str, dict],
) -> list[dict]:
    assignments = []
    for i, doc in enumerate(docs):
        topic_id = topics[i]
        # Clean probability extraction
        p_val = probs[i]
        prob = float(p_val.max()) if hasattr(p_val, "max") else float(p_val)
        
        keywords = topic_model.get_topic(topic_id) if topic_id != -1 else []
        ner = ner_tagged.get(doc["question"], {})
        
        assignments.append({
            "id": doc["id"],
            "course": doc.get("course", "unknown"),
            "section": doc.get("section", "general"),
            "topic": topic_id,
            "topic_probability": prob,
            "question": doc["question"],
            "ner_category": ner.get("category", "OTHER"),
            "ner_primary_entity": ner.get("primary_entity"),
            "subtopic": None,
            "subtopic_keywords": [],
            "keywords": keywords,
        })
    return assignments


def _apply_subtopics(
    assignments: list[dict],
    questions: list[str],
    embeddings,
    subtopic_threshold: int,
    subtopic_min_size: int,
) -> list[dict]:
    from rag_pipeline.eda.topics.core.topic_subtopics import build_subtopics as generate_subtopics
    
    topic_counts = {t: sum(1 for a in assignments if a["topic"] == t) 
                    for t in set(a["topic"] for a in assignments) if t != -1}
    
    large_topics = [t for t, c in topic_counts.items() if c > subtopic_threshold]
    if not large_topics:
        return assignments
        
    logger.info("Found %d topics > %d docs — generating subtopics", len(large_topics), subtopic_threshold)
    subtopic_map = generate_subtopics(assignments, questions, embeddings, subtopic_threshold, subtopic_min_size)
    
    for t_id, sub_info in subtopic_map.items():
        for local in sub_info:
            idx = local["orig_idx"]
            sub_t = local["subtopic_id"]
            kw_text = [w for w, _ in local["keywords"]] if sub_t != -1 else []
            assignments[idx]["subtopic"] = sub_t
            assignments[idx]["subtopic_keywords"] = kw_text
    return assignments


def _build_report(
    model_name: str,
    docs: list[dict],
    topics: list[int],
    assignments: list[dict],
    min_topic_size: int,
    min_samples: int,
) -> dict:
    outlier_count = sum(1 for a in assignments if a["topic"] == -1)
    outlier_ratio = outlier_count / len(assignments) if assignments else 0.0
    
    if outlier_ratio > 0.2:
        logger.warning("High outlier ratio %.1f%% — consider tuning parameters", outlier_ratio * 100)
        
    return {
        "metadata": {
            "model": model_name,
            "total_documents": len(docs),
            "num_topics": len(set(topics) - {-1}),
            "min_topic_size": min_topic_size,
            "min_samples": min_samples,
            "outlier_count": outlier_count,
            "outlier_ratio": round(outlier_ratio, 4),
        },
        "assignments": assignments,
    }


# --- main pipeline function -------------------------------------------------

def process_model(
    model_name: str,
    output_path: Path,
    min_topic_size: int,
    min_samples: int,
    subtopic_threshold: int,
    subtopic_min_size: int,
    input_path: Path,
    ner_tagged: dict[str, dict],
) -> None:
    logger.info("Processing model: %s", model_name)
    docs = _load_documents(input_path)
    questions = [d["question"] for d in docs]
    stopwords = load_stopwords(Paths.stopwords_path())

    clusterer = TopicCluster(model_name)
    topic_model, topics, probs, embeddings = clusterer.run_clustering_raw(
        questions=questions,
        min_topic_size=min_topic_size,
        min_samples=min_samples,
        stopwords=stopwords,
    )

    topics, probs = _reassign_outliers(topic_model, questions, topics, probs)
    assignments = _build_assignments(docs, topics, probs, topic_model, ner_tagged)

    rules = ClassificationRules.load()
    for a in assignments:
        new_cat, source = rules.reclassify(a["ner_category"], a["question"], a["topic"], assignments)
        a["classification_source"] = source
        if source != "unchanged":
            a["ner_category"] = new_cat
            if a.get("ner_primary_entity") is None:
                entity = rules.extract_entity(new_cat, a["question"])
                a["ner_primary_entity"] = entity
            
    if subtopic_threshold > 0:
        assignments = _apply_subtopics(
            assignments, questions, embeddings, subtopic_threshold, subtopic_min_size
        )

    report = _build_report(model_name, docs, topics, assignments, min_topic_size, min_samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    logger.info("Saved %d assignments → %s", len(assignments), output_path)
    
    # Explicit cleanup to prevent CPU memory bloat during --run-all
    del topic_model
    del embeddings
    gc.collect()


# --- entrypoint -------------------------------------------------------------

def main() -> None:
    defaults = Paths.topic_modeling_defaults()
    args = create_topic_modeling_parser().parse_args()

    min_topic_size     = args.min_topic_size     if args.min_topic_size     is not None else defaults["min_topic_size"]
    min_samples        = args.min_samples        if args.min_samples        is not None else defaults["min_samples"]
    subtopic_threshold = args.subtopic_threshold if args.subtopic_threshold is not None else defaults["subtopic_threshold"]
    subtopic_min_size = defaults["subtopic_min_size"]
    input_path = args.input or Paths.input_file("eda")
    base_output = args.output or Paths.topics_default_output()
    embedding_model = args.embedding_model or TopicsConfig.DEFAULT_MODEL

    models_to_run = TopicsConfig.get_embedding_models() if args.run_all else [embedding_model]

    # OPTIMIZATION: Tag NER once for the session, not once per model
    docs = _load_documents(input_path)
    questions = [d["question"] for d in docs]
    ner_tagged = _tag_ner(questions)

    for model in models_to_run:
        target = _model_output_path(base_output, model, args.run_all)
        if args.run_all and target.exists():
            logger.info("Skipping %s — %s already exists", model, target.name)
            continue
            
        process_model(
            model_name=model,
            output_path=target,
            min_topic_size=min_topic_size,
            min_samples=min_samples,
            subtopic_threshold=subtopic_threshold,
            subtopic_min_size=subtopic_min_size,
            input_path=input_path,
            ner_tagged=ner_tagged,
        )


if __name__ == "__main__":
    main()

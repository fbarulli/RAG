"""
p02_topic_modeling.py
=====================
Data-driven topic modeling for FAQ retrieval evaluation.

Uses BERTopic to cluster questions semantically, then creates subtopics
for large clusters to provide fine-grained control for evaluation metrics.

Replaces heuristic intent tagging with learned topic assignments.

Input:  clean.jsonl
Output: experiments/topic_assignments.json
        experiments/bertopic_model/ (saved BERTopic artifact)

Run:    uv run python production_pipeline/p02_eda/p02_topic_modeling.py
        uv run python production_pipeline/p02_eda/p02_topic_modeling.py --min-topic-size 15
"""
import argparse
import sys
from pathlib import Path

from rag_pipeline.paths import Paths
from rag_pipeline.logging import get_logger

from ._topic_loader import load_documents
from ._topic_cluster import cluster_topics
from ._topic_subtopics import build_subtopics
from ._topic_assignments import build_assignments, build_output, attach_subtopics
from ._topic_report import print_full_topic_report, save_results

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_INPUT = Paths.processed_dir() / "clean.jsonl"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_MIN_TOPIC_SIZE = 10
DEFAULT_SUBTOPIC_THRESHOLD = 40
DEFAULT_SUBTOPIC_MIN_SIZE = 3
OUTPUT_DIR = Paths.experiments_dir()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BERTopic clustering on FAQ questions")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSONL path")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL, help="Sentence transformer model name")
    parser.add_argument("--min-topic-size", type=int, default=DEFAULT_MIN_TOPIC_SIZE, help="Minimum documents per topic")
    parser.add_argument("--subtopic-threshold", type=int, default=DEFAULT_SUBTOPIC_THRESHOLD, help="Trigger subtopic generation for topics larger than this")
    parser.add_argument("--subtopic-min-size", type=int, default=DEFAULT_SUBTOPIC_MIN_SIZE, help="Minimum docs per subtopic")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory for results")
    return parser


def main():
    args = _build_parser().parse_args()
    
    try:
        # Step 1: Load and validate documents
        logger.info("Step 1/7: Loading and validating documents")
        docs = load_documents(args.input)
        questions = [d["question"] for d in docs]
        
        # Step 2: Fit BERTopic and get embeddings
        logger.info("Step 2/7: Fitting BERTopic model")
        topic_model, topics, probs, embeddings = cluster_topics(
            questions=questions,
            embedding_model_name=args.embedding_model,
            min_topic_size=args.min_topic_size,
        )
        
        # Step 3: Build base assignments
        logger.info("Step 3/7: Building topic assignments")
        assignments = build_assignments(docs, topics, probs)
        
        # Step 4: Generate subtopics for large parent topics
        logger.info("Step 4/7: Generating subtopics for large topics")
        subtopics = build_subtopics(
            assignments=assignments,
            questions=questions,
            embeddings=embeddings,
            subtopic_threshold=args.subtopic_threshold,
            subtopic_min_size=args.subtopic_min_size,
        )
        
        # Step 5: Attach subtopics to assignments (explicit mutation)
        logger.info("Step 5/7: Attaching subtopics to assignments")
        attach_subtopics(assignments, subtopics)
        
        # Step 6: Compile final output structure
        logger.info("Step 6/7: Compiling output structure")
        results = build_output(
            assignments=assignments,
            topic_model=topic_model,
            embedding_model_name=args.embedding_model,
            min_topic_size=args.min_topic_size,
            subtopic_threshold=args.subtopic_threshold,
            total_docs=len(docs),
        )
        
        # Step 7: Report and save
        logger.info("Step 7/7: Printing report and saving results")
        print_full_topic_report(results)
        save_results(results, args.output_dir, topic_model)
        
        logger.info("Topic modeling complete.")
        
    except Exception as e:
        logger.exception(f"Topic modeling failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
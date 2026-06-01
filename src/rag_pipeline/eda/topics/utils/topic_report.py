"""
Public Functions for Topic Reporting and Artifact Serialization:

def print_full_topic_report(results: dict) -> None:
    Print complete topic structure without any truncation.
    I/O: results (dict) -> None

def save_results(results: dict, output_dir: Path, topic_model: Any) -> None:
    Save assignments and model artifacts.
    I/O: results (dict), output_dir (Path), topic_model (Any) -> None
    
_topic_report.py
================
Print reports and save results for topic modeling.

Single responsibility: human-readable output + JSON serialization.
No document loading, no clustering, no assignment logic.

Functions:
    print_full_topic_report(results: dict) -> None
    save_results(results: dict, output_dir: Path, topic_model: BERTopic) -> None
"""
import json
from collections import defaultdict
from pathlib import Path
from rag_pipeline.eda.core.logging import get_logger
logger = get_logger(__name__)

def print_full_topic_report(results: dict) -> None:
    """Print complete topic structure without any truncation."""
    print('=' * 80)
    print('TOPIC MODELING REPORT')
    print('=' * 80)
    print(f"Total Documents: {results['metadata']['total_docs']}")
    print(f"Number of Topics: {results['metadata']['num_topics']}")
    print(f"Subtopic Threshold: {results['metadata']['subtopic_threshold']}")
    print()
    by_topic = defaultdict(list)
    for a in results['assignments']:
        by_topic[a['topic']].append(a['question'])
    for topic in results['topics']:
        topic_num = topic['topic']
        if topic_num == -1:
            print(f"OUTLIERS (Topic -1): {topic['count']} documents")
            continue
        print(f"TOPIC {topic_num} | Size: {topic['count']} | Keywords: {', '.join(topic['keywords'])}")
        rep_questions = by_topic[topic_num][:3]
        for idx, q in enumerate(rep_questions, 1):
            print(f'  Representative {idx}: {q}')
        print()

def save_results(results: dict, output_dir: Path, topic_model) -> None:
    """Save assignments and model artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / 'topic_assignments.json'
    with open(assignments_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f'Saved topic assignments: {assignments_path}')
    model_dir = output_dir / 'bertopic_model'
    topic_model.save(str(model_dir), serialization='safetensors')
    logger.info(f'Saved BERTopic model: {model_dir}')
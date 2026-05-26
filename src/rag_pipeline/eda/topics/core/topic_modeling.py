"""
Public Functions for Main Topic Modeling Runner Pipeline:

def load_questions(path: Path) -> list[dict]:
    Load cleaned FAQs.
    I/O: path (Path) -> list[dict]

def generate_subtopics(topic_model: Any, questions: list[str], topics: list[int]) -> dict[int, list[dict]]:
    Fit separate BERTopic models per large topic to discover subtopics.
    I/O: topic_model (Any), questions (list[str]), topics (list[int]) -> dict[int, list[dict]]

def process_model(model_name: str, output_path: Path, min_topic_size: int, min_samples: int, subtopic_threshold: int, input_path: Path) -> None:
    Run the full topic modeling pipeline for a single model.
    I/O: model_name (str), output_path (Path), min_topic_size (int), min_samples (int), subtopic_threshold (int), input_path (Path) -> None



topic_modeling.py
=====================
Runs BERTopic to discover and label topics in cleaned FAQ data.
Supports batch processing of multiple embedding models and auto-skipping completed runs.

Output: experiments/topic_assignments_*.json
Run:    uv run python -m rag_pipeline.p02_eda.p02_topic_modeling --run-all
        uv run python -m ... --embedding-model "BAAI/bge-small-en-v1.5"
"""
import argparse
import json
from pathlib import Path
from typing import Any
from rag_pipeline.logging import get_logger
from rag_pipeline.core.paths import Paths
from .topic_cluster import cluster_topics
from rag_pipeline.eda.topics.tfidf_stopwords import load_stopwords
from rag_pipeline.eda.topics.entity_pattern_learner import build_base_nlp, extract_missed_terms, suggest_patterns, update_entity_ruler
logger = get_logger(__name__)
DEFAULT_INPUT = Paths.processed_dir() / 'clean.jsonl'
DEFAULT_OUTPUT = Path(__file__).parent / 'experiments' / 'topic_assignments.json'
DEFAULT_EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5'
DEFAULT_MIN_TOPIC_SIZE = 5
DEFAULT_SUBTOPIC_THRESHOLD = 40
DEFAULT_MIN_SAMPLES = 1
TEST_MODELS = ['BAAI/bge-small-en-v1.5', 'BAAI/bge-base-en-v1.5', 'sentence-transformers/all-mpnet-base-v2', 'nomic-ai/nomic-embed-text-v1.5', 'intfloat/e5-small-v2', 'intfloat/e5-base-v2']

def load_questions(path: Path) -> list[dict]:
    """Load cleaned FAQs."""
    if not path.exists():
        raise FileNotFoundError(f'Input file not found: {path}')
    docs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            docs.append(doc)
    return docs

def generate_subtopics(topic_model: Any, questions: list[str], topics: list[int]) -> dict[int, list[dict]]:
    """Fit separate BERTopic models per large topic to discover subtopics."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    umap_n = getattr(topic_model.umap_model, 'n_neighbors', 15)
    min_size = max(15, umap_n + 5)
    logger.info(f'Dynamic subtopic threshold: {min_size} (based on UMAP n_neighbors={umap_n})')
    unique_topics = set(topics)
    subtopics = {}
    for t in unique_topics:
        if t == -1:
            continue
        indices = [i for i, topic in enumerate(topics) if topic == t]
        if len(indices) < min_size:
            continue
        sub_questions = [questions[i] for i in indices]
        logger.info(f'Generating subtopics for parent topic {t} ({len(sub_questions)} docs)')
        try:
            sub_hdbscan = HDBSCAN(min_cluster_size=3, min_samples=1, prediction_data=True)
            sub_model = BERTopic(embedding_model=topic_model.embedding_model, hdbscan_model=sub_hdbscan, language='english', verbose=False)
            sub_labels, _ = sub_model.fit_transform(sub_questions)
            subtopic_info = []
            for i, sub_t in enumerate(sub_labels):
                subtopic_info.append({'orig_idx': indices[i], 'subtopic_id': sub_t, 'keywords': sub_model.get_topic(sub_t) if sub_t != -1 else []})
            subtopics[t] = subtopic_info
        except Exception as e:
            logger.warning(f'Skipping subtopics for topic {t}: {e}')
            continue
    return subtopics

def process_model(model_name: str, output_path: Path, min_topic_size: int, min_samples: int, subtopic_threshold: int, input_path: Path) -> None:
    """Run the full topic modeling pipeline for a single model."""
    logger.info(f'Processing model: {model_name}')
    docs = load_questions(input_path)
    questions = [d['question'] for d in docs]
    stopwords = load_stopwords(Path('rag_pipeline/eda/topics/experiments/tfidf_analysis/stopwords/stopwords_pass2.txt'))
    topic_model, topics, probs, embeddings = cluster_topics(questions=questions, embedding_model_name=model_name, min_topic_size=min_topic_size, stopwords=stopwords, min_samples=min_samples)
    topics = list(topics)
    outlier_indices = [i for i, t in enumerate(topics) if t == -1]
    if outlier_indices:
        outlier_questions = [questions[i] for i in outlier_indices]
        topic_distr, _ = topic_model.approximate_distribution(outlier_questions)
        for idx, dist in zip(outlier_indices, topic_distr):
            best_topic = int(dist.argmax())
            best_prob = float(dist.max())
            if best_prob >= 0.1:
                topics[idx] = best_topic
                probs[idx] = best_prob
    nlp = build_base_nlp()
    missed = extract_missed_terms(questions, nlp)
    suggestions = suggest_patterns(missed, min_count=3)
    nlp = update_entity_ruler(nlp, suggestions)
    ner_tagged = {}
    for doc in nlp.pipe(questions, batch_size=64):
        ents = list(doc.ents)
        ner_tagged[doc.text] = {'category': ents[0].label_ if ents else 'OTHER', 'primary_entity': ents[0].text.lower() if ents else None}
    assignments = []
    for i, doc in enumerate(docs):
        topic_id = topics[i]
        prob = float(probs[i]) if i < len(probs) else 0.0
        keywords = topic_model.get_topic(topic_id) if topic_id != -1 else []
        assignments.append({'id': doc['id'], 'course': doc.get('course', 'unknown'), 'section': doc.get('section', 'general'), 'topic': topic_id, 'topic_probability': prob, 'question': doc['question'], 'ner_category': ner_tagged.get(doc['question'], {}).get('category', 'OTHER'), 'ner_primary_entity': ner_tagged.get(doc['question'], {}).get('primary_entity'), 'subtopic': None, 'subtopic_keywords': [], 'keywords': keywords})
    outlier_count = sum((1 for a in assignments if a['topic'] == -1))
    outlier_ratio = outlier_count / len(assignments) if assignments else 0.0
    logger.info(f'Outlier stats: {outlier_count}/{len(assignments)} ({outlier_ratio:.1%}) assigned to topic -1')
    if outlier_ratio > 0.2:
        logger.warning('High outlier ratio — consider tuning parameters')
    if subtopic_threshold > 0:
        topic_counts = {t: topics.count(t) for t in set(topics) if t != -1}
        large_topics = [t for t, c in topic_counts.items() if c > subtopic_threshold]
        if large_topics:
            logger.info(f'Found {len(large_topics)} topics > {subtopic_threshold} docs. Generating subtopics...')
            subtopic_map = generate_subtopics(topic_model, questions, topics)
            for t_id, sub_info in subtopic_map.items():
                for local in sub_info:
                    orig_idx = local['orig_idx']
                    sub_topic_id = local['subtopic_id']
                    kw_text = [w for w, _ in local['keywords']] if sub_topic_id != -1 else []
                    assignments[orig_idx]['subtopic'] = sub_topic_id
                    assignments[orig_idx]['subtopic_keywords'] = kw_text
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {'metadata': {'model': model_name, 'total_documents': len(docs), 'num_topics': len(set(topics) - {-1}), 'min_topic_size': min_topic_size, 'min_samples': min_samples, 'outlier_count': outlier_count, 'outlier_ratio': round(outlier_ratio, 4)}, 'assignments': assignments}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    logger.info(f'Saved {len(assignments)} topic assignments to {output_path}')

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run BERTopic on cleaned FAQs')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--embedding-model', type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument('--min-topic-size', type=int, default=DEFAULT_MIN_TOPIC_SIZE)
    parser.add_argument('--min-samples', type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument('--subtopic-threshold', type=int, default=DEFAULT_SUBTOPIC_THRESHOLD)
    parser.add_argument('--run-all', action='store_true', help='Run for all test models, skipping existing files.')
    return parser

def main() -> None:
    args = _build_parser().parse_args()
    models_to_run = TEST_MODELS if args.run_all else [args.embedding_model]
    for model in models_to_run:
        slug = model.replace('/', '_').replace('-', '_')
        target_output = args.output.parent / f'topic_assignments_{slug}.json' if args.run_all else args.output
        if args.run_all and target_output.exists():
            logger.info(f'Skipping {model}: {target_output.name} already exists.')
            continue
        process_model(model_name=model, output_path=target_output, min_topic_size=args.min_topic_size, min_samples=args.min_samples, subtopic_threshold=args.subtopic_threshold, input_path=args.input)
if __name__ == '__main__':
    main()
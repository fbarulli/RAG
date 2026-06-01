"""
Public Functions for Embedding Model Comparison and Performance Profiling:

def load_questions(path: Path) -> list[str]:
    Load questions from JSONL file.
    I/O: path (Path) -> list[str]

def get_embedding_cache_path(model_name: str, questions: list[str], cache_dir: Path) -> Path:
    Generate deterministic cache path based on model + questions hash.
    I/O: model_name (str), questions (list[str]), cache_dir (Path) -> Path

def load_or_compute_embeddings(questions: list[str], model_name: str, cache_dir: Path, use_cache: bool = True) -> tuple[np.ndarray, SentenceTransformer]:
    Load cached embeddings or compute + save them. Returns both the array and the loaded model.
    I/O: questions (list[str]), model_name (str), cache_dir (Path), use_cache (bool) -> tuple[np.ndarray, SentenceTransformer]

def compute_cluster_quality(embeddings: np.ndarray, topics: list[int]) -> dict:
    Compute internal clustering validation metrics.
    I/O: embeddings (np.ndarray), topics (list[int]) -> dict

def run_clustering(questions: list[str], model_name: str, min_topic_size: int, min_samples: int, cache_dir: Path, use_cache: bool = True) -> dict:
    Run BERTopic and return comprehensive metrics.
    I/O: questions (list[str]), model_name (str), min_topic_size (int), min_samples (int), cache_dir (Path), use_cache (bool) -> dict

p05_embedding_comparison.py
===========================
Compares topic modeling results across embedding models.

Run one model at a time:
    uv run python -m rag_pipeline.p02_eda.p05_embedding_comparison --model BAAI/bge-base-en-v1.5

Or loop in shell:
    for m in "BAAI/bge-small-en-v1.5" "BAAI/bge-base-en-v1.5" "sentence-transformers/all-mpnet-base-v2"; do
        uv run python -m rag_pipeline.p02_eda.p05_embedding_comparison --model "$m"
    done
"""
import argparse
import hashlib
import itertools
import json
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Optional
import numpy as np
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from rag_pipeline.eda.core.logging import get_logger
from rag_pipeline.eda.core.paths import Paths
logger = get_logger(__name__)
DEFAULT_INPUT = Paths.processed_dir() / 'clean.jsonl'
DEFAULT_OUTPUT_DIR = Paths.experiments_dir() / 'embedding_comparison'
DEFAULT_CACHE_DIR = Paths.experiments_dir() / 'embedding_cache'
DEFAULT_MIN_TOPIC_SIZE = 5
DEFAULT_MIN_SAMPLES = 1
TEST_MODELS = ['BAAI/bge-small-en-v1.5', 'BAAI/bge-base-en-v1.5', 'sentence-transformers/all-mpnet-base-v2', 'nomic-ai/nomic-embed-text-v1.5', 'intfloat/e5-small-v2']

def load_questions(path: Path) -> list[str]:
    """Load questions from JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f'Input file not found: {path}')
    questions = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            questions.append(doc['question'])
    return questions

def get_embedding_cache_path(model_name: str, questions: list[str], cache_dir: Path) -> Path:
    """Generate deterministic cache path based on model + questions hash."""
    h = hashlib.md5()
    for q in sorted(questions):
        h.update(q.encode('utf-8'))
        h.update(b'|||')
    questions_hash = h.hexdigest()[:12]
    model_slug = model_name.replace('/', '_').replace('-', '_')
    return cache_dir / f'embeddings_{model_slug}_{questions_hash}.npy'

def load_or_compute_embeddings(questions: list[str], model_name: str, cache_dir: Path, use_cache: bool=True) -> tuple[np.ndarray, SentenceTransformer]:
    """Load cached embeddings or compute + save them. Returns both the array and the loaded model."""
    logger.info(f'Loading model: {model_name}...')
    embedder = SentenceTransformer(model_name, trust_remote_code=True)
    if use_cache:
        cache_path = get_embedding_cache_path(model_name, questions, cache_dir)
        if cache_path.exists():
            logger.info(f'Loading cached embeddings: {cache_path}')
            return (np.load(cache_path), embedder)
    logger.info(f'Computing embeddings with {model_name}...')
    embeddings = embedder.encode(questions, convert_to_numpy=True, batch_size=32, show_progress_bar=True)
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = get_embedding_cache_path(model_name, questions, cache_dir)
        np.save(cache_path, embeddings)
        logger.info(f'Cached embeddings: {cache_path}')
    return (embeddings, embedder)

def compute_cluster_quality(embeddings: np.ndarray, topics: list[int]) -> dict:
    """Compute internal clustering validation metrics."""
    mask = [t != -1 for t in topics]
    if sum(mask) < 2:
        return {'silhouette_score': None, 'davies_bouldin_index': None}
    emb_filtered = embeddings[mask]
    topics_filtered = np.array([t for t, m in zip(topics, mask) if m])
    if len(set(topics_filtered)) < 2:
        return {'silhouette_score': None, 'davies_bouldin_index': None}
    sil = silhouette_score(emb_filtered, topics_filtered, metric='cosine')
    db = davies_bouldin_score(emb_filtered, topics_filtered)
    return {'silhouette_score': round(float(sil), 4), 'davies_bouldin_index': round(float(db), 4)}

def _compute_keyword_overlap(topic_keywords: dict[int, set[str]]) -> Optional[float]:
    """Average pairwise Jaccard similarity across topic keyword sets."""
    ids = list(topic_keywords.keys())
    if len(ids) < 2:
        return None
    if len(ids) > 100:
        logger.warning(f'Skipping keyword overlap for {len(ids)} topics (>100, too expensive)')
        return None
    pairs, total = (0, 0.0)
    for id_a, id_b in itertools.combinations(ids, 2):
        a, b = (topic_keywords[id_a], topic_keywords[id_b])
        if a or b:
            total += len(a & b) / len(a | b)
        pairs += 1
    return round(total / pairs, 4) if pairs else None

def run_clustering(questions: list[str], model_name: str, min_topic_size: int, min_samples: int, cache_dir: Path, use_cache: bool=True) -> dict:
    """Run BERTopic and return comprehensive metrics."""
    start_time = time.time()
    tracemalloc.start()
    embeddings, embedder = load_or_compute_embeddings(questions, model_name, cache_dir, use_cache)
    hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size, min_samples=min_samples, prediction_data=True, metric='euclidean')
    topic_model = BERTopic(embedding_model=embedder, hdbscan_model=hdbscan_model, ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True), verbose=False)
    logger.info(f'Clustering with {model_name}...')
    topics, probs = topic_model.fit_transform(questions, embeddings=embeddings)
    total = len(topics)
    outliers = sum((1 for t in topics if t == -1))
    outlier_ratio = outliers / total if total > 0 else 0.0
    real_topics = [t for t in topics if t != -1]
    topic_counts = Counter(real_topics)
    topic_keywords = {}
    for t in set(real_topics):
        topic_info = topic_model.get_topic(t)
        words = [w for w, _ in topic_info] if topic_info else []
        topic_keywords[t] = set((w.lower() for w in words[:10]))
    keyword_overlap = _compute_keyword_overlap(topic_keywords)
    valid_probs = [p for p in probs if p is not None and (not np.isnan(p))]
    mean_conf = mean(valid_probs) if valid_probs else 0.0
    low_conf_count = sum((1 for p in valid_probs if p < 0.5))
    quality = compute_cluster_quality(embeddings, topics)
    sizes = list(topic_counts.values()) if topic_counts else []
    size_stats = {'min': min(sizes) if sizes else 0, 'max': max(sizes) if sizes else 0, 'mean': round(mean(sizes), 1) if sizes else 0.0, 'median': float(np.median(sizes)) if sizes else 0.0, 'std': round(stdev(sizes), 1) if len(sizes) > 1 else 0.0}
    elapsed = time.time() - start_time
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem_bytes / 1024 ** 2
    return {'model': model_name, 'total_documents': total, 'num_topics': len(topic_counts), 'outlier_count': outliers, 'outlier_ratio': round(outlier_ratio, 4), 'avg_keyword_overlap': round(keyword_overlap, 4) if keyword_overlap else None, 'mean_confidence': round(mean_conf, 4), 'low_confidence_count': low_conf_count, 'low_confidence_ratio': round(low_conf_count / len(valid_probs), 4) if valid_probs else None, 'topic_size_stats': size_stats, 'silhouette_score': quality['silhouette_score'], 'davies_bouldin_index': quality['davies_bouldin_index'], 'runtime_seconds': round(elapsed, 1), 'peak_memory_mb': round(peak_mem_mb, 1), 'embedding_dim': embeddings.shape[1] if len(embeddings.shape) > 1 else None}

def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    if not results:
        return
    print('\n' + '=' * 100)
    print('EMBEDDING MODEL COMPARISON SUMMARY')
    print('=' * 100)
    print(f"{'Model':<45} {'Outliers':>10} {'Topics':>8} {'Silhouette':>12} {'Runtime':>10} {'Memory':>10}")
    print('-' * 100)
    for r in sorted(results, key=lambda x: x['outlier_ratio']):
        model_short = r['model'].split('/')[-1] if '/' in r['model'] else r['model']
        print(f"{model_short:<45} {r['outlier_ratio']:>9.1%} {r['num_topics']:>8} {r['silhouette_score'] or 'N/A':>12} {r['runtime_seconds']:>9.1f}s {r['peak_memory_mb']:>9.0f}MB")
    print('=' * 100)
    best_outliers = min(results, key=lambda x: x['outlier_ratio'])
    best_silhouette = max((r for r in results if r['silhouette_score']), key=lambda x: x['silhouette_score'], default=None)
    print(f"\n✓ Best outlier ratio: {best_outliers['model'].split('/')[-1]} ({best_outliers['outlier_ratio']:.1%})")
    if best_silhouette:
        print(f"✓ Best cluster separation: {best_silhouette['model'].split('/')[-1]} (silhouette={best_silhouette['silhouette_score']:.3f})")
    print('=' * 100 + '\n')

def main() -> None:
    parser = argparse.ArgumentParser(description='Compare embedding models for topic modeling')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--model', type=str, help='Embedding model to test')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument('--min-topic-size', type=int, default=DEFAULT_MIN_TOPIC_SIZE)
    parser.add_argument('--min-samples', type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument('--no-cache', action='store_true', help='Disable embedding cache')
    parser.add_argument('--compare-all', action='store_true', help='Run comparison across all TEST_MODELS')
    args = parser.parse_args()
    if not args.compare_all and (not args.model):
        parser.error('You must specify either --model or --compare-all')
    use_cache = not args.no_cache
    if args.compare_all:
        models_to_test = TEST_MODELS
        logger.info(f'Running comparison across {len(models_to_test)} models')
    else:
        models_to_test = [args.model]
    logger.info(f'Loading questions from {args.input}')
    questions = load_questions(args.input)
    logger.info(f'Loaded {len(questions)} questions')
    all_results = []
    for model_name in models_to_test:
        logger.info(f'Running clustering with {model_name}')
        metrics = run_clustering(questions=questions, model_name=model_name, min_topic_size=args.min_topic_size, min_samples=args.min_samples, cache_dir=args.cache_dir, use_cache=use_cache)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        model_slug = model_name.replace('/', '_').replace('-', '_')
        output_path = args.output_dir / f'{model_slug}.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f'Saved metrics: {output_path}')
        print(f"\n{'=' * 80}")
        print(f'MODEL: {model_name}')
        print(f"{'=' * 80}")
        print(f"  Outlier ratio:     {metrics['outlier_ratio']:.1%} ({metrics['outlier_count']}/{metrics['total_documents']})")
        print(f"  Num topics:        {metrics['num_topics']}")
        print(f"  Keyword overlap:   {metrics['avg_keyword_overlap'] or 'N/A':>6}  (<0.10 good)")
        print(f"  Mean confidence:   {metrics['mean_confidence']:.3f}")
        print(f"  Silhouette score:  {metrics['silhouette_score'] or 'N/A':>6}  (higher=better)")
        print(f"  Davies-Bouldin:    {metrics['davies_bouldin_index'] or 'N/A':>6}  (lower=better)")
        print(f"  Embedding dim:     {metrics['embedding_dim']}")
        print(f"  Runtime:           {metrics['runtime_seconds']:.1f}s")
        print(f"  Peak memory:       {metrics['peak_memory_mb']:.0f}MB")
        print(f"{'=' * 80}")
        all_results.append(metrics)
    if len(all_results) > 1:
        print_comparison_table(all_results)
        combined_path = args.output_dir / 'comparison_summary.json'
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f'Saved comparison summary: {combined_path}')
if __name__ == '__main__':
    main()
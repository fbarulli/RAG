"""
p04_stratified_test_split.py
============================
Creates a stratified holdout test set that preserves course/section/topic distribution.
Handles severe class imbalance with min/max per-group constraints.

Output: data/processed/test.jsonl
Run:    uv run python -m rag_pipeline.cleaning.p04_stratified_test_split
        uv run python -m ... --n 280 --stratify-by course topic --exclude-outliers
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional
from rag_pipeline.core.logging import get_logger
from rag_pipeline.core.paths import Paths
logger = get_logger(__name__)
DEFAULT_INPUT = Paths.processed_dir() / 'clean.jsonl'
DEFAULT_OUTPUT = Paths.processed_dir() / 'test.jsonl'
DEFAULT_N = 280
DEFAULT_STRATIFY_BY = ['course']
DEFAULT_SEED = 42
DEFAULT_TOPIC_ASSIGNMENTS = Path('../../experiments/topic_assignments_all.json')

def load_documents(path: Path) -> list[dict]:
    """Load documents from JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f'Input not found: {path}')
    docs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            docs.append(json.loads(line))
    return docs

def load_documents_with_topics(clean_path: Path, topic_assignments_path: Path, model: str='BAAI/bge-base-en-v1.5') -> list[dict]:
    """Load clean docs and join with topic assignments."""
    assignments = json.load(open(topic_assignments_path))
    topic_map = {a['id']: a for a in assignments['results'][model]['assignments']}
    docs = []
    for line in open(clean_path, encoding='utf-8'):
        if not line.strip():
            continue
        doc = json.loads(line)
        topic_info = topic_map.get(doc['id'], {})
        doc['topic'] = topic_info.get('topic', -1)
        doc['ner_category'] = topic_info.get('ner_category', 'OTHER')
        doc['ner_primary_entity'] = topic_info.get('ner_primary_entity')
        docs.append(doc)
    return docs

def _make_composite_key(doc: dict, keys: list[str]) -> str:
    """Create a composite stratification key from multiple fields."""
    return '::'.join((str(doc.get(k, 'unknown')) for k in keys))

def stratified_sample_imbalanced(docs: list[dict], n: int, stratify_by: list[str], seed: int, min_per_group: int=3, max_per_group: Optional[int]=None, exclude_outliers: bool=False) -> list[dict]:
    """
    Sample exactly n documents while preserving proportional group distribution,
    with safeguards for severe class imbalance.
    
    Args:
        docs: List of document dicts
        n: Target sample size
        stratify_by: List of keys to stratify by (e.g., ["course", "topic"])
        seed: Random seed for reproducibility
        min_per_group: Minimum samples per group (prevents small groups from being excluded)
        max_per_group: Maximum samples per group (prevents large groups from dominating)
        exclude_outliers: If True, exclude docs with topic == -1 from sampling pool
    """
    random.seed(seed)
    if exclude_outliers:
        original_count = len(docs)
        docs = [d for d in docs if d.get('topic', -1) != -1]
        logger.info(f'Excluded {original_count - len(docs)} outlier docs (topic -1)')
    if len(docs) < n:
        raise ValueError(f'Requested {n} test docs, but only {len(docs)} available after filtering')
    groups: dict[str, list[dict]] = defaultdict(list)
    for doc in docs:
        key = _make_composite_key(doc, stratify_by)
        groups[key].append(doc)
    total = len(docs)
    raw_allocation = {k: len(v) / total * n for k, v in groups.items()}
    allocation = {}
    for key, pool in groups.items():
        target = raw_allocation[key]
        count = max(min_per_group, min(target, len(pool)))
        if max_per_group is not None:
            count = min(count, max_per_group)
        allocation[key] = int(count)
    remainder = n - sum(allocation.values())
    if remainder > 0:
        sorted_keys = sorted(groups.keys(), key=lambda k: len(groups[k]), reverse=True)
        for key in sorted_keys:
            if remainder <= 0:
                break
            current = allocation[key]
            pool_size = len(groups[key])
            if current < pool_size and (max_per_group is None or current < max_per_group):
                allocation[key] += 1
                remainder -= 1
    if remainder > 0:
        logger.warning(f'Could not allocate all {n} samples; short by {remainder}. Consider increasing min_per_group or reducing max_per_group.')
    sampled = []
    warnings = []
    for key, count in allocation.items():
        pool = groups[key]
        if count >= len(pool):
            sampled.extend(pool)
            if count > len(pool):
                warnings.append(f"Group '{key}': requested {count}, using all {len(pool)}")
        else:
            sampled.extend(random.sample(pool, count))
    for w in warnings:
        logger.warning(w)
    random.shuffle(sampled)
    return sampled

def compute_stratification_metrics(full_docs: list[dict], test_docs: list[dict], stratify_by: list[str]) -> dict:
    """Compute metrics to verify stratification quality."""
    from scipy.stats import chisquare
    import numpy as np

    def count_by_key(docs: list[dict]) -> Counter:
        return Counter((_make_composite_key(d, stratify_by) for d in docs))
    full_counts = count_by_key(full_docs)
    test_counts = count_by_key(test_docs)
    all_keys = sorted(set(full_counts) | set(test_counts))
    expected = np.array([full_counts.get(k, 0) for k in all_keys], dtype=float)
    observed = np.array([test_counts.get(k, 0) for k in all_keys], dtype=float)
    scale = len(test_docs) / len(full_docs)
    expected_scaled = expected * scale
    mask = expected_scaled > 0
    if mask.sum() >= 2:
        chi2, p_value = chisquare(observed[mask], f_exp=expected_scaled[mask])
    else:
        chi2, p_value = (None, None)
    deviations = []
    for key in all_keys:
        full_pct = full_counts.get(key, 0) / len(full_docs) * 100
        test_pct = test_counts.get(key, 0) / len(test_docs) * 100
        deviations.append(abs(full_pct - test_pct))
    return {'chi2_statistic': round(float(chi2), 3) if chi2 is not None else None, 'p_value': round(float(p_value), 4) if p_value is not None else None, 'max_deviation_pct': round(max(deviations), 2), 'mean_deviation_pct': round(np.mean(deviations), 2), 'groups_matched': sum((1 for k in all_keys if full_counts.get(k, 0) > 0 and test_counts.get(k, 0) > 0)), 'total_groups': len(all_keys)}

def print_distribution_comparison(full_docs: list[dict], test_docs: list[dict], stratify_by: list[str]) -> None:
    """Print a formatted comparison of distributions."""
    full_counts = Counter((_make_composite_key(d, stratify_by) for d in full_docs))
    test_counts = Counter((_make_composite_key(d, stratify_by) for d in test_docs))
    print(f"\n{'=' * 80}")
    print(f'STRATIFICATION COMPARISON (by {stratify_by})')
    print(f"{'=' * 80}")
    print(f"{'Group':<50} {'Full %':>10} {'Test %':>10} {'Diff':>8}")
    print(f"{'-' * 80}")
    all_keys = sorted(set(full_counts) | set(test_counts))
    for key in all_keys[:20]:
        full_pct = full_counts.get(key, 0) / len(full_docs) * 100
        test_pct = test_counts.get(key, 0) / len(test_docs) * 100
        diff = test_pct - full_pct
        marker = '⚠️' if abs(diff) > 2 else ''
        print(f'{key:<50} {full_pct:>9.1f}% {test_pct:>9.1f}% {diff:>+7.1f}% {marker}')
    if len(all_keys) > 20:
        print(f'... and {len(all_keys) - 20} more groups')
    print(f"{'=' * 80}\n")

def main() -> None:
    parser = argparse.ArgumentParser(description='Create stratified test set with imbalance handling')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--n', type=int, default=DEFAULT_N, help='Target test set size')
    parser.add_argument('--stratify-by', type=str, nargs='+', default=DEFAULT_STRATIFY_BY, help='Keys for stratification (e.g., course topic)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--min-per-group', type=int, default=3, help='Minimum samples per group (default: 3)')
    parser.add_argument('--max-per-group', type=int, default=None, help='Maximum samples per group (optional)')
    parser.add_argument('--exclude-outliers', action='store_true', help='Exclude docs with topic==-1 from sampling pool')
    args = parser.parse_args()
    logger.info(f'Loading documents from {args.input}')
    docs = load_documents_with_topics(args.input, DEFAULT_TOPIC_ASSIGNMENTS)
    logger.info(f'Loaded {len(docs)} documents')
    topic_counts = Counter((d.get('topic', -1) for d in docs))
    logger.info(f'Topic distribution: {dict(sorted(topic_counts.items()))}')
    logger.info(f'Sampling {args.n} docs stratified by {args.stratify_by}...')
    test_docs = stratified_sample_imbalanced(docs=docs, n=args.n, stratify_by=args.stratify_by, seed=args.seed, min_per_group=args.min_per_group, max_per_group=args.max_per_group, exclude_outliers=args.exclude_outliers)
    metrics = compute_stratification_metrics(docs, test_docs, args.stratify_by)
    logger.info(f'Stratification metrics: {metrics}')
    print_distribution_comparison(docs, test_docs, args.stratify_by)
    print(f'\n✓ Created test set: {len(test_docs)} docs')
    print(f"✓ Max deviation: {metrics['max_deviation_pct']:.1f}% (target: <5%)")
    if metrics['p_value'] is not None:
        status = '✓ PASS' if metrics['p_value'] > 0.05 else '⚠️  CHECK'
        print(f"✓ Chi-square p-value: {metrics['p_value']:.4f} {status}")
    print(f'✓ Seed: {args.seed} (for reproducibility)')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for doc in test_docs:
            f.write(json.dumps(doc) + '\n')
    logger.info(f'Saved {len(test_docs)} test queries to {args.output}')
    metrics_path = args.output.with_suffix('.metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({'stratify_by': args.stratify_by, 'seed': args.seed, 'min_per_group': args.min_per_group, 'max_per_group': args.max_per_group, 'exclude_outliers': args.exclude_outliers, 'metrics': metrics}, f, indent=2)
    logger.info(f'Saved metrics to {metrics_path}')
if __name__ == '__main__':
    main()
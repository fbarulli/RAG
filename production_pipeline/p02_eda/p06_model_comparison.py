"""
Public Functions for Topic Model Comparison Strategy Selection:

def load_model_results(input_dir: Path) -> list[dict[str, Any]]:
    Load all topic_assignments_*.json files and compute all metrics.
    I/O: input_dir (Path) -> list[dict[str, Any]]

def normalize_metric(value: float, all_values: list[float], lower_is_better: bool = True) -> float:
    Normalize a metric to [0, 1] where 1.0 is always best.
    I/O: value (float), all_values (list[float]), lower_is_better (bool) -> float

def compute_composite_score(result: dict[str, Any], all_results: list[dict[str, Any]]) -> float:
    Compute a weighted composite score for a model (higher is better, range [0, 1]).
    I/O: result (dict[str, Any]), all_results (list[dict[str, Any]]) -> float

def evaluate_status_fixed(metrics: dict[str, Any], thresholds: dict[str, float]) -> tuple[str, list[str]]:
    Determine OK/WARN/FAIL status based on fixed thresholds.
    I/O: metrics (dict[str, Any]), thresholds (dict[str, float]) -> tuple[str, list[str]]
    
p06_model_comparison.py
=======================
Compares topic modeling results across embedding models using multiple quality metrics.

Metrics evaluated:
- outlier_ratio: fraction of docs assigned to topic -1 (target: <0.15)
- avg_keyword_overlap: mean pairwise Jaccard of topic keywords (target: <0.10)
- confidence_mean: mean assignment probability (target: >0.7)
- topic_size_std: std deviation of topic sizes (lower = more balanced)
- cross_course_topics: count of topics spanning multiple courses (lower = better isolation)

Supports fixed thresholds OR adaptive percentile-based thresholds.

Output: experiments/model_comparison.json
Run:    uv run python -m production_pipeline.p02_eda.p06_model_comparison
        uv run python -m ... --adaptive --percentile 0.25
"""
import argparse
import json
from pathlib import Path
from typing import Any
from collections import defaultdict
import numpy as np
from rag_pipeline.logging import get_logger
from rag_pipeline.core.paths import Paths
logger = get_logger(__name__)
DEFAULT_INPUT_DIR = Paths.experiments_dir()
DEFAULT_OUTPUT = Paths.experiments_dir() / 'model_comparison.json'
DEFAULT_THRESHOLDS = {'outlier_ratio_max': 0.15, 'keyword_overlap_max': 0.1, 'confidence_min': 0.7, 'cross_course_max': 10}
_RAW_METRIC_WEIGHTS: dict[str, float] = {'outlier_ratio': 0.35, 'avg_keyword_overlap': 0.25, 'confidence_mean': 0.25, 'cross_course_topics': 0.15}
_weight_total = sum(_RAW_METRIC_WEIGHTS.values())
METRIC_WEIGHTS: dict[str, float] = {k: v / _weight_total for k, v in _RAW_METRIC_WEIGHTS.items()}

def load_model_results(input_dir: Path) -> list[dict[str, Any]]:
    """Load all topic_assignments_*.json files and compute all metrics."""
    results: list[dict[str, Any]] = []
    for f in sorted(input_dir.glob('topic_assignments_*.json')):
        if '_validated' in f.name:
            continue
        try:
            with open(f, encoding='utf-8') as fh:
                data = json.load(fh)
            meta = data.get('metadata', {})
            assignments = data.get('assignments', [])
            topics = [a.get('topic') for a in assignments if a.get('topic') != -1]
            probs = [a.get('topic_probability', 0) for a in assignments if a.get('topic') != -1]
            topic_keywords = defaultdict(set)
            for a in assignments:
                t = a.get('topic')
                if t != -1:
                    kws = a.get('keywords', [])
                    if kws and isinstance(kws[0], (list, tuple)):
                        words = [kw[0] for kw in kws if isinstance(kw, (list, tuple)) and len(kw) > 0]
                    else:
                        words = [kw for kw in kws if isinstance(kw, str)]
                    topic_keywords[t].update((w.lower() for w in words))
            overlap_scores = []
            topic_ids = list(topic_keywords.keys())
            for i in range(len(topic_ids)):
                for j in range(i + 1, len(topic_ids)):
                    a, b = (topic_keywords[topic_ids[i]], topic_keywords[topic_ids[j]])
                    if a or b:
                        jaccard = len(a & b) / len(a | b) if a | b else 0
                        overlap_scores.append(jaccard)
            avg_overlap = np.mean(overlap_scores) if overlap_scores else 0
            conf_mean = np.mean(probs) if probs else 0
            from collections import Counter
            sizes = list(Counter(topics).values())
            size_std = np.std(sizes) if len(sizes) > 1 else 0
            topic_courses = defaultdict(set)
            for a in assignments:
                t, c = (a.get('topic'), a.get('course'))
                if t != -1 and c:
                    topic_courses[t].add(c)
            cross_course = sum((1 for courses in topic_courses.values() if len(courses) > 1))
            results.append({'model': meta.get('model', f.stem.replace('topic_assignments_', '').replace('_', '/').replace('__', '-')), 'file': f.name, 'outlier_ratio': meta.get('outlier_ratio', 0.0), 'num_topics': meta.get('num_topics', 0), 'avg_keyword_overlap': round(avg_overlap, 4), 'confidence_mean': round(conf_mean, 3), 'topic_size_std': round(size_std, 1), 'cross_course_topics': cross_course})
        except Exception as e:
            logger.warning(f'Skipping {f.name}: {e}')
            continue
    logger.info(f'Loaded {len(results)} model results from {input_dir}')
    return results

def normalize_metric(value: float, all_values: list[float], lower_is_better: bool=True) -> float:
    """
    Normalize a metric to [0, 1] where 1.0 is always best.

    Returns 0.5 (neutral) when ``value`` is None or ``all_values`` is empty
    so that missing data neither rewards nor punishes a model.
    """
    if not all_values:
        return 0.5
    min_val, max_val = (min(all_values), max(all_values))
    if max_val == min_val:
        return 1.0
    normalised = (value - min_val) / (max_val - min_val)
    return 1.0 - normalised if lower_is_better else normalised

def compute_composite_score(result: dict[str, Any], all_results: list[dict[str, Any]]) -> float:
    """
    Compute a weighted composite score for a model (higher is better, range [0, 1]).

    Metrics with missing values contribute a neutral 0.5 so the composite
    degrades gracefully rather than silently inflating or deflating.
    """
    outlier_vals = [r['outlier_ratio'] for r in all_results if r['outlier_ratio'] is not None]
    overlap_vals = [r['avg_keyword_overlap'] for r in all_results if r['avg_keyword_overlap'] is not None]
    conf_vals = [r['confidence_mean'] for r in all_results if r['confidence_mean'] is not None]
    cross_vals = [r['cross_course_topics'] for r in all_results if r['cross_course_topics'] is not None]
    scores: dict[str, float] = {'outlier_ratio': normalize_metric(result['outlier_ratio'], outlier_vals, lower_is_better=True) if result['outlier_ratio'] is not None else 0.5, 'avg_keyword_overlap': normalize_metric(result['avg_keyword_overlap'], overlap_vals, lower_is_better=True) if result['avg_keyword_overlap'] is not None else 0.5, 'confidence_mean': normalize_metric(result['confidence_mean'], conf_vals, lower_is_better=False) if result['confidence_mean'] is not None else 0.5, 'cross_course_topics': normalize_metric(result['cross_course_topics'], cross_vals, lower_is_better=True) if result['cross_course_topics'] is not None else 0.5}
    return sum((scores[metric] * weight for metric, weight in METRIC_WEIGHTS.items()))

def evaluate_status_fixed(metrics: dict[str, Any], thresholds: dict[str, float]) -> tuple[str, list[str]]:
    """Determine OK/WARN/FAIL status based on fixed thresholds."""
    violations: list[str] = []
    if metrics['outlier_ratio'] >= thresholds['outlier_ratio_max']:
        violations.append(f"outlier_ratio {metrics['outlier_ratio']:.1%} >= {thresholds['outlier_ratio_max']:.1%}")
    if metrics['avg_keyword_overlap'] is not None:
        if metrics['avg_keyword_overlap'] >= thresholds['keyword_overlap_max']:
            violations.append(f"keyword_overlap {metrics['avg_keyword_overlap']:.4f} >= {thresholds['keyword_overlap_max']:.4f}")
    if metrics['confidence_mean'] is not None:
        if metrics['confidence_mean'] < thresholds['confidence_min']:
            violations.append(f"confidence_mean {metrics['confidence_mean']:.3f} < {thresholds['confidence_min']:.3f}")
    if metrics['cross_course_topics'] is not None:
        if metrics['cross_course_topics'] > thresholds['cross_course_max']:
            violations.append(f"cross_course_topics {metrics['cross_course_topics']} > {thresholds['cross_course_max']}")
    if not violations:
        return ('OK', [])
    if len(violations) == 1:
        return ('WARN', violations)
    return ('FAIL', violations)

def _adaptive_thresholds(all_results: list[dict[str, Any]], percentile: float) -> dict[str, float]:
    """
    Compute adaptive thresholds from the distribution of results.

    For lower-is-better metrics (outlier_ratio, avg_keyword_overlap) a model
    must be *below* the upper percentile cutoff to pass, i.e. we use
    ``(1 - percentile) * 100`` so that the *top* ``percentile`` fraction passes.

    For higher-is-better metrics (confidence_mean) a model must be *above* the
    lower percentile cutoff, i.e. we use ``percentile * 100`` for the same reason.

    Example with percentile=0.25 (top 25% pass):
      - outlier_ratio threshold = p75  → top 25% of models (lowest outlier ratios) pass
      - confidence_mean threshold = p25 → top 25% of models (highest confidence) pass
    """
    outlier_vals = [r['outlier_ratio'] for r in all_results if r['outlier_ratio'] is not None]
    overlap_vals = [r['avg_keyword_overlap'] for r in all_results if r['avg_keyword_overlap'] is not None]
    conf_vals = [r['confidence_mean'] for r in all_results if r['confidence_mean'] is not None]
    upper_pct = (1.0 - percentile) * 100
    lower_pct = percentile * 100
    return {'outlier_thresh': float(np.percentile(outlier_vals, upper_pct)) if outlier_vals else DEFAULT_THRESHOLDS['outlier_ratio_max'], 'overlap_thresh': float(np.percentile(overlap_vals, upper_pct)) if overlap_vals else DEFAULT_THRESHOLDS['keyword_overlap_max'], 'conf_thresh': float(np.percentile(conf_vals, lower_pct)) if conf_vals else DEFAULT_THRESHOLDS['confidence_min']}

def evaluate_status_adaptive(metrics: dict[str, Any], all_results: list[dict[str, Any]], percentile: float=0.25) -> tuple[str, list[str]]:
    """
    Evaluate against percentile-based thresholds derived from current results.

    Args:
        metrics: Single model's metrics to evaluate.
        all_results: All model results for computing percentiles.
        percentile: Top-N fraction that should pass (0.25 = top 25% pass).

    Returns:
        status: "OK" | "WARN" | "FAIL"
        violations: Human-readable violation strings.
    """
    violations: list[str] = []
    thresh = _adaptive_thresholds(all_results, percentile)
    pct_label = f'p{(1 - percentile) * 100:.0f}'
    if metrics['outlier_ratio'] is not None and metrics['outlier_ratio'] > thresh['outlier_thresh']:
        violations.append(f"outlier_ratio {metrics['outlier_ratio']:.1%} > {thresh['outlier_thresh']:.1%} ({pct_label})")
    if metrics['avg_keyword_overlap'] is not None and metrics['avg_keyword_overlap'] > thresh['overlap_thresh']:
        violations.append(f"keyword_overlap {metrics['avg_keyword_overlap']:.4f} > {thresh['overlap_thresh']:.4f} ({pct_label})")
    if metrics['confidence_mean'] is not None and metrics['confidence_mean'] < thresh['conf_thresh']:
        violations.append(f"confidence_mean {metrics['confidence_mean']:.3f} < {thresh['conf_thresh']:.3f} ({pct_label})")
    if metrics['cross_course_topics'] is not None:
        if metrics['cross_course_topics'] > DEFAULT_THRESHOLDS['cross_course_max']:
            violations.append(f"cross_course_topics {metrics['cross_course_topics']} > {DEFAULT_THRESHOLDS['cross_course_max']}")
    if not violations:
        return ('OK', [])
    if len(violations) == 1:
        return ('WARN', violations)
    return ('FAIL', violations)

def find_pareto_front(results: list[dict[str, Any]]) -> list[str]:
    """
    Identify non-dominated models (Pareto-optimal).

    A model is dominated if another model is at least as good on *all* tracked
    metrics and strictly better on *at least one*.

    Bug fix vs original: the dominance check now correctly evaluates *all*
    metric groups before concluding that one model dominates another.
    The original implementation short-circuited out of the minimize loop via
    ``continue`` without ever checking the maximize metrics.
    """
    minimize = ['outlier_ratio', 'avg_keyword_overlap']
    maximize = ['confidence_mean']
    all_tracked = minimize + maximize
    valid = [r for r in results if all((r.get(m) is not None for m in all_tracked))]
    pareto_models: list[str] = []
    for i, candidate in enumerate(valid):
        is_dominated = False
        for j, other in enumerate(valid):
            if i == j:
                continue
            at_least_as_good = True
            strictly_better = False
            for metric in minimize:
                if other[metric] > candidate[metric]:
                    at_least_as_good = False
                    break
                if other[metric] < candidate[metric]:
                    strictly_better = True
            if not at_least_as_good:
                continue
            for metric in maximize:
                if other[metric] < candidate[metric]:
                    at_least_as_good = False
                    break
                if other[metric] > candidate[metric]:
                    strictly_better = True
            if at_least_as_good and strictly_better:
                is_dominated = True
                break
        if not is_dominated:
            pareto_models.append(candidate['model'])
    return pareto_models

def _format_optional(value: float | None, fmt: str, fallback: str='N/A') -> str:
    return format(value, fmt) if value is not None else fallback

def print_comparison_table(results: list[dict[str, Any]], thresholds: dict[str, float], adaptive: bool=False, percentile: float=0.25) -> None:
    """Print a formatted comparison table to console.

    Uses pre-computed ``status`` / ``violations`` already stored on each result
    dict rather than re-evaluating per row.
    """
    if not results:
        print('No results to display.')
        return
    mode_str = f' (Adaptive p{percentile * 100:.0f})' if adaptive else ' (Fixed)'
    print(f'\nTOPIC MODELING QUALITY COMPARISON{mode_str}')
    print('=' * 160)
    print(f"{'Model':<45} {'Out%':>7} {'Overlap':>10} {'Conf':>8} {'Std':>8} {'Cross':>7} {'Topics':>7} {'Score':>7} {'Status':>8} {'Violations':<40}")
    print('-' * 160)
    for r in results:
        violations_str = '; '.join(r.get('violations', []))[:40]
        score_str = f"{r['composite_score']:.3f}" if 'composite_score' in r else 'N/A'
        print(f"{r['model']:<45} {r['outlier_ratio']:>6.1%} {_format_optional(r['avg_keyword_overlap'], '.4f'):>10} {_format_optional(r['confidence_mean'], '.3f'):>8} {_format_optional(r['topic_size_std'], '.1f'):>8} {(str(r['cross_course_topics']) if r['cross_course_topics'] is not None else 'N/A'):>7} {r['num_topics']:>7} {score_str:>7} {r.get('status', '?'):>8} {violations_str:<40}")
    print('=' * 160)
    if adaptive:
        thresh = _adaptive_thresholds(results, percentile)
        pct = (1 - percentile) * 100
        print(f"Adaptive thresholds (top {percentile * 100:.0f}% pass): outlier≤{thresh['outlier_thresh']:.1%} (p{pct:.0f}), overlap≤{thresh['overlap_thresh']:.4f} (p{pct:.0f}), conf≥{thresh['conf_thresh']:.3f} (p{percentile * 100:.0f})")
    else:
        print(f"Fixed thresholds: outlier≤{thresholds['outlier_ratio_max']:.0%}, overlap≤{thresholds['keyword_overlap_max']:.2f}, conf≥{thresholds['confidence_min']:.2f}, cross≤{thresholds['cross_course_max']}")
    pareto = find_pareto_front(results)
    if pareto:
        print(f'\nPareto-optimal models (no single model dominates these):')
        for m in pareto:
            print(f'  ✓ {m}')
    scored = sorted([r for r in results if 'composite_score' in r], key=lambda x: x['composite_score'], reverse=True)
    if scored:
        print('\nTop 3 by composite score:')
        for rank, r in enumerate(scored[:3], 1):
            print(f"  {rank}. {r['model']} (score: {r['composite_score']:.3f})")
    print()

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Compare topic modeling results across embedding models')
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR, help='Directory containing topic_assignments_*.json files')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Output JSON file path')
    parser.add_argument('--outlier-threshold', type=float, default=DEFAULT_THRESHOLDS['outlier_ratio_max'])
    parser.add_argument('--overlap-threshold', type=float, default=DEFAULT_THRESHOLDS['keyword_overlap_max'])
    parser.add_argument('--confidence-threshold', type=float, default=DEFAULT_THRESHOLDS['confidence_min'])
    parser.add_argument('--cross-course-threshold', type=int, default=DEFAULT_THRESHOLDS['cross_course_max'])
    parser.add_argument('--adaptive', action='store_true', help='Use adaptive percentile-based thresholds instead of fixed values')
    parser.add_argument('--adaptive-method', type=str, choices=['percentile', 'pareto'], default='percentile', help="Method for adaptive evaluation. 'percentile' uses distribution-based cutoffs; 'pareto' skips per-metric thresholds and reports the Pareto front.")
    parser.add_argument('--percentile', type=float, default=0.25, help='Top-N fraction that should pass (0.25 = top 25%% pass). Only used with --adaptive.')
    parser.add_argument('--no-composite', action='store_true', help='Disable composite score calculation')
    return parser

def _evaluate_all(results: list[dict[str, Any]], thresholds: dict[str, float], adaptive: bool, adaptive_method: str, percentile: float) -> None:
    """Attach 'status' and 'violations' to each result in-place."""
    for r in results:
        if adaptive and adaptive_method == 'percentile':
            status, violations = evaluate_status_adaptive(r, results, percentile)
        else:
            status, violations = evaluate_status_fixed(r, thresholds)
        r['status'] = status
        r['violations'] = violations

def main() -> None:
    args = build_arg_parser().parse_args()
    thresholds = {'outlier_ratio_max': args.outlier_threshold, 'keyword_overlap_max': args.overlap_threshold, 'confidence_min': args.confidence_threshold, 'cross_course_max': args.cross_course_threshold}
    logger.info(f'Loading results from {args.input_dir}')
    results = load_model_results(args.input_dir)
    if not results:
        logger.error('No topic assignment files found. Expected pattern: topic_assignments_*.json')
        raise SystemExit(1)
    if not args.no_composite:
        logger.info('Computing composite scores…')
        for r in results:
            r['composite_score'] = compute_composite_score(r, results)
    _evaluate_all(results, thresholds, args.adaptive, args.adaptive_method, args.percentile)
    if not args.no_composite:
        results.sort(key=lambda x: x.get('composite_score', 0.0), reverse=True)
    else:
        results.sort(key=lambda x: (x['outlier_ratio'], x['avg_keyword_overlap'] if x['avg_keyword_overlap'] is not None else 1.0))
    pareto_models = find_pareto_front(results)
    print_comparison_table(results, thresholds, args.adaptive, args.percentile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {'thresholds': thresholds, 'adaptive_mode': args.adaptive, 'adaptive_method': args.adaptive_method if args.adaptive else None, 'percentile': args.percentile if args.adaptive else None, 'composite_score_enabled': not args.no_composite, 'results': results, 'best_model': results[0]['model'] if results else None, 'pareto_optimal': pareto_models or None, 'metadata': {'total_models': len(results), 'models_ok': sum((1 for r in results if r.get('status') == 'OK')), 'models_warn': sum((1 for r in results if r.get('status') == 'WARN')), 'models_fail': sum((1 for r in results if r.get('status') == 'FAIL'))}}
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f'Saved comparison summary: {args.output}')
    if results:
        best = results[0]
        print(f"\n🎯 Recommendation: Use '{best['model']}'")
        print(f"   • Outlier ratio:  {best['outlier_ratio']:.1%}")
        if 'composite_score' in best:
            print(f"   • Composite score: {best['composite_score']:.3f}")
        if best['avg_keyword_overlap'] is not None:
            print(f"   • Keyword overlap: {best['avg_keyword_overlap']:.4f}")
        if best['confidence_mean'] is not None:
            print(f"   • Mean confidence: {best['confidence_mean']:.3f}")
        print(f"   • Status: {best['status']}")
        if best['violations']:
            print(f"   • Note: {best['violations'][0]}")
        if pareto_models and best['model'] not in pareto_models:
            print('\n💡 Pareto-optimal alternatives to consider:')
            for m in pareto_models[:3]:
                print(f'   • {m}')
if __name__ == '__main__':
    main()
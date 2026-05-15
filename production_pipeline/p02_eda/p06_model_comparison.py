"""
p06_model_comparison.py
=======================
Compares topic modeling results across embedding models using multiple quality metrics.

Metrics evaluated:
- outlier_ratio: fraction of docs assigned to topic -1 (target: <0.15)
- avg_keyword_overlap: mean pairwise Jaccard of topic keywords (target: <0.10)
- confidence_mean: mean assignment probability (target: >0.7)
- topic_size_std: std deviation of topic sizes (lower = more balanced)
- cross_course_topics: count of topics spanning multiple courses (lower = better isolation)

Output: experiments/model_comparison.json
Run:    uv run python -m production_pipeline.p02_eda.p06_model_comparison
"""
import argparse
import json
from pathlib import Path
from typing import Any, Optional

from rag_pipeline.logging import get_logger
from rag_pipeline.paths import Paths

logger = get_logger(__name__)

DEFAULT_INPUT_DIR = Paths.experiments_dir()
DEFAULT_OUTPUT = Paths.experiments_dir() / "model_comparison.json"

# Default thresholds for pass/warn/fail evaluation
DEFAULT_THRESHOLDS = {
    "outlier_ratio_max": 0.15,
    "keyword_overlap_max": 0.10,
    "confidence_min": 0.70,
    "cross_course_max": 10,
}


def load_model_results(input_dir: Path) -> list[dict[str, Any]]:
    """Load all topic_assignments_*.json files and extract metrics."""
    results = []
    for f in sorted(input_dir.glob("topic_assignments_*.json")):
        try:
            data = json.load(open(f, encoding="utf-8"))
            meta = data.get("metadata", {})
            
            result = {
                "model": meta.get("model", f.stem.replace("topic_assignments_", "")),
                "file": f.name,
                "outlier_ratio": meta.get("outlier_ratio", 0.0),
                "num_topics": meta.get("num_topics", 0),
                "avg_keyword_overlap": data.get("avg_keyword_overlap"),
                "confidence_mean": (
                    data.get("confidence_stats", {}).get("mean")
                    if data.get("confidence_stats") else None
                ),
                "topic_size_std": (
                    data.get("topic_size_stats", {}).get("std")
                    if data.get("topic_size_stats") else None
                ),
                "cross_course_topics": data.get("cross_course_topics"),
            }
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to load {f.name}: {e}")
            continue
    return results


def evaluate_status(metrics: dict[str, Any], thresholds: dict[str, float]) -> tuple[str, list[str]]:
    """Determine OK/WARN/FAIL status based on all thresholds."""
    violations = []
    
    if metrics["outlier_ratio"] >= thresholds["outlier_ratio_max"]:
        violations.append(f"outlier_ratio {metrics['outlier_ratio']:.1%} >= {thresholds['outlier_ratio_max']:.1%}")
    
    if metrics["avg_keyword_overlap"] is not None:
        if metrics["avg_keyword_overlap"] >= thresholds["keyword_overlap_max"]:
            violations.append(f"keyword_overlap {metrics['avg_keyword_overlap']:.4f} >= {thresholds['keyword_overlap_max']:.4f}")
    
    if metrics["confidence_mean"] is not None:
        if metrics["confidence_mean"] < thresholds["confidence_min"]:
            violations.append(f"confidence_mean {metrics['confidence_mean']:.3f} < {thresholds['confidence_min']:.3f}")
    
    if metrics["cross_course_topics"] is not None:
        if metrics["cross_course_topics"] > thresholds["cross_course_max"]:
            violations.append(f"cross_course_topics {metrics['cross_course_topics']} > {thresholds['cross_course_max']}")
    
    if not violations:
        return "OK", []
    elif len(violations) == 1:
        return "WARN", violations
    else:
        return "FAIL", violations


def print_comparison_table(results: list[dict[str, Any]], thresholds: dict[str, float]) -> None:
    """Print a formatted comparison table to console."""
    if not results:
        print("No results to display.")
        return
    
    print("\nTOPIC MODELING QUALITY COMPARISON")
    print("=" * 140)
    header = (
        f"{'Model':<40} "
        f"{'Out%':>6} {'Overlap':>9} {'Conf':>7} {'Std':>7} {'Cross':>7} "
        f"{'Topics':>7} {'Status':>8} {'Violations':<30}"
    )
    print(header)
    print("-" * 140)
    
    for r in results:
        status, violations = evaluate_status(r, thresholds)
        violations_str = "; ".join(violations)[:30] if violations else ""
        
        overlap_str = f"{r['avg_keyword_overlap']:.4f}" if r['avg_keyword_overlap'] is not None else "N/A"
        conf_str = f"{r['confidence_mean']:.3f}" if r['confidence_mean'] is not None else "N/A"
        std_str = f"{r['topic_size_std']:.1f}" if r['topic_size_std'] is not None else "N/A"
        cross_str = str(r['cross_course_topics']) if r['cross_course_topics'] is not None else "N/A"
        
        print(
            f"{r['model']:<40} "
            f"{r['outlier_ratio']:>5.1%} {overlap_str:>9} {conf_str:>7} {std_str:>7} {cross_str:>7} "
            f"{r['num_topics']:>7} {status:>8} {violations_str:<30}"
        )
    
    print("=" * 140)
    print(f"Thresholds: outlier<={thresholds['outlier_ratio_max']:.0%}, "
          f"overlap<={thresholds['keyword_overlap_max']:.2f}, "
          f"conf>={thresholds['confidence_min']:.2f}, "
          f"cross<={thresholds['cross_course_max']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare topic modeling results across embedding models")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Directory containing topic_assignments_*.json files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--outlier-threshold", type=float, default=DEFAULT_THRESHOLDS["outlier_ratio_max"])
    parser.add_argument("--overlap-threshold", type=float, default=DEFAULT_THRESHOLDS["keyword_overlap_max"])
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_THRESHOLDS["confidence_min"])
    parser.add_argument("--cross-course-threshold", type=int, default=DEFAULT_THRESHOLDS["cross_course_max"])
    args = parser.parse_args()

    thresholds = {
        "outlier_ratio_max": args.outlier_threshold,
        "keyword_overlap_max": args.overlap_threshold,
        "confidence_min": args.confidence_threshold,
        "cross_course_max": args.cross_course_threshold,
    }

    logger.info(f"Loading results from {args.input_dir}")
    results = load_model_results(args.input_dir)
    
    if not results:
        logger.error("No topic assignment files found.")
        raise SystemExit(1)
    
    logger.info(f"Loaded {len(results)} model results")
    
    # Add status evaluation to each result
    for r in results:
        status, violations = evaluate_status(r, thresholds)
        r["status"] = status
        r["violations"] = violations
    
    # Sort by outlier_ratio ascending, then by keyword_overlap
    results.sort(key=lambda x: (x["outlier_ratio"], x["avg_keyword_overlap"] or 1.0))
    
    # Print console table
    print_comparison_table(results, thresholds)
    
    # Save JSON summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "thresholds": thresholds,
        "results": results,
        "best_model": results[0]["model"] if results else None,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved comparison summary: {args.output}")


if __name__ == "__main__":
    main()
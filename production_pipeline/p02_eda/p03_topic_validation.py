"""
p03_topic_validation.py
=======================
Validates topic modeling outputs by computing outlier ratios, topic size distributions,
confidence statistics, keyword overlap, and quality proxies.

Separates pure measurement from threshold-based evaluation to enable independent testing.

Output: experiments/topic_validation.json
Run:    uv run python -m production_pipeline.p02_eda.p03_topic_validation
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Optional

from rag_pipeline.logging import get_logger
from rag_pipeline.paths import Paths

logger = get_logger(__name__)

DEFAULT_INPUT = Paths.experiments_dir() / "topic_assignments.json"
DEFAULT_OUTPUT = Paths.experiments_dir() / "topic_validation.json"


# ---------------------------------------------------------------------------
# Load & Validate
# ---------------------------------------------------------------------------

def load_assignments(path: Path) -> list[dict]:
    """Load and validate topic assignment records from JSON."""
    if not path.exists():
        logger.error(f"Input file not found: {path}")
        raise SystemExit(1)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    assignments = data.get("assignments")
    if not assignments:
        logger.error("No assignments loaded — check input file structure (expected key: 'assignments')")
        raise SystemExit(1)

    return assignments


# ---------------------------------------------------------------------------
# Core Metrics (Pure Computation)
# ---------------------------------------------------------------------------

def _compute_keyword_overlap(topic_keywords: dict[int, set[str]]) -> Optional[float]:
    """
    Average pairwise Jaccard similarity across all topic keyword sets.
    Returns None if fewer than 2 topics or more than 100 (too expensive).
    A score < 0.1 indicates well-separated topics.
    """
    ids = list(topic_keywords.keys())
    if len(ids) < 2:
        return None
    if len(ids) > 100:
        logger.warning(f"Skipping keyword overlap for {len(ids)} topics (>100, too expensive)")
        return None

    pairs, total = 0, 0.0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = topic_keywords[ids[i]], topic_keywords[ids[j]]
            if a or b:
                total += len(a & b) / len(a | b)
            pairs += 1

    return round(total / pairs, 4) if pairs else None


def compute_raw_metrics(assignments: list[dict]) -> dict[str, Any]:
    """
    Compute all raw measurement metrics. No pass/fail logic.

    Returns keys:
        total_documents        : int
        num_topics             : int
        outlier_count          : int
        outlier_ratio          : float
        topic_sizes            : dict[str, int]  — string keys for JSON round-trip safety
        topic_size_stats       : dict with min, max, mean, median, std (all float)
        largest_topic_share    : float
        tiny_topic_count       : int
        tiny_topic_ids         : list[int]
        confidence_stats       : Optional[dict]  — None if no probability field present
        cross_course_topics    : int
        avg_keyword_overlap    : Optional[float] — None if <2 topics or >100 topics
    """
    topics = [a.get("topic", -1) for a in assignments]
    topic_counts = Counter(topics)

    total = len(topics)
    outliers = topic_counts.get(-1, 0)
    outlier_ratio = outliers / total if total > 0 else 0.0

    real_topics = {t: c for t, c in topic_counts.items() if t != -1}
    sizes = list(real_topics.values())

    # Size statistics (consistent float types)
    size_stats: dict[str, float] = {
        "min": float(min(sizes)) if sizes else 0.0,
        "max": float(max(sizes)) if sizes else 0.0,
        "mean": round(mean(sizes), 1) if sizes else 0.0,
        "median": round(median(sizes), 1) if sizes else 0.0,
        "std": round(stdev(sizes), 1) if len(sizes) > 1 else 0.0,
    }

    total_in_topics = sum(sizes) if sizes else 0
    largest_topic_share = (max(sizes) / total_in_topics) if sizes else 0.0

    tiny_topic_ids = sorted([t for t, c in real_topics.items() if c <= 3])
    tiny_topic_count = len(tiny_topic_ids)

    # Confidence stats — only if probability field exists in data
    probs = [
        a["topic_probability"]
        for a in assignments
        if a.get("topic", -1) != -1 and "topic_probability" in a
    ]
    confidence_stats: Optional[dict[str, Any]] = None
    if probs:
        low_conf = sum(1 for p in probs if p < 0.5)
        confidence_stats = {
            "mean": round(mean(probs), 4),
            "low_confidence_count": low_conf,
            "low_confidence_ratio": round(low_conf / len(probs), 4),
        }

    # Per-course topic distribution
    course_topic_map: dict[str, set] = defaultdict(set)
    for a in assignments:
        course_topic_map[a.get("course", "unknown")].add(a.get("topic", -1))
    cross_course_count = sum(1 for t_set in course_topic_map.values() if len(t_set) > 1)

    # Keyword overlap (uses subtopic_keywords or keywords if present)
    topic_kw: dict[int, set[str]] = defaultdict(set)
    for a in assignments:
        t = a.get("topic", -1)
        if t != -1:
            for kw in (a.get("subtopic_keywords") or a.get("keywords") or []):
                topic_kw[t].add(kw.lower())
    keyword_overlap = _compute_keyword_overlap(topic_kw)

    return {
        "total_documents": total,
        "num_topics": len(real_topics),
        "outlier_count": outliers,
        "outlier_ratio": round(outlier_ratio, 4),
        "topic_sizes": {str(k): v for k, v in sorted(real_topics.items())},
        "topic_size_stats": size_stats,
        "largest_topic_share": round(largest_topic_share, 4),
        "tiny_topic_count": tiny_topic_count,
        "tiny_topic_ids": tiny_topic_ids,
        "confidence_stats": confidence_stats,
        "cross_course_topics": cross_course_count,
        "avg_keyword_overlap": keyword_overlap,
    }


# ---------------------------------------------------------------------------
# Evaluation Logic (Threshold Interpretation)
# ---------------------------------------------------------------------------

def evaluate_status(
    metrics: dict[str, Any],
    outlier_thresh: float,
    tiny_thresh: int,
    conc_thresh: float,
) -> tuple[str, list[str]]:
    """
    Determine OK / WARN / FAIL status based on configurable thresholds.
    Collects all violations rather than returning on first match.

    Returns:
        status     : "OK" | "WARN" | "FAIL"
        violations : list of human-readable violation strings
    """
    violations: list[str] = []

    if metrics["outlier_ratio"] > outlier_thresh:
        violations.append(
            f"outlier_ratio {metrics['outlier_ratio']:.1%} > threshold {outlier_thresh:.1%}"
        )
    if metrics["tiny_topic_count"] > tiny_thresh:
        violations.append(
            f"tiny_topics {metrics['tiny_topic_count']} > threshold {tiny_thresh}"
        )
    if metrics["largest_topic_share"] > conc_thresh:
        violations.append(
            f"concentration {metrics['largest_topic_share']:.1%} > threshold {conc_thresh:.1%}"
        )

    status = "WARN" if violations else "OK"

    if metrics.get("confidence_stats") and metrics["confidence_stats"]["mean"] < 0.4:
        violations.append(
            f"mean_confidence {metrics['confidence_stats']['mean']:.3f} < 0.4"
        )
        status = "FAIL"

    return status, violations


# ---------------------------------------------------------------------------
# CLI / Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate topic modeling outputs")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--outlier-threshold", type=float, default=0.15,
                        help="Max acceptable outlier ratio (default: 0.15)")
    parser.add_argument("--tiny-topic-threshold", type=int, default=3,
                        help="Max acceptable number of tiny topics ≤3 docs (default: 3)")
    parser.add_argument("--concentration-threshold", type=float, default=0.30,
                        help="Max acceptable largest-topic share (default: 0.30)")
    args = parser.parse_args()

    logger.info(f"Loading topic assignments from {args.input}")
    assignments = load_assignments(args.input)

    logger.info("Computing validation metrics...")
    metrics = compute_raw_metrics(assignments)

    status, violations = evaluate_status(
        metrics,
        args.outlier_threshold,
        args.tiny_topic_threshold,
        args.concentration_threshold,
    )
    metrics["status"] = status
    metrics["violations"] = violations
    metrics["thresholds_used"] = {
        "outlier": args.outlier_threshold,
        "tiny_topic": args.tiny_topic_threshold,
        "concentration": args.concentration_threshold,
    }

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved validation report: {args.output}")

    # Console summary
    print("\nTOPIC MODELING VALIDATION REPORT")
    print("=" * 60)
    print(f"Total documents:     {metrics['total_documents']}")
    print(f"Number of topics:    {metrics['num_topics']}")
    print(f"Outliers (topic -1): {metrics['outlier_count']} ({metrics['outlier_ratio']:.1%})")
    print(f"Topic size range:    {metrics['topic_size_stats']['min']:.0f} – {metrics['topic_size_stats']['max']:.0f}")
    print(f"Mean / std:          {metrics['topic_size_stats']['mean']} / {metrics['topic_size_stats']['std']}")
    print(f"Largest topic share: {metrics['largest_topic_share']:.1%}")
    print(f"Tiny topics (≤3):    {metrics['tiny_topic_count']}", end="")
    if metrics["tiny_topic_ids"]:
        print(f"  → IDs: {metrics['tiny_topic_ids']}", end="")
    print()
    print(f"Cross-course topics: {metrics['cross_course_topics']}")
    if metrics["avg_keyword_overlap"] is not None:
        print(f"Keyword overlap:     {metrics['avg_keyword_overlap']:.4f}  (< 0.10 is good)")
    if metrics.get("confidence_stats"):
        cs = metrics["confidence_stats"]
        print(f"Mean confidence:     {cs['mean']}")
        print(f"Low-conf docs (<0.5):{cs['low_confidence_count']} ({cs['low_confidence_ratio']:.1%})")
    print(f"Overall status:      {metrics['status']}")
    if violations:
        print("\nViolations:")
        for v in violations:
            print(f"  ⚠  {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
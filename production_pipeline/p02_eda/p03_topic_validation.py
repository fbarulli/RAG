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
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from rag_pipeline.logging import get_logger
from rag_pipeline.paths import Paths

logger = get_logger(__name__)

DEFAULT_INPUT = Paths.experiments_dir() / "topic_assignments.json"
DEFAULT_OUTPUT = Paths.experiments_dir() / "topic_validation.json"

# Maximum number of topics before keyword overlap switches to random-pair sampling.
_OVERLAP_EXACT_LIMIT = 100
# Number of random pairs to sample when above the exact limit.
_OVERLAP_SAMPLE_PAIRS = 2_000


# ---------------------------------------------------------------------------
# Load & validate
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised when the input file is missing or structurally invalid."""


def load_assignments(path: Path) -> list[dict]:
    """
    Load and validate topic assignment records from JSON.

    Raises:
        ValidationError: if the file is missing, unparseable, or lacks 'assignments'.
    """
    if not path.exists():
        raise ValidationError(f"Input file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in {path}: {exc}") from exc

    assignments = data.get("assignments")
    if not assignments:
        raise ValidationError(
            "No assignments found — check input file structure (expected key: 'assignments')"
        )

    return assignments


# ---------------------------------------------------------------------------
# Core metrics (pure computation)
# ---------------------------------------------------------------------------

def _extract_keywords(record: dict) -> list[str]:
    """
    Return a normalised list of keyword strings from a single assignment record.

    Handles both flat ``["word", ...]`` and BERTopic-style ``[["word", weight], ...]``
    formats transparently.
    """
    raw = record.get("subtopic_keywords") or record.get("keywords") or []
    if raw and isinstance(raw[0], (list, tuple)):
        # BERTopic format: [(word, weight), ...]
        return [item[0] for item in raw if isinstance(item, (list, tuple)) and len(item) >= 1 and isinstance(item[0], str)]
    return [kw for kw in raw if isinstance(kw, str)]


def _compute_keyword_overlap(topic_keywords: dict[int, set[str]]) -> float | None:
    """
    Average pairwise Jaccard similarity across topic keyword sets.

    - Fewer than 2 topics → ``None`` (undefined).
    - Up to ``_OVERLAP_EXACT_LIMIT`` topics → exact O(n²) calculation.
    - More than ``_OVERLAP_EXACT_LIMIT`` topics → Monte-Carlo estimate over
      ``_OVERLAP_SAMPLE_PAIRS`` random pairs, which is far cheaper while still
      giving a useful signal instead of silently returning ``None``.

    A score < 0.1 indicates well-separated topics.
    """
    ids = list(topic_keywords.keys())
    n = len(ids)

    if n < 2:
        return None

    def _jaccard(a: set[str], b: set[str]) -> float:
        union = a | b
        return len(a & b) / len(union) if union else 0.0

    if n <= _OVERLAP_EXACT_LIMIT:
        pairs, total = 0, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                total += _jaccard(topic_keywords[ids[i]], topic_keywords[ids[j]])
                pairs += 1
        return round(total / pairs, 4) if pairs else None

    # Approximate via random sampling for large topic counts.
    logger.info(
        f"Topic count {n} exceeds exact limit ({_OVERLAP_EXACT_LIMIT}); "
        f"estimating keyword overlap from {_OVERLAP_SAMPLE_PAIRS} random pairs."
    )
    total = 0.0
    for _ in range(_OVERLAP_SAMPLE_PAIRS):
        i, j = random.sample(ids, 2)
        total += _jaccard(topic_keywords[i], topic_keywords[j])
    return round(total / _OVERLAP_SAMPLE_PAIRS, 4)


def _build_cross_course_topic_count(assignments: list[dict]) -> int:
    """
    Count topics that appear in more than one course.

    Iterates over assignments and, for each real topic (not -1), records which
    courses it is present in. A topic is "cross-course" if that set has size > 1.
    """
    topic_courses: dict[int, set[str]] = defaultdict(set)
    for a in assignments:
        t = a.get("topic", -1)
        if t != -1:
            topic_courses[t].add(a.get("course", "unknown"))
    return sum(1 for courses in topic_courses.values() if len(courses) > 1)


def compute_raw_metrics(assignments: list[dict]) -> dict[str, Any]:
    """
    Compute all raw measurement metrics. No pass/fail logic.

    Returns keys:
        total_documents        : int
        num_topics             : int
        outlier_count          : int
        outlier_ratio          : float   — outliers / total_documents
        topic_sizes            : dict[str, int]  — string keys for JSON round-trip
        topic_size_stats       : dict with min, max, mean, median, std (float)
        largest_topic_share    : float   — largest topic size / total_documents
        tiny_topic_count       : int
        tiny_topic_ids         : list[int]
        confidence_stats       : dict | None  — None if no 'topic_probability' field
                                 present; excludes outlier (topic -1) assignments
        cross_course_topics    : int  — number of topics spanning >1 course
        avg_keyword_overlap    : float | None  — None if <2 topics
    """
    topics = [a.get("topic", -1) for a in assignments]
    topic_counts = Counter(topics)

    total = len(topics)
    outliers = topic_counts.get(-1, 0)
    outlier_ratio = outliers / total if total > 0 else 0.0

    real_topics = {t: c for t, c in topic_counts.items() if t != -1}
    sizes = list(real_topics.values())

    size_stats: dict[str, float] = {
        "min": float(min(sizes)) if sizes else 0.0,
        "max": float(max(sizes)) if sizes else 0.0,
        "mean": round(mean(sizes), 1) if sizes else 0.0,
        "median": round(median(sizes), 1) if sizes else 0.0,
        "std": round(stdev(sizes), 1) if len(sizes) > 1 else 0.0,
    }

    # Share of *all* documents occupied by the single largest topic, consistent
    # with how outlier_ratio is defined (denominator = total_documents).
    largest_topic_share = (max(sizes) / total) if sizes and total > 0 else 0.0

    tiny_topic_ids = sorted(t for t, c in real_topics.items() if c <= 3)
    tiny_topic_count = len(tiny_topic_ids)

    # Confidence stats — only non-outlier assignments with a probability field.
    probs = [
        a["topic_probability"]
        for a in assignments
        if a.get("topic", -1) != -1 and "topic_probability" in a
    ]
    confidence_stats: dict[str, Any] | None = None
    if probs:
        low_conf = sum(1 for p in probs if p < 0.5)
        confidence_stats = {
            "mean": round(mean(probs), 4),
            "low_confidence_count": low_conf,
            "low_confidence_ratio": round(low_conf / len(probs), 4),
        }

    cross_course_count = _build_cross_course_topic_count(assignments)

    # Build per-topic keyword sets (real topics only).
    topic_kw: defaultdict[int, set[str]] = defaultdict(set)
    for a in assignments:
        t = a.get("topic", -1)
        if t != -1:
            for kw in _extract_keywords(a):
                topic_kw[t].add(kw.lower())

    keyword_overlap = _compute_keyword_overlap(dict(topic_kw))

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
# Evaluation logic (threshold interpretation)
# ---------------------------------------------------------------------------

def evaluate_status(
    metrics: dict[str, Any],
    outlier_thresh: float,
    tiny_thresh: int,
    conc_thresh: float,
    confidence_fail_thresh: float = 0.4,
) -> tuple[str, list[str]]:
    """
    Determine OK / WARN / FAIL status based on configurable thresholds.

    All violations are collected before the status is assigned so that the
    final grade reflects the full picture rather than the order checks run.

    Rules:
      - Any violation of outlier, tiny-topic, or concentration → at least WARN.
      - Low mean confidence → at least WARN; if it is the *only* violation, WARN
        not FAIL (confidence is treated as a soft signal, unlike hard constraints).
      - Two or more violations of any kind → FAIL.
      - Zero violations → OK.

    Args:
        metrics: Output of ``compute_raw_metrics``.
        outlier_thresh: Maximum acceptable outlier_ratio.
        tiny_thresh: Maximum acceptable tiny_topic_count.
        conc_thresh: Maximum acceptable largest_topic_share.
        confidence_fail_thresh: Mean confidence below this triggers a violation.

    Returns:
        status     : "OK" | "WARN" | "FAIL"
        violations : Human-readable violation strings.
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
    if (
        metrics.get("confidence_stats")
        and metrics["confidence_stats"]["mean"] < confidence_fail_thresh
    ):
        violations.append(
            f"mean_confidence {metrics['confidence_stats']['mean']:.3f} < {confidence_fail_thresh}"
        )

    if not violations:
        return "OK", []
    if len(violations) == 1:
        return "WARN", violations
    return "FAIL", violations


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate topic modeling outputs")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--outlier-threshold", type=float, default=0.15,
        help="Max acceptable outlier ratio (default: 0.15)",
    )
    parser.add_argument(
        "--tiny-topic-threshold", type=int, default=3,
        help="Max acceptable number of tiny topics ≤3 docs (default: 3)",
    )
    parser.add_argument(
        "--concentration-threshold", type=float, default=0.30,
        help="Max acceptable largest-topic share of total documents (default: 0.30)",
    )
    parser.add_argument(
        "--confidence-fail-threshold", type=float, default=0.4,
        help="Mean confidence below this value triggers a violation (default: 0.4)",
    )
    args = parser.parse_args()

    logger.info(f"Loading topic assignments from {args.input}")
    try:
        assignments = load_assignments(args.input)
    except ValidationError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc

    logger.info("Computing validation metrics...")
    metrics = compute_raw_metrics(assignments)

    status, violations = evaluate_status(
        metrics,
        args.outlier_threshold,
        args.tiny_topic_threshold,
        args.concentration_threshold,
        args.confidence_fail_threshold,
    )
    metrics["status"] = status
    metrics["violations"] = violations
    metrics["thresholds_used"] = {
        "outlier": args.outlier_threshold,
        "tiny_topic": args.tiny_topic_threshold,
        "concentration": args.concentration_threshold,
        "confidence_fail": args.confidence_fail_threshold,
    }

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
    print(
        f"Topic size range:    "
        f"{metrics['topic_size_stats']['min']:.0f} – {metrics['topic_size_stats']['max']:.0f}"
    )
    print(f"Mean / std:          {metrics['topic_size_stats']['mean']} / {metrics['topic_size_stats']['std']}")
    print(f"Largest topic share: {metrics['largest_topic_share']:.1%}  (of all documents)")
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
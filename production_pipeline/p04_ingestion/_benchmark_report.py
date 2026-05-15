"""
_benchmark_report.py
====================
Print reports and save results for the retrieval benchmark.

Single responsibility: human-readable output + JSON serialization.
No metric computation, no data loading, no retrieval logic.

Functions:
    print_full_benchmark_report(summaries: list[MetricSummary]) -> None
    save_benchmark_results(summaries: list[MetricSummary], output_dir: Path) -> None
"""
import json
from pathlib import Path

from rag_pipeline.logging import get_logger

from ._benchmark_types import MetricSummary

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _format_summary(s: MetricSummary) -> list[str]:
    """Format a single MetricSummary into a list of display lines."""
    lines = [f"Config: {s.config_name} | Model: {s.model_name}"]
    if s.topic is not None:
        lines.append(f"  Topic: {s.topic} | Subtopic: {s.subtopic} | Intent: {s.intent}")
    ret_integrity = f"{s.avg_code_integrity_retrieved:.1%}" if s.avg_code_integrity_retrieved is not None else "N/A"
    lines += [
        f"  Queries: {s.num_queries}",
        f"  Hit Rate @1: {s.hit_rate_1:.1%} | @3: {s.hit_rate_3:.1%} | @5: {s.hit_rate_5:.1%} | @10: {s.hit_rate_10:.1%}",
        f"  MRR: {s.mrr:.4f} | NDCG@10: {s.ndcg_10:.4f}",
        f"  Latency: p50={s.latency_p50:.1f}ms | p95={s.latency_p95:.1f}ms | p99={s.latency_p99:.1f}ms",
        f"  Code Integrity (Ref): {s.avg_code_integrity_ref:.1%} | (Retrieved): {ret_integrity}",
    ]
    return lines


def _sort_summaries(summaries: list[MetricSummary]) -> list[MetricSummary]:
    """Sort summaries deterministically for consistent reporting."""
    return sorted(
        summaries,
        key=lambda s: (s.config_name, s.model_name, s.topic or -1, s.subtopic or -1, s.intent or "")
    )


def print_full_benchmark_report(summaries: list[MetricSummary]) -> None:
    """Print complete benchmark results without any truncation."""
    print("=" * 120)
    print("RETRIEVAL BENCHMARK REPORT")
    print("=" * 120)

    for s in _sort_summaries(summaries):
        if s.num_queries == 0:
            continue
        print()
        print("\n".join(_format_summary(s)))

    print("\n" + "=" * 120)


def save_benchmark_results(summaries: list[MetricSummary], output_dir: Path) -> None:
    """Save benchmark results to JSON and human-readable summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in summaries], f, indent=2, ensure_ascii=False)
    logger.info(f"Saved benchmark results: {results_path}")

    # Text summary
    summary_path = output_dir / "benchmark_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("RETRIEVAL BENCHMARK SUMMARY\n")
        f.write("=" * 120 + "\n\n")
        for s in _sort_summaries(summaries):
            if s.num_queries == 0:
                continue
            f.write("\n".join(_format_summary(s)) + "\n\n")
    logger.info(f"Saved benchmark summary: {summary_path}")
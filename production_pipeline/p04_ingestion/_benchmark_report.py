"""
_benchmark_report.py
====================
Print reports and save results for the retrieval benchmark.

Single responsibility: human-readable output + JSON serialization.
No metric computation, no data loading, no retrieval logic.

Functions:
    print_full_benchmark_report(summaries)  -> None
    save_benchmark_results(summaries, output_dir) -> None
    save_performance_summary(summaries, output_dir) -> None
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rag_pipeline.logging import get_logger
from ._benchmark_types import MetricSummary

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_summary(s: MetricSummary) -> list[str]:
    """Format a single MetricSummary into display lines."""
    lines = [f"Config: {s.config_name} | Model: {s.model_name}"]
    if s.topic is not None:
        lines.append(f"  Topic: {s.topic} | Subtopic: {s.subtopic}")

    ret_integrity = (
        f"{s.avg_code_integrity_retrieved:.1%}"
        if s.avg_code_integrity_retrieved is not None
        else "N/A"
    )

    lines += [
        f"  Queries: {s.num_queries}",
        f"  Hit Rate @1: {s.hit_rate_1:.1%} | @3: {s.hit_rate_3:.1%} | @5: {s.hit_rate_5:.1%} | @10: {s.hit_rate_10:.1%}",
        f"  MRR: {s.mrr:.4f} | NDCG@10: {s.ndcg_10:.4f}",
        f"  Latency: p50={s.latency_p50:.1f}ms | p95={s.latency_p95:.1f}ms | p99={s.latency_p99:.1f}ms",
        f"  Code Integrity (Ref): {s.avg_code_integrity_ref:.1%} | (Retrieved): {ret_integrity}",
        f"  Cross-Course Contamination: {s.cross_course_contamination:.2%}",
        f"  Rank Std Dev: {s.rank_std:.2f}   (lower is better)",
        f"  Failures (Hit@10=0): {s.failure_count}",
    ]

    if s.avg_failure_similarity is not None:
        lines.append(f"  Avg Failure Similarity: {s.avg_failure_similarity:.4f}")

    return lines


def _sort_summaries(summaries: list[MetricSummary]) -> list[MetricSummary]:
    """Sort summaries deterministically for consistent reporting."""
    return sorted(
        summaries,
        key=lambda s: (s.config_name, s.model_name, s.topic or -1, s.subtopic or -1),
    )


def _winner_key(row: dict[str, Any]) -> tuple[float, float, float]:
    """Tie-breaking sort key: MRR → Hit@1 → NDCG@10."""
    return (row.get("mrr", 0.0), row.get("hit_rate_1", 0.0), row.get("ndcg_10", 0.0))


def _short_model_name(full_name: str, max_len: int = 35) -> str:
    """Extract short model name from full path."""
    name = full_name.split('/')[-1]
    return name[:max_len] if len(name) > max_len else name


# ---------------------------------------------------------------------------
# Table Formatters
# ---------------------------------------------------------------------------

def print_winner_table(summaries: list[MetricSummary]) -> None:
    """Print a compact winner table showing best config per model."""
    if not summaries:
        print("No results to display.")
        return

    # Best config per model (highest MRR)
    best_per_model: dict[str, MetricSummary] = {}
    for s in summaries:
        if s.model_name not in best_per_model or s.mrr > best_per_model[s.model_name].mrr:
            best_per_model[s.model_name] = s

    print("\n" + "=" * 95)
    print("BEST CONFIG PER MODEL (by MRR)")
    print("=" * 95)
    print(f"{'Model':<35} {'Config':<22} {'H@1':>8} {'H@5':>8} {'MRR':>8} {'p50(ms)':>9}")
    print(f"{'-' * 35} {'-' * 22} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 9}")

    for s in sorted(best_per_model.values(), key=lambda x: x.mrr, reverse=True):
        model_short = _short_model_name(s.model_name, 34)
        print(f"{model_short:<35} "
              f"{s.config_name:<22} "
              f"{s.hit_rate_1:>7.1%} "
              f"{s.hit_rate_5:>7.1%} "
              f"{s.mrr:>7.4f} "
              f"{s.latency_p50:>8.1f}")

    # Overall winner
    winner = max(best_per_model.values(), key=lambda x: x.mrr)
    print(f"\n{'=' * 95}")
    print(f"🏆 OVERALL WINNER: {_short_model_name(winner.model_name)} with {winner.config_name}")
    print(f"   MRR: {winner.mrr:.4f} | H@1: {winner.hit_rate_1:.1%} | H@5: {winner.hit_rate_5:.1%}")
    print("=" * 95)


def print_comparison_table(summaries: list[MetricSummary]) -> None:
    """Print a compact comparison table across all model/config combinations."""
    if not summaries:
        print("No results to display.")
        return

    # Group by config
    by_config: dict[str, list[MetricSummary]] = defaultdict(list)
    for s in summaries:
        by_config[s.config_name].append(s)

    for config_name, config_summaries in by_config.items():
        print(f"\n{'=' * 100}")
        print(f"Config: {config_name}")
        print(f"{'=' * 100}")

        # Header
        print(f"{'Model':<35} {'H@1':>8} {'H@5':>8} {'H@10':>9} {'MRR':>8} {'p50(ms)':>9} {'Fail':>6}")
        print(f"{'-' * 35} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 8} {'-' * 9} {'-' * 6}")

        # Sort by MRR descending
        for s in sorted(config_summaries, key=lambda x: x.mrr, reverse=True):
            model_short = _short_model_name(s.model_name, 34)
            print(f"{model_short:<35} "
                  f"{s.hit_rate_1:>7.1%} "
                  f"{s.hit_rate_5:>7.1%} "
                  f"{s.hit_rate_10:>8.1%} "
                  f"{s.mrr:>7.4f} "
                  f"{s.latency_p50:>8.1f} "
                  f"{s.failure_count:>6}")

        # Highlight best MRR
        best = max(config_summaries, key=lambda x: x.mrr)
        print(f"\n🏆 Best for {config_name}: {_short_model_name(best.model_name)} (MRR={best.mrr:.4f})")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def print_full_benchmark_report(summaries: list[MetricSummary]) -> None:
    """
    Print benchmark results in clean table format.

    Shows:
        1. Winner table (best config per model)
        2. Per-config comparison tables
    """
    print("\n" + "=" * 100)
    print("RETRIEVAL BENCHMARK REPORT")
    print("=" * 100)

    # Show winner table first
    print_winner_table(summaries)

    # Then per-config tables
    print_comparison_table(summaries)


def save_benchmark_results(summaries: list[MetricSummary], output_dir: Path) -> None:
    """
    Save per-run benchmark results to JSON and a human-readable text summary.

    Also calls ``save_performance_summary`` to update the canonical
    ``benchmark_performance.json`` rankings file.

    Files written
    -------------
    ``benchmark_results.json``  — full results for every model/config combination
    ``benchmark_summary.txt``   — human-readable equivalent
    ``benchmark_performance.json`` — ranked view with winner (via save_performance_summary)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_summaries = _sort_summaries(summaries)

    # JSON results
    results_path = output_dir / "benchmark_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(
            [asdict(s) for s in sorted_summaries],
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"Saved benchmark results: {results_path}")

    # Text summary (detailed format)
    summary_path = output_dir / "benchmark_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("RETRIEVAL BENCHMARK SUMMARY\n")
        f.write("=" * 120 + "\n\n")
        for s in sorted_summaries:
            if s.num_queries == 0:
                continue
            f.write("\n".join(_format_summary(s)) + "\n\n")
    logger.info(f"Saved benchmark summary: {summary_path}")

    # Canonical rankings
    save_performance_summary(summaries, output_dir)


def save_performance_summary(
    summaries: list[MetricSummary],
    output_dir: Path,
) -> None:
    """
    Save ``benchmark_performance.json`` — a consolidated rankings view.

    Contains all model/config combinations ranked by MRR, per-model best-config
    rows, and an overall winner (tie-broken by Hit@1 then NDCG@10).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = [
        asdict(s) for s in summaries if s.num_queries > 0
    ]

    # Best config per model (highest MRR)
    best_per_model: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        m = row["model_name"]
        if m not in best_per_model or row["mrr"] > best_per_model[m]["mrr"]:
            best_per_model[m] = row

    winner = max(best_per_model.values(), key=_winner_key, default=None)

    payload: dict[str, Any] = {
        "winner_model":  winner["model_name"] if winner else None,
        "winner_config": winner["config_name"] if winner else None,
        "winner_mrr":    winner["mrr"] if winner else None,
        "best_per_model": sorted(
            best_per_model.values(), key=lambda r: r["mrr"], reverse=True
        ),
        "all_results": sorted(all_rows, key=lambda r: r["mrr"], reverse=True),
    }

    perf_path = output_dir / "benchmark_performance.json"
    with perf_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved performance summary: {perf_path}")
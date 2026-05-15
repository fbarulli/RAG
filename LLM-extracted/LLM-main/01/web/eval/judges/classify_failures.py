#!/usr/bin/env python3
"""
classify_failures.py
====================
Loads judge results CSV (from eval/run_judge.py) and allows manual classification
of queries where the judge said the context does NOT answer the question.
Each failure is shown with its query, retrieved contexts, and verdicts.
User assigns a category (C/G/R/E/S) – results saved to a new CSV.

Usage:
    uv run python3 classify_failures.py
    uv run python3 classify_failures.py --judge-csv experiments/judge/20250512_123456_judge_results.csv
"""
import csv
import sys
import glob
import json
import argparse
from datetime import datetime
from pathlib import Path

# Module-level constants
OUTPUT_FILE = "failure_classifications.csv"
FIELDNAMES = [
    "query",
    "expected_id",
    "found_id",
    "subset",
    "judge_verdicts",
    "category",
    "reason_notes",
]

# Classification categories
CATEGORIES = {
    "C": "Corpus gap (FAQ lacks answer)",
    "G": "Generation poor (answer in context but LLM ignored)",
    "R": "Retrieval wrong (wrong document retrieved)",
    "E": "Eval artifact (malformed query / bad paraphrase / truncated)",
    "S": "Skip – cannot decide",
}


def load_judge_failures(judge_csv_path: str) -> list[dict]:
    """Load rows with judge_any_yes=False from judge results CSV."""
    failures = []
    with open(judge_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("judge_any_yes", "").strip().lower() in ("false", "0", "no"):
                failures.append(row)
    return failures


def load_contexts_from_benchmark(benchmark_config: str = "bm25_default", k: int = 5) -> dict:
    """
    Load contexts from benchmark results (k=5, success=True) into a dict keyed by query.
    """
    path = f"experiments/results/{benchmark_config}.json"
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Benchmark file {path} not found.")
        sys.exit(1)

    ctx_map = {}
    seen = set()
    for r in data["results"]:
        if r.get("k") == k and r.get("success") is True:
            q = r["query"]
            if q not in seen:
                seen.add(q)
                ctx_map[q] = r.get("contexts", [])
    return ctx_map


def display_failure(idx: int, total: int, row: dict, contexts: list, verdicts_str: str):
    """Pretty print a failure for manual classification."""
    print("\n" + "=" * 80)
    print(f"Failure {idx}/{total}")
    print("=" * 80)
    print(f"Query: {row['query']}")
    print(f"Expected ID: {row['expected_id']}")
    print(f"Found ID: {row['found_id']}")
    print(f"Subset: {row.get('subset', 'unknown')}")
    print(f"Judge verdicts: {verdicts_str}")
    print("\nRetrieved contexts (up to 5):")
    for i, ctx in enumerate(contexts[:5], start=1):
        # Truncate for display, but full context is stored in the original JSON
        preview = ctx[:500] + ("..." if len(ctx) > 500 else "")
        print(f"\n--- Context {i} ---")
        print(preview)
    print("\nCategories:")
    for key, desc in CATEGORIES.items():
        print(f"  {key} - {desc}")
    print("  (Enter just the letter, or add a reason after a colon, e.g., 'C: missing documentation')")


def classify_failures():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge-csv",
        type=str,
        default=None,
        help="Path to judge results CSV (default: most recent in experiments/judge/)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="bm25_default",
        help="Benchmark config to pull contexts from (default: bm25_default)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=50,
        help="Maximum number of failures to classify (default: 50)",
    )
    args = parser.parse_args()

    # Find judge results CSV
    if args.judge_csv:
        judge_csv = args.judge_csv
    else:
        files = sorted(glob.glob("experiments/judge/*_judge_results.csv"))
        if not files:
            print("No judge results CSV found in experiments/judge/. Run eval/run_judge.py first.")
            return
        judge_csv = files[-1]
        print(f"Using latest judge results: {judge_csv}")

    # Load failures from judge CSV
    failures = load_judge_failures(judge_csv)
    if not failures:
        print("No judge failures (judge_any_yes = False) found in the CSV. Exiting.")
        return

    print(f"Loaded {len(failures)} judge failures.")

    # Load contexts from benchmark
    contexts_by_query = load_contexts_from_benchmark(args.config, k=5)
    print(f"Loaded contexts for {len(contexts_by_query)} unique queries from {args.config}.")

    # Classify up to max failures
    results = []
    classified = 0
    for idx, row in enumerate(failures, start=1):
        if classified >= args.max:
            break
        query = row["query"]
        contexts = contexts_by_query.get(query, [])
        if not contexts:
            print(f"\n[WARN] No contexts found for query: {query[:60]}... skipping.")
            continue

        verdicts = row.get("judge_verdicts", "[]")
        display_failure(classified + 1, min(len(failures), args.max), row, contexts, verdicts)

        while True:
            ans = input("\nYour classification (C/G/R/E/S) [or Q to quit]: ").strip()
            if ans.upper() == "Q":
                break
            if ans and ans[0].upper() in CATEGORIES:
                category = ans[0].upper()
                # Extract optional reason after colon
                reason = ""
                if ":" in ans:
                    reason = ans.split(":", 1)[1].strip()
                results.append(
                    {
                        "query": query,
                        "expected_id": row["expected_id"],
                        "found_id": row["found_id"],
                        "subset": row.get("subset", ""),
                        "judge_verdicts": verdicts,
                        "category": category,
                        "reason_notes": reason,
                    }
                )
                classified += 1
                break
            else:
                print("Invalid input. Please enter C, G, R, E, S, or Q to quit.")

        if ans.upper() == "Q":
            print("Exiting early.")
            break

    # Save results
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} classifications to {OUTPUT_FILE}")
    print("\nCategory distribution:")
    from collections import Counter

    cat_counts = Counter(r["category"] for r in results)
    for cat, count in cat_counts.items():
        desc = CATEGORIES.get(cat, "Unknown")
        print(f"  {cat}: {count} ({desc})")


if __name__ == "__main__":
    classify_failures()
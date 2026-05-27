# ablation/report.py
"""
Print summary table across all completed experiments, with per-query-type breakdown.

Usage:
    uv run python -m ablation report
"""
import json
from pathlib import Path
from typing import Optional
from ablation.compare import compare, breakdown

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_all() -> list[dict]:
    return [json.load(open(p)) for p in sorted(RESULTS_DIR.glob("*_meta.json"))]


def print_report(results: Optional[list[dict]] = None) -> None:
    if results is None:
        results = load_all()
    if not results:
        print("No experiment results found in ablation/results/")
        return

    # overall summary
    print(f"\n{'Experiment':<25} {'Patch':<15} {'Config':<22} {'H@1':>7} {'H@5':>7} {'MRR':>8}")
    print("-" * 86)
    for r in results:
        for cfg, m in r.get("metrics", {}).items():
            h1  = m.get("h1",  "?")
            h5  = m.get("h5",  "?")
            mrr = m.get("mrr", "?")
            print(f"{r['name']:<25} {r['patch']:<15} {cfg:<22} "
                  f"{h1:>7.1%} {h5:>7.1%} {mrr:>8.4f}")

    # per-query-type breakdown comparing each experiment to baseline
    baseline_files = [
        f.stem.replace("_meta", "")
        for f in RESULTS_DIR.glob("baseline_meta.json")
    ]
    if not baseline_files:
        return

    print("\n--- Per-query-type breakdown vs baseline ---")
    for r in results:
        if r["name"] == "baseline":
            continue
        for cfg in r.get("configs", []):
            exp_a = f"baseline__{cfg}"
            exp_b = f"{r['name']}__{cfg}"
            try:
                rows = compare(exp_a, exp_b)
            except FileNotFoundError:
                continue
            bd = breakdown(rows)
            print(f"\n  {r['name']} vs baseline  [{cfg}]")
            print(f"  {'query_type':<20} {'n':>5} {'baseline H@1':>13} {'exp H@1':>9} {'delta':>7}")
            print(f"  {'-'*58}")
            for qt, s in bd.items():
                print(f"  {qt:<20} {s['n']:>5} {s['h1_a']:>13.1%} {s['h1_b']:>9.1%} {s['delta_h1']:>+7.1%}")


if __name__ == "__main__":
    print_report()

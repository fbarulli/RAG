# src/rag_pipeline/ablation/report.py
"""
Print summary table across all completed experiments, with per-query-type breakdown.

Usage:
    uv run python -m rag_pipeline.ablation report
"""
import json
import sys
from pathlib import Path
from typing import Optional

from rag_pipeline.ablation.compare import compare, breakdown
from rag_pipeline.core.paths import Paths


def _results_dir() -> Path:
    return Paths.ablation_results_dir()


def load_all() -> list[dict]:
    return [json.load(open(p)) for p in sorted(_results_dir().glob("*_meta.json"))]


def check_regressions(results: list[dict], ci: bool = False) -> bool:
    """Compare each experiment to baseline.json. Returns True if any regression found."""
    baseline_path = Paths.experiments_dir() / "baseline.json"
    if not baseline_path.exists():
        if ci:
            print("[ERROR] No baseline.json found. Regression check is REQUIRED for CI.")
            return True # Signal failure in CI
        print("[WARN] No baseline.json found — skipping regression check")
        return False

    baseline = json.load(open(baseline_path))
    ablation_cfg = json.load(open(Paths.ablation_config()))
    threshold = ablation_cfg.get("regression_threshold_pp", 2.0) / 100

    regressions = []
    for r in results:
        if r["name"] == "baseline":
            continue
        for cfg, m in r.get("metrics", {}).items():
            base_m = baseline.get("metrics", {}).get(cfg, {})
            base_h1 = base_m.get("h1")
            exp_h1  = m.get("h1")
            if base_h1 is None or exp_h1 is None:
                continue
            delta = exp_h1 - base_h1
            if delta < -threshold:
                regressions.append((r["name"], cfg, base_h1, exp_h1, delta))

    if regressions:
        print()
        print('=' * 60)
        print("REGRESSIONS DETECTED")
        print('=' * 60)
        for name, cfg, base, exp, delta in regressions:
            print(f"  {name:<25} [{cfg}]  baseline={base:.1%}  exp={exp:.1%}  delta={delta:+.1%}")
        print('=' * 60)
        print()
        return True

    if ci:
        print("[ci] No regressions detected — all experiments within threshold")
    return False


def print_report(results: Optional[list[dict]] = None, ci: bool = False) -> None:
    if results is None:
        results = load_all()
    if not results:
        print(f"No experiment results found in {_results_dir()}")
        return

    # overall summary
    print()
    print(f"{'Experiment':<25} {'Patch':<15} {'Config':<22} {'H@1':>7} {'H@5':>7} {'MRR':>8}")
    print("-" * 86)
    for r in results:
        for cfg, m in r.get("metrics", {}).items():
            h1  = m.get("h1",  "?")
            h5  = m.get("h5",  "?")
            mrr = m.get("mrr", "?")
            # Format as percentages if they are numbers, else print as is
            h1_str = f"{h1:>7.1%}" if isinstance(h1, (int, float)) else f"{h1:>7}"
            h5_str = f"{h5:>7.1%}" if isinstance(h5, (int, float)) else f"{h5:>7}"
            mrr_str = f"{mrr:>8.4f}" if isinstance(mrr, (int, float)) else f"{mrr:>8}"
            
            print(f"{r['name']:<25} {r['patch']:<15} {cfg:<22} {h1_str} {h5_str} {mrr_str}")

    # per-query-type breakdown comparing each experiment to baseline
    baseline_files = [
        f.stem.replace("_meta", "")
        for f in _results_dir().glob("baseline_meta.json")
    ]
    if not baseline_files:
        if ci and check_regressions(results, ci=ci):
            sys.exit(1)
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
            print(f"  {'-' * 58}")
            for qt, s in bd.items():
                print(
                    f"  {qt:<20} {s['n']:>5} {s['h1_a']:>13.1%} "
                    f"{s['h1_b']:>9.1%} {s['delta_h1']:>+7.1%}"
                )

    if check_regressions(results, ci=ci):
        if ci:
            sys.exit(1)


if __name__ == "__main__":
    print_report()

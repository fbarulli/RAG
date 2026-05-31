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
    from rag_pipeline.core.models import ExperimentResult
    return [ExperimentResult.model_validate(json.load(open(p))).model_dump() for p in sorted(_results_dir().glob("*_meta.json"))]


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
            exp_h1  = m.get("hit_rate_1", m.get("h1"))
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


QUERY_TYPES = ["chaos_monkey", "creative_student", "grounded_analyst", "original"]


def _qt_breakdown(name: str, cfg: str) -> dict:
    # Return H@1 per query_type for a given experiment+config, or {} if missing.
    import json
    from collections import defaultdict
    jsonl = _results_dir() / f"{name}__{cfg}_query_results.jsonl"
    if not jsonl.exists():
        return {}
    buckets: dict = defaultdict(list)
    with open(jsonl) as f:
        for line in f:
            row = json.loads(line)
            qt = row.get("query_type", "unknown")
            hit = row.get("hit_ids", [])
            expected = row.get("expected_id", "")
            buckets[qt].append(int(bool(hit and hit[0] == expected)))
    return {qt: sum(v) / len(v) for qt, v in buckets.items() if v}


def _failure_rate(m: dict):
    fc = m.get("failure_count")
    nq = m.get("num_queries") or 1
    return fc / nq if isinstance(fc, (int, float)) else m.get("failure_rate", "?")

def _fmt_row(name, patch, cfg, m, qt_data, widths):
    E, P, C, qt_w = widths
    h1      = m.get("hit_rate_1", m.get("h1", "?"))
    h5      = m.get("hit_rate_5", m.get("h5", "?"))
    mrr     = m.get("mrr",         "?")
    ndcg    = m.get("ndcg_10",     "?")
    p50     = m.get("latency_p50", "?")
    fail    = _failure_rate(m)
    h1_str   = f"{h1:>7.1%}"   if isinstance(h1,   (int, float)) else f"{h1:>7}"
    h5_str   = f"{h5:>7.1%}"   if isinstance(h5,   (int, float)) else f"{h5:>7}"
    mrr_str  = f"{mrr:>8.4f}"  if isinstance(mrr,  (int, float)) else f"{mrr:>8}"
    ndcg_str = f"{ndcg:>9.4f}" if isinstance(ndcg, (int, float)) else f"{ndcg:>9}"
    p50_str  = f"{p50:>8.1f}"  if isinstance(p50,  (int, float)) else f"{p50:>8}"
    fail_str = f"{fail:>8.1%}" if isinstance(fail, (int, float)) else f"{fail:>8}"
    qt_str = "".join(
        f" {qt_data[qt]:>{qt_w}.1%}" if qt in qt_data else f" {chr(8212):>{qt_w}}"
        for qt in QUERY_TYPES
    )
    return f"{name:<{E}} {patch:<{P}} {cfg:<{C}}{h1_str} {h5_str} {mrr_str}{ndcg_str}{p50_str}{fail_str}{qt_str}"


def print_report(results: Optional[list[dict]] = None, ci: bool = False) -> None:
    if results is None:
        results = load_all()
    if not results:
        print(f"No experiment results found in {_results_dir()}")
        return

    qt_w = 9
    rows_data = []
    for r in results:
        for cfg, m in r.get("metrics", {}).items():
            h1 = m.get("hit_rate_1", m.get("h1", 0.0))
            qt_data = _qt_breakdown(r["name"], cfg)
            rows_data.append((r["name"], r["patch"], cfg, m, h1, qt_data))

    E = max(len(x[0]) for x in rows_data) + 2
    P = max(len(x[1]) for x in rows_data) + 2
    C = max(len(x[2]) for x in rows_data) + 2
    widths = (E, P, C, qt_w)

    qt_abbrev = {"chaos_monkey": "chaos", "creative_student": "creative",
                 "grounded_analyst": "grounded", "original": "original"}
    header = (
        f"{'Experiment':<{E}} {'Patch':<{P}} {'Config':<{C}}"
        f" {'H@1':>7} {'H@5':>7} {'MRR':>8} {'NDCG@10':>8} {'p50ms':>7} {'fail':>7}"
        + "".join(f" {qt_abbrev.get(qt, qt):>{qt_w}}" for qt in QUERY_TYPES)
    )
    divider = "-" * len(header)
    sorted_rows = sorted(rows_data, key=lambda x: x[4], reverse=True)
    mid = len(sorted_rows) // 2

    print()
    print("  TOP PERFORMERS")
    print(header)
    print(divider)
    for name, patch, cfg, m, h1, qt_data in sorted_rows[:mid]:
        print(_fmt_row(name, patch, cfg, m, qt_data, widths))

    print()
    print("  BOTTOM PERFORMERS")
    print(header)
    print(divider)
    for name, patch, cfg, m, h1, qt_data in sorted_rows[mid:]:
        print(_fmt_row(name, patch, cfg, m, qt_data, widths))

    if check_regressions(results, ci=ci):
        if ci:
            sys.exit(1)

if __name__ == "__main__":
    print_report()

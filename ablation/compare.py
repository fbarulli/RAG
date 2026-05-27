# ablation/compare.py
"""
Compare two experiment result files query-by-query.

Usage:
    from ablation.compare import compare, summary, breakdown, print_diff
    rows = compare("baseline__entity_boosted", "no_category__entity_boosted")
    print(summary(rows))
    print(breakdown(rows))
"""
import json
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _load(name: str) -> dict[str, dict]:
    path = RESULTS_DIR / f"{name}_query_results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No results file: {path}")
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["query_id"]] = r
    return records


def compare(exp_a: str, exp_b: str, top_k: int = 1) -> list[dict]:
    a = _load(exp_a)
    b = _load(exp_b)
    rows = []
    for qid in sorted(set(a) | set(b)):
        ra, rb = a.get(qid), b.get(qid)
        if ra is None or rb is None:
            continue
        expected = ra["expected_id"]
        hit_a = expected in ra["hit_ids"][:top_k]
        hit_b = expected in rb["hit_ids"][:top_k]
        rows.append({
            "query_id":   qid,
            "query_text": ra["query_text"],
            "expected_id": expected,
            "course":     ra["course"],
            "query_type": ra.get("query_type", "unknown"),
            "hit_a":      hit_a,
            "hit_b":      hit_b,
            "delta":      int(hit_b) - int(hit_a),
        })
    return rows


def summary(rows: list[dict]) -> dict:
    total  = len(rows)
    hit_a  = sum(1 for r in rows if r["hit_a"])
    hit_b  = sum(1 for r in rows if r["hit_b"])
    wins   = sum(1 for r in rows if r["delta"] ==  1)
    losses = sum(1 for r in rows if r["delta"] == -1)
    return {
        "total_queries": total,
        "h1_a":    round(hit_a / total, 4),
        "h1_b":    round(hit_b / total, 4),
        "delta_h1": round((hit_b - hit_a) / total, 4),
        "b_beats_a": wins,
        "a_beats_b": losses,
        "neutral":   total - wins - losses,
    }


def breakdown(rows: list[dict]) -> dict[str, dict]:
    """H@1 and delta broken down by query_type."""
    buckets: dict[str, list] = defaultdict(list)
    for r in rows:
        buckets[r["query_type"]].append(r)
    result = {}
    for qt, qrows in sorted(buckets.items()):
        n     = len(qrows)
        hit_a = sum(1 for r in qrows if r["hit_a"])
        hit_b = sum(1 for r in qrows if r["hit_b"])
        result[qt] = {
            "n":       n,
            "h1_a":    round(hit_a / n, 4),
            "h1_b":    round(hit_b / n, 4),
            "delta_h1": round((hit_b - hit_a) / n, 4),
            "b_beats_a": sum(1 for r in qrows if r["delta"] ==  1),
            "a_beats_b": sum(1 for r in qrows if r["delta"] == -1),
        }
    return result


def print_diff(rows: list[dict], show: str = "all") -> None:
    filtered = [r for r in rows if (
        show == "all"   and r["delta"] != 0 or
        show == "wins"  and r["delta"] ==  1 or
        show == "losses" and r["delta"] == -1
    )]
    for r in filtered:
        symbol = "+" if r["delta"] == 1 else "-"
        print(f"[{symbol}] [{r['query_type']:<18}] [{r['course'][:25]:<25}] {r['query_text'][:80]}")

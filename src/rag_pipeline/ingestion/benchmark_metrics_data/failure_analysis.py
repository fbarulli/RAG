"""
Failure triage for benchmark JSONL results.

CLI via configs/benchmark_cli.py:
    # summary
uv run python -m rag_pipeline.ingestion.benchmark_metrics_data.failure_analysis \
    experiments/results/ablation/llm_ner__entity_boosted_query_results.jsonl \
    src/rag_pipeline/eda/topics/output/topic_assignments_all.json

# with detail
uv run python -m rag_pipeline.ingestion.benchmark_metrics_data.failure_analysis \
    experiments/results/ablation/llm_ner__entity_boosted_query_results.jsonl \
    src/rag_pipeline/eda/topics/output/topic_assignments_all.json \
    --detail no_entity
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

W = 78  # terminal width


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(n: int, total: int, width: int = 20) -> str:
    filled = round(width * n / max(total, 1))
    return "█" * filled + "░" * (width - filled)

def _rule(char: str = "─") -> str:
    return char * W

def _pct(n: int, d: int) -> str:
    return f"{100*n/max(d,1):.0f}%"

def _q(text: str, width: int = W - 5) -> str:
    text = text.replace("\n", " ").strip()
    return text[:width] + "…" if len(text) > width else text


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FailureRecord:
    query_id: str
    query_text: str
    query_type: str
    expected_id: str
    hit_ids: list[str]
    rank: Optional[int]
    entity: Optional[str]
    entity_cluster_size: int
    course: str

    @classmethod
    def from_result(cls, r: dict, cluster_size: Counter) -> "FailureRecord":
        e = r.get("ner_primary_entity")
        return cls(
            query_id=r.get("query_id", ""),
            query_text=r.get("query_text", ""),
            query_type=r.get("query_type", ""),
            expected_id=r.get("expected_id", ""),
            hit_ids=r.get("hit_ids", []),
            rank=r.get("rank"),
            entity=e,
            entity_cluster_size=cluster_size.get(e, 0) if e else 0,
            course=r.get("course", ""),
        )

    @property
    def cluster_bucket(self) -> str:
        if not self.entity:
            return "no_entity"
        n = self.entity_cluster_size
        if n <= 1:  return "1"
        if n <= 5:  return "2-5"
        if n <= 15: return "6-15"
        if n <= 40: return "16-40"
        return "40+"

    @property
    def is_rank2(self) -> bool:
        return self.rank == 2


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_results(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_cluster_sizes(
    assignments_path: Path,
    model_key: str = "BAAI/bge-base-en-v1.5",
) -> Counter:
    d = json.loads(assignments_path.read_text())
    assignments = d["results"][model_key]["assignments"]
    return Counter(
        a.get("ner_primary_entity")
        for a in assignments
        if a.get("ner_primary_entity")
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def triage(results: list[dict], cluster_size: Counter) -> dict:
    failures = [
        FailureRecord.from_result(r, cluster_size)
        for r in results
        if not r.get("hit_at_1", False)
    ]
    by_bucket: dict[str, list[FailureRecord]] = defaultdict(list)
    for f in failures:
        by_bucket[f.cluster_bucket].append(f)

    return {
        "total_queries": len(results),
        "total_failures": len(failures),
        "rank2_failures": sum(f.is_rank2 for f in failures),
        "by_bucket": dict(by_bucket),
        "by_query_type": dict(Counter(f.query_type for f in failures)),
        "actionable_failures": [f for f in failures if f.query_type != "chaos_monkey"],
    }


# ---------------------------------------------------------------------------
# Printers
# ---------------------------------------------------------------------------

def print_summary(t: dict, path: Path) -> None:
    n, f, r2 = t["total_queries"], t["total_failures"], t["rank2_failures"]

    print(_rule())
    print(f" FAILURE TRIAGE  {path.name}")
    print(_rule())
    print(f" Queries: {n}   Failures: {f} ({_pct(f,n)})   "
          f"Rank-2: {r2}/{f} ({_pct(r2,f)}) — cross-encoder candidates")


def print_modes(t: dict) -> None:
    buckets = t["by_bucket"]
    no_ent  = len(buckets.get("no_entity", []))
    rerank  = sum(
        1 for label in ["1","2-5","6-15","16-40","40+"]
        for f in buckets.get(label, []) if f.is_rank2
    )
    routing = t["total_failures"] - t["rank2_failures"]

    print(f"\n FAILURE MODES")
    print(f" {'─'*60}")
    print(f" {'Mode A  reranking':<28} correct doc retrieved, wrong rank   "
          f"{t['rank2_failures']:3d}  {_bar(t['rank2_failures'], t['total_failures'])}  "
          f"{_pct(t['rank2_failures'], t['total_failures'])}")
    print(f" {'Mode B  routing':<28} no entity or rank > 2               "
          f"{routing:3d}  {_bar(routing, t['total_failures'])}  "
          f"{_pct(routing, t['total_failures'])}")
    print(f"   {'└─ no entity assigned':<26} {no_ent:3d}")
    print(f"   {'└─ entity present, rank>2':<26} {routing - no_ent:3d}")


def print_cluster_table(t: dict) -> None:
    print(f"\n BY CLUSTER SIZE")
    print(f" {'─'*50}")
    print(f" {'bucket':<12} {'failures':>8}   {'rank-2':>6}   {'rank>2':>6}")
    print(f" {'─'*50}")
    for label in ["no_entity", "1", "2-5", "6-15", "16-40", "40+"]:
        items = t["by_bucket"].get(label, [])
        if not items:
            continue
        r2    = sum(f.is_rank2 for f in items)
        notR2 = len(items) - r2
        print(f" {label:<12} {len(items):>8}   {r2:>6}   {notR2:>6}")


def print_query_type_row(t: dict) -> None:
    print(f"\n BY QUERY TYPE")
    print(f" {'─'*50}")
    row = "  ".join(f"{k}: {v}" for k, v in sorted(t["by_query_type"].items()))
    print(f" {row}")


def print_detail(t: dict, bucket: str) -> None:
    items = t["by_bucket"].get(bucket, [])
    print(f"\n{_rule()}")
    print(f" {bucket}  ·  {len(items)} failures")
    print(_rule())
    for i, f in enumerate(items, 1):
        got = f.hit_ids[0] if f.hit_ids else "—"
        match = "✓" if got == f.expected_id else "✗"
        rank_str = str(f.rank) if f.rank else "None"
        print(f" {i}/{len(items)}  rank={rank_str:<4}  {f.query_type:<20}  {f.course}")
        print(f"   Q  {_q(f.query_text)}")
        ent_str = f"{f.entity} (cluster={f.entity_cluster_size})" if f.entity else "—"
        print(f"   E  {ent_str}")
        print(f"   {match}  got={got:<14}  exp={f.expected_id}")
        if i < len(items):
            print(f" {'·'*W}")


def print_triage(t: dict, path: Path) -> None:
    print_summary(t, path)
    print_modes(t)
    print_cluster_table(t)
    print_query_type_row(t)
    print(f"\n{_rule()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Triage benchmark failures by cluster size and failure mode.")
    p.add_argument("results_jsonl",    type=Path)
    p.add_argument("assignments_json", type=Path)
    p.add_argument("--model-key", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--detail", choices=["no_entity","1","2-5","6-15","16-40","40+"])
    args = p.parse_args()

    results      = load_results(args.results_jsonl)
    cluster_size = load_cluster_sizes(args.assignments_json, args.model_key)
    t            = triage(results, cluster_size)

    print_triage(t, args.results_jsonl)
    if args.detail:
        print_detail(t, args.detail)
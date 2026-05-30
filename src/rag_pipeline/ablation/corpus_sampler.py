# rag_pipeline/ablation/corpus_sampler.py
"""
Corpus size ablation — quantify when retrieval performance degrades.

Trains the pipeline on progressively smaller corpus fractions (100% → 80% → … → 20%),
benchmarks only on ``original`` queries, and prints a summary table.

Strategy: re-ingests into the real Qdrant collection at each fraction
(overwriting it), runs benchmark, then restores the full corpus at the end.

Usage:
    uv run python -m rag_pipeline.ablation.corpus_sampler
    uv run python -m rag_pipeline.ablation.corpus_sampler --fractions 1.0 0.6 0.2
    uv run python -m rag_pipeline.ablation.corpus_sampler --config entity_boosted --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from rag_pipeline.core.paths import Paths
from rag_pipeline.logging import get_logger

logger = get_logger(__name__)

DEFAULT_FRACTIONS = [1.0, 0.8, 0.6, 0.4, 0.2]
DEFAULT_SEED      = 42


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------



def _stratified_sample(docs: list[dict], fraction: float, seed: int) -> list[dict]:
    """Sample ``fraction`` of docs stratified by course."""
    if fraction >= 1.0:
        return list(docs)

    random.seed(seed)
    by_course: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        by_course[d.get("course", "unknown")].append(d)

    sampled: list[dict] = []
    for pool in by_course.values():
        k = max(1, round(len(pool) * fraction))
        k = min(k, len(pool))
        sampled.extend(random.sample(pool, k))

    random.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _run(cmd: str, env: dict | None = None) -> None:
    merged = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, shell=True, env=merged)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}")


def _parse_benchmark_output(
    output_dir: Path, config: str, model: str
) -> tuple[float | None, float | None]:
    """Find H@1 and MRR from benchmark_results.json written by save_benchmark_results."""
    candidates = list(output_dir.glob("*.json")) + list(output_dir.glob("**/*.json"))
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, list):
            for row in data:
                if row.get("config_name") == config and row.get("model_name") == model:
                    h1  = row.get("hit_rate_1")
                    mrr = row.get("mrr")
                    if h1 is not None:
                        return float(h1), float(mrr) if mrr is not None else None
    return None, None


# ---------------------------------------------------------------------------
# Per-fraction run
# ---------------------------------------------------------------------------

def run_fraction(
    fraction: float,
    docs: list[dict],
    model: str,
    config: str,
    seed: int,
    tmp_dir: Path,
    host: str,
    port: int,
    query_type: str = "grounded_analyst",
) -> dict:
    sampled = _stratified_sample(docs, fraction, seed)
    n = len(sampled)
    label = f"frac_{int(fraction * 100):03d}"

    logger.info("--- Fraction %.0f%% (%d / %d docs) ---", fraction * 100, n, len(docs))

    # Write sampled corpus to temp file
    corpus_path = tmp_dir / f"corpus_{label}.jsonl"
    with open(corpus_path, "w") as f:
        for doc in sampled:
            f.write(json.dumps(doc) + "\n")

    # 1. Re-ingest (overwrites the real collection)
    _run(
        f'uv run python -m rag_pipeline.ingestion.ingest_models '
        f'--models "{model}" '
        f'--clean-path "{corpus_path}" '
        f'--qdrant-host {host} '
        f'--qdrant-port {port}'
    )

    # 2. Benchmark — original queries only, isolated output dir
    bench_dir = tmp_dir / f"bench_{label}"
    bench_dir.mkdir()
    _run(
        f'uv run python -m rag_pipeline.ingestion.benchmark '
        f'--model "{model}" '
        f'--configs {config} '
        f'--query-type {query_type} '
        f'--output-dir "{bench_dir}" '
        f'--no-resume '
        f'--reset '
        f'--no-skip-existing '
        f'--qdrant-host {host} '
        f'--qdrant-port {port}'
    )

    # 3. Parse results
    h1, mrr = _parse_benchmark_output(bench_dir, config, model)
    if h1 is None:
        logger.warning("Could not parse H@1 from benchmark output in %s", bench_dir)

    return {"fraction": fraction, "n_docs": n, "h1": h1, "mrr": mrr}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(rows: list[dict], config: str, query_type: str = "original") -> None:
    print(f"\n{'=' * 62}")
    print(f"  Corpus Size Ablation | config={config} | query_type={query_type}")
    print(f"{'=' * 62}")
    print(f"  {'Fraction':>8}  {'N docs':>7}  {'H@1':>8}  {'MRR':>8}  {'ΔH@1':>8}")
    print(f"  {'-' * 58}")

    baseline_h1 = None
    for row in rows:
        h1  = row["h1"]
        mrr = row["mrr"]
        h1_str  = f"{h1:.1%}"  if isinstance(h1,  float) else "—"
        mrr_str = f"{mrr:.4f}" if isinstance(mrr, float) else "—"

        if baseline_h1 is None and isinstance(h1, float):
            baseline_h1 = h1
            delta_str = "baseline"
        elif isinstance(h1, float) and baseline_h1 is not None:
            delta_str = f"{h1 - baseline_h1:+.1%}"
        else:
            delta_str = "—"

        print(
            f"  {row['fraction']:>7.0%}  {row['n_docs']:>7d}  "
            f"{h1_str:>8}  {mrr_str:>8}  {delta_str:>8}"
        )
    print(f"{'=' * 62}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    defaults = Paths.defaults()
    qdrant   = defaults.get("qdrant", {})

    parser = argparse.ArgumentParser(
        prog="rag_pipeline.ablation.corpus_sampler",
        description="Benchmark H@1 vs corpus size on original queries only",
    )
    parser.add_argument(
        "--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS,
        help="Corpus fractions to test, largest first (default: 1.0 0.8 0.6 0.4 0.2)",
    )
    parser.add_argument(
        "--model", default=defaults.get("production_model"),
        help="Embedding model (default: production_model from defaults.json)",
    )
    parser.add_argument(
        "--config", default=defaults.get("production_config", "entity_boosted"),
        help="Retrieval config name (default: production_config from defaults.json)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--input", type=Path, default=Paths.input_file("eda"),
        help="Corpus JSONL (default: eda input from paths.json)",
    )
    parser.add_argument(
        "--query-type", default="grounded_analyst",
        help="Query type to benchmark (default: grounded_analyst)",
    )
    parser.add_argument(
        "--host", default=qdrant.get("host", "localhost"),
        help="Qdrant host (default: qdrant.host from defaults.json)",
    )
    parser.add_argument(
        "--port", type=int, default=qdrant.get("port", 6333),
        help="Qdrant port (default: qdrant.port from defaults.json)",
    )
    args = parser.parse_args()

    fractions = sorted(args.fractions, reverse=True)  # largest first → baseline first

    docs = [json.loads(line) for line in open(args.input) if line.strip()]
    logger.info(
        "Loaded %d docs | courses=%s",
        len(docs),
        dict(Counter(d.get("course") for d in docs)),
    )

    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="corpus_sampler_") as tmp:
        tmp_dir = Path(tmp)
        for frac in fractions:
            try:
                row = run_fraction(
                    fraction=frac,
                    docs=docs,
                    model=args.model,
                    config=args.config,
                    seed=args.seed,
                    query_type=args.query_type,
                    tmp_dir=tmp_dir,
                    host=args.host,
                    port=args.port,
                )
                rows.append(row)
                logger.info(
                    "Fraction %.0f%% → H@1=%s  MRR=%s",
                    frac * 100,
                    f"{row['h1']:.1%}" if isinstance(row["h1"], float) else "?",
                    f"{row['mrr']:.4f}" if isinstance(row["mrr"], float) else "?",
                )
            except Exception as e:
                logger.error("Fraction %.0f%% failed: %s", frac * 100, e)
                rows.append({"fraction": frac, "n_docs": 0, "h1": None, "mrr": None})

    # Always restore full corpus at the end
    logger.info("Restoring full corpus (%d docs)…", len(docs))
    try:
        _run(
            f'uv run python -m rag_pipeline.ingestion.ingest_models '
            f'--models "{args.model}" '
            f'--clean-path "{args.input}" '
            f'--qdrant-host {args.host} '
            f'--qdrant-port {args.port}'
        )
        logger.info("Full corpus restored.")
    except Exception as e:
        logger.error("RESTORE FAILED — re-run ingest manually: %s", e)

    _print_report(rows, args.config, args.query_type)

    results_dir = Paths.ablation_results_dir() / "corpus_sampler"
    out_path = results_dir / f"corpus_sampler__{args.config}__{args.model.replace('/', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "model":     args.model,
                "config":    args.config,
                "seed":      args.seed,
                "fractions": fractions,
                "rows":      rows,
            },
            f,
            indent=2,
        )
    logger.info("Results saved → %s", out_path)


if __name__ == "__main__":
    main()
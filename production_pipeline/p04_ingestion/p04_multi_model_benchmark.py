"""
p04_multi_model_benchmark.py
============================
Run the retrieval benchmark across all (or selected) embedding models.

Run:
    uv run python -m production_pipeline.p04_ingestion.p04_multi_model_benchmark
    uv run python -m production_pipeline.p04_ingestion.p04_multi_model_benchmark --models BAAI/bge-base-en-v1.5
"""

from __future__ import annotations

import gc
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from sentence_transformers import SentenceTransformer

from rag_pipeline.logging import get_logger
from ._benchmark_config import BenchmarkConfig
from ._benchmark_metrics.aggregation import aggregate_metrics
from ._benchmark_metrics.evaluation import evaluate_config
from ._benchmark_report import (
    print_full_benchmark_report,
    save_benchmark_results,
)
from configs.benchmark_cli import create_multi_benchmark_parser

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Resume state
# ---------------------------------------------------------------------------

class BenchmarkState:
    """
    Persist completed (model, config) pairs to disk so a failed run can
    resume from where it left off rather than starting over.
    """

    def __init__(self, output_dir: Path) -> None:
        self._state_file = output_dir / "benchmark_state.json"
        self._completed: set[str] = set()
        self._load()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _key(self, model: str, config: str) -> str:
        return f"{model}|{config}"

    def _load(self) -> None:
        if not self._state_file.exists():
            return
        try:
            with self._state_file.open(encoding="utf-8") as f:
                data = json.load(f)
            self._completed = set(data.get("completed", []))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not load benchmark state: {e} — starting fresh")
            self._completed = set()

    def _save(self) -> None:
        try:
            with self._state_file.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "completed": sorted(self._completed),
                        "updated_at": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except OSError as e:
            logger.warning(f"Could not save benchmark state: {e}")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def is_completed(self, model: str, config: str) -> bool:
        return self._key(model, config) in self._completed

    def mark_completed(self, model: str, config: str) -> None:
        self._completed.add(self._key(model, config))
        self._save()

    def reset(self) -> None:
        self._completed.clear()
        if self._state_file.exists():
            self._state_file.unlink()
        logger.info("Benchmark state reset.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = create_multi_benchmark_parser()
    args = parser.parse_args()

    # Start from defaults.json, then overlay whatever the user passed on the CLI.
    config = BenchmarkConfig.from_defaults().merge_args(args)

    if config.output_dir is None:
        logger.error("output_dir is not set — add 'experiments_dir' to defaults.json or pass --output-dir")
        sys.exit(1)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Connections — fail loudly at startup, not mid-benchmark
    # ------------------------------------------------------------------
    try:
        qdrant = config.make_qdrant_client()
    except Exception as e:
        logger.error(f"Cannot connect to Qdrant: {e}")
        sys.exit(1)

    es = config.make_es_client()  # None if ES not configured / unreachable

    # ------------------------------------------------------------------
    # Resume state
    # ------------------------------------------------------------------
    state = BenchmarkState(config.output_dir)
    if config.reset:
        state.reset()

    # ------------------------------------------------------------------
    # Load shared inputs once
    # ------------------------------------------------------------------
    try:
        test_set = config.get_test_set()
        configs = config.get_configs()
        model_entries = config.get_model_entries()
    except (FileNotFoundError, ValueError, KeyError) as e:
        logger.error(f"Failed to load benchmark inputs: {e}")
        sys.exit(1)

    logger.info(
        f"Benchmark: {len(model_entries)} model(s), "
        f"{len(configs)} config(s), "
        f"{len(test_set)} queries"
    )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    all_summaries = []
    failed: list[str] = []

    for entry in model_entries:
        model_name = entry["name"]
        collection = entry["collection"]

        if not qdrant.collection_exists(collection):
            logger.warning(f"Collection '{collection}' not found — skipping '{model_name}'")
            failed.append(model_name)
            if config.fail_fast:
                break
            continue

        logger.info("=" * 60)
        logger.info(f"Model: {model_name}")

        topic_map = config.get_topic_map(model_name)

        try:
            trust = entry.get("trust_remote_code", False)
            model = SentenceTransformer(model_name, trust_remote_code=trust)
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            failed.append(model_name)
            if config.fail_fast:
                break
            continue

        model_failed = False

        for cfg_name, cfg in configs.items():
            if config.resume and state.is_completed(model_name, cfg_name):
                logger.info(f"  Skipping {cfg_name} (already completed)")
                continue

            logger.info(f"  Config: {cfg_name}")

            try:
                results = evaluate_config(
                    client=qdrant,
                    collection=collection,
                    model=model,
                    test_set=test_set,
                    topic_map=topic_map,
                    config=cfg,
                    top_k=config.top_k,
                    es=es,
                    es_index=config.es_index,
                    encode_batch_size=config.encode_batch_size,
                )
            except Exception as e:
                logger.error(f"  evaluate_config failed for {model_name}/{cfg_name}: {e}")
                traceback.print_exc()
                failed.append(f"{model_name}/{cfg_name}")
                model_failed = True
                if config.fail_fast:
                    break
                continue

            summary = aggregate_metrics(results, cfg_name, model_name)
            all_summaries.append(summary)
            state.mark_completed(model_name, cfg_name)
            logger.info(f"    Hit@5={summary.hit_rate_5:.1%}  MRR={summary.mrr:.4f}")

        del model
        gc.collect()

        if model_failed and config.fail_fast:
            break

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    if not all_summaries:
        logger.error("No results collected — nothing to save.")
        sys.exit(1)

    if not config.no_detail:
        print_full_benchmark_report(all_summaries)

    # save_benchmark_results also calls save_performance_summary internally
    save_benchmark_results(all_summaries, config.output_dir)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("BENCHMARK COMPLETE")
    if failed:
        logger.warning(f"Failed ({len(failed)}): {failed}")
    else:
        logger.info("All models completed successfully.")


if __name__ == "__main__":
    main()
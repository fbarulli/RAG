
"""
run_clean_pipeline.py
=====================
End-to-end clean run: clears old outputs, runs topic modeling for all models,
merges NER assignments, ingests into Qdrant + ES, compares models,
runs retrieval benchmarking, and runs LLM-as-judge evaluation.

Run: uv run python -m production_pipeline.run_clean_pipeline
"""
import subprocess
import sys
import json
import traceback
from pathlib import Path

from production_pipeline.p02_eda._topic_merge import merge as merge_topic_assignments
from rag_pipeline.logging import get_logger

logger = get_logger(__name__)

PROJECT = Path(__file__).parent.parent
TOPIC_DIR = PROJECT / "production_pipeline/p02_eda/experiments"
BENCH_DIR = PROJECT / "production_pipeline/experiments"
TEST_FILE = PROJECT / "production_pipeline/p01_data_cleaning/data/processed/test.jsonl"

MODELS = [
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-small-v2",
    "nomic-ai/nomic-embed-text-v1.5",
    "sentence-transformers/all-mpnet-base-v2",
]

JUDGE_MODEL = "BAAI/bge-base-en-v1.5"
JUDGE_CONFIG = "entity_boosted"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], step: str = "") -> bool:
    label = f"[{step}] " if step else ""
    logger.info(f"{label}Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT)
    if result.returncode != 0:
        logger.warning(f"{label}Command exited with code {result.returncode}")
        return False
    return True


def clean() -> None:
    logger.info("Step 0: Clearing previous experiment outputs")
    for d in [TOPIC_DIR, BENCH_DIR]:
        if d.exists():
            for f in d.glob("*.json"):
                if "tfidf" not in f.name:
                    f.unlink(missing_ok=True)
                    logger.debug(f"Deleted: {f.name}")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def run_topic_modeling() -> None:
    logger.info("Step 1/6: Topic Modeling (all models)")
    run([sys.executable, "-m", "production_pipeline.p02_eda.p02_topic_modeling",
         "--run-all", "--min-topic-size", "5", "--min-samples", "1"],
        step="topic_modeling")


def run_merge() -> None:
    logger.info("Step 2/6: Merging topic assignments + NER reclassification")
    try:
        merge_topic_assignments()
    except Exception:
        logger.error(f"Merge failed:\n{traceback.format_exc()}")


def run_ingestion() -> None:
    logger.info("Step 3/6: Ingesting into Qdrant + ES")
    for model in MODELS:
        logger.info(f"  Qdrant: {model}")
        run([sys.executable, "-m", "production_pipeline.p04_ingestion.p00_ingest_qdrant",
             "--model", model], step="ingest_qdrant")

    logger.info("  ES: BAAI/bge-base-en-v1.5 (model-agnostic BM25)")
    run([sys.executable, "-m", "production_pipeline.p04_ingestion.p00_ingest_es",
         "--model", "BAAI/bge-base-en-v1.5"], step="ingest_es")

    logger.info("  Building payload indexes for entity boosting")
    _build_payload_indexes()


def _build_payload_indexes() -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PayloadSchemaType
    client = QdrantClient(host="localhost", port=6333)
    fields = ["ner_category", "ner_primary_entity", "topic", "section"]
    for model in MODELS:
        short = model.split("/")[-1].replace("-", "_").replace(".", "_")
        collection = f"faqs_{short}"
        if not client.collection_exists(collection):
            logger.warning(f"Collection {collection} not found — skipping index creation")
            continue
        for field in fields:
            try:
                client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass  # Index may already exist
        logger.info(f"  Payload indexes ready: {collection}")


def run_comparison() -> None:
    logger.info("Step 4/6: Model Comparison")
    run([sys.executable, "-m", "production_pipeline.p02_eda.p06_model_comparison",
         "--input-dir", str(TOPIC_DIR)], step="comparison")


def run_benchmark() -> None:
    logger.info("Step 5/6: Retrieval Benchmark (all models)")
    for model in MODELS:
        slug = model.replace("/", "_").replace("-", "_")
        topic_file = TOPIC_DIR / f"topic_assignments_{slug}.json"
        if not topic_file.exists():
            logger.warning(f"Skipping {model}: {topic_file.name} not found")
            continue
        logger.info(f"  Benchmarking: {model}")
        run([sys.executable, "-m", "production_pipeline.p04_ingestion.p03_benchmark",
             "--model", model,
             "--test-set", str(TEST_FILE)], step="benchmark")


def run_judge() -> None:
    logger.info(f"Step 6/6: LLM-as-judge ({JUDGE_MODEL} / {JUDGE_CONFIG})")
    run([sys.executable, "-m", "production_pipeline.p05_evaluation.p05_llm_judge",
         "--model", JUDGE_MODEL,
         "--config", JUDGE_CONFIG], step="judge")


def print_summary() -> None:
    logger.info("Pipeline complete — benchmark summary:")
    print(f"\n{'='*90}")
    print(f"{'Model':<45} {'Config':<20} {'Hit@5':>6} {'MRR':>6} {'Latency':>8}")
    print(f"{'='*90}")

    results_path = BENCH_DIR / "benchmark_results.json"
    if not results_path.exists():
        print("  No benchmark results found.")
        return

    results = json.load(open(results_path))
    for r in sorted(results, key=lambda x: -x.get("hit_rate_5", 0)):
        if r.get("num_queries", 0) == 0:
            continue
        print(
            f"  {r.get('model_name','?'):<43} "
            f"  {r.get('config_name','?'):<18} "
            f"  {r.get('hit_rate_5', 0):>5.1%} "
            f"  {r.get('mrr', 0):>5.3f} "
            f"  {r.get('latency_p50', 0):>6.1f}ms"
        )

    print(f"{'='*90}")

    judge_path = BENCH_DIR / "judge_results.json"
    if judge_path.exists():
        judge = json.load(open(judge_path))
        faith = [r["faithfulness"] for r in judge if "faithfulness" in r]
        fc = [r["factual_correctness"] for r in judge if "factual_correctness" in r]
        non_chaos = [r for r in judge if r.get("query_type") != "chaos_monkey"]
        composite = sum(
            (r["faithfulness"] + r["factual_correctness"]) / 2
            for r in non_chaos
        ) / len(non_chaos) if non_chaos else 0
        print(f"\n  Judge ({JUDGE_MODEL} / {JUDGE_CONFIG})")
        print(f"    Queries scored    : {len(judge)}")
        print(f"    Mean faithfulness : {sum(faith)/len(faith):.3f}")
        print(f"    Mean factual corr.: {sum(fc)/len(fc):.3f}")
        print(f"    Composite (non-chaos): {composite:.3f}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    clean()
    run_topic_modeling()
    run_merge()
    run_ingestion()
    run_comparison()
    run_benchmark()
    run_judge()
    print_summary()

"""
run_clean_pipeline.py
=====================
End-to-end clean run: clears old outputs, runs topic modeling for all models,
compares them, and runs retrieval benchmarking.

Run: uv run python -m production_pipeline.run_clean_pipeline
"""
import subprocess
import sys
import json
from pathlib import Path
from production_pipeline.p02_eda._topic_merge import merge as merge_topic_assignments


PROJECT = Path(__file__).parent.parent
TOPIC_DIR = PROJECT / "production_pipeline/p02_eda/experiments"
BENCH_DIR = PROJECT / "production_pipeline/experiments"
TEST_FILE = PROJECT / "production_pipeline/p01_data_cleaning/data/processed/test.jsonl"

MODELS = [
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-small-v2",
    "nomic-ai/nomic-embed-text-v1.5",
    "sentence-transformers/all-mpnet-base-v2"
]
def run_merge() -> None:
    print("🔗 Step 2/5: Merging topic assignments + NER reclassification")
    merge_topic_assignments()


def run(cmd: list[str]) -> bool:
    print(f"\n🚀 {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT)
    if result.returncode != 0:
        print(f"⚠️ Command exited with code {result.returncode}")
        return False
    return True


def clean() -> None:
    print("🧹 Clearing previous experiment outputs...")
    for d in [TOPIC_DIR, BENCH_DIR]:
        if d.exists():
            for f in d.glob("*.json"):
                if "tfidf" not in f.name:
                    f.unlink(missing_ok=True)


def run_topic_modeling() -> None:
    print("📦 Step 1/5: Topic Modeling (all models)")
    run([sys.executable, "-m", "production_pipeline.p02_eda.p02_topic_modeling",
         "--run-all", "--min-topic-size", "5", "--min-samples", "1"])


def run_ingestion() -> None:
    print("📥 Step 3/5: Ingesting into Qdrant + ES (all models)")
    for model in MODELS:
        print(f"   ↳ Qdrant: {model}")
        run([sys.executable, "-m", "production_pipeline.p04_ingestion.p00_ingest_qdrant",
             "--model", model])
    print("   ↳ ES ingestion (bge-base only — model-agnostic BM25)")
    run([sys.executable, "-m", "production_pipeline.p04_ingestion.p00_ingest_es",
         "--model", "BAAI/bge-base-en-v1.5"])


def run_comparison() -> None:
    print("📊 Step 4/5: Model Comparison")
    run([sys.executable, "-m", "production_pipeline.p02_eda.p06_model_comparison",
         "--input-dir", str(TOPIC_DIR)])


def run_benchmark() -> None:
    print("🎯 Step 5/5: Retrieval Benchmark (all models)")
    for model in MODELS:
        slug = model.replace("/", "_").replace("-", "_")
        topic_file = TOPIC_DIR / f"topic_assignments_{slug}.json"

        if not topic_file.exists():
            print(f"⚠️ Skipping {model}: {topic_file.name} not found")
            continue

        print(f"   ↳ Benchmarking {model}")
        run([sys.executable, "-m", "production_pipeline.p04_ingestion.p03_benchmark",
             "--model", model,
             "--test-set", str(TEST_FILE)])


def print_summary() -> None:
    print("\n📈 BENCHMARK SUMMARY")
    print("-" * 90)
    print(f"{'Model':<45} {'R@1':>6} {'R@5':>6} {'MRR':>6} {'Latency':>8}")
    print("-" * 90)
    for f in sorted(BENCH_DIR.glob("benchmark_*.json")):
        try:
            data = json.load(open(f))
            m = data.get("metrics", {})
            model = data.get("embedding_model", data.get("model", f.stem.replace("benchmark_", "")))
            lat = m.get("avg_latency_ms", m.get("latency_p50_ms", 0))
            print(f"{model:<45} {m.get('hit_at_1', 0):>5.1%} {m.get('hit_at_5', 0):>5.1%} {m.get('mrr', 0):>5.3f} {lat:>7.1f}ms")
        except Exception:
            pass
    print("-" * 90)


if __name__ == "__main__":
    clean()
    run_topic_modeling()
    run_merge()
    run_ingestion()
    run_comparison()
    run_benchmark()
    print_summary()
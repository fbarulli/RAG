# test_multi_model.py
"""Test multiple models."""

from production_pipeline.p04_ingestion._benchmark_loader import (
    load_defaults, load_model_registry, load_test_set, 
    load_topic_assignments, load_configs
)
from production_pipeline.p04_ingestion._benchmark_metrics.evaluation import evaluate_config
from production_pipeline.p04_ingestion._benchmark_metrics.aggregation import aggregate_metrics
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

def main():
    defaults = load_defaults()
    paths_cfg = defaults.get("paths", {})
    qdrant_cfg = defaults.get("qdrant", {})
    
    base = Path("/workspaces/LLM")
    test_set_path = base / "production_pipeline/p01_data_cleaning/data/processed/test.jsonl"
    clean_path = base / "production_pipeline/p01_data_cleaning/data/processed/clean.jsonl"
    topic_path = base / "production_pipeline/p02_eda/experiments/topic_assignments_all.json"
    configs_path = base / "configs/retrieval_configs.json"
    
    # Load test set
    test_set = load_test_set(test_set_path, clean_path)[:20]  # 20 queries
    
    # Load configs
    configs = load_configs(configs_path)
    config = configs["vector_default"]  # Use same config for all models
    
    # Connect to Qdrant
    client = QdrantClient(host=qdrant_cfg.get("host", "localhost"), port=qdrant_cfg.get("port", 6333))
    
    # Load models
    models = load_model_registry(enabled_only=True)
    
    print("\n" + "="*80)
    print(f"{'Model':<40} {'Recall@5':<10} {'MRR':<8} {'Latency p50':<12}")
    print("="*80)
    
    results = []
    
    for model_entry in models:
        model_name = model_entry["name"]
        collection = model_entry["collection"]
        
        if not client.collection_exists(collection):
            print(f"{model_name:<40} {'SKIPPED':<10} (collection not found)")
            continue
        
        print(f"Loading {model_name}...")
        model = SentenceTransformer(model_name, trust_remote_code=model_entry.get("trust_remote_code", False))
        topic_map = load_topic_assignments(topic_path, model=model_name)
        
        eval_results = evaluate_config(
            client=client, collection=collection, model=model,
            test_set=test_set, topic_map=topic_map, config=config,
            top_k=10, encode_batch_size=32,
        )
        
        summary = aggregate_metrics(eval_results, "vector_default", model_name)
        results.append((model_name, summary))
        
        print(f"{model_name:<40} {summary.hit_rate_5:<9.2%} {summary.mrr:<8.4f} {summary.latency_p50:<11.1f}ms")
        
        del model  # Free memory
    
    print("="*80)
    
    if results:
        best = max(results, key=lambda x: x[1].mrr)
        print(f"\n🏆 Winner: {best[0]} (MRR={best[1].mrr:.4f})")

if __name__ == "__main__":
    from pathlib import Path
    main()
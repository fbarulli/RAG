# test_pipeline_probe.py
"""
Standalone probe - run directly to isolate where the hang occurs.
Usage: python test_pipeline_probe.py
"""
import logging
import sys
import time
from typing import List, Dict, Any
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)

print("=== STEP 1: imports ===", flush=True)
try:
    from production_pipeline.p04_ingestion._benchmark_metrics.retrievers import run_entity_boosted_retrieval
    from production_pipeline.p04_ingestion._benchmark_reranker import evaluate_with_reranker
    from production_pipeline.p04_ingestion._reranker_runner import run_reranking
    print("=== imports OK ===", flush=True)
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

print("=== STEP 2: qdrant client ===", flush=True)
try:
    from qdrant_client import QdrantClient
    client = QdrantClient(host="localhost", port=6333)
    print(f"Collections: {client.get_collections()}", flush=True)
except Exception as e:
    print(f"Qdrant connection failed: {e}")
    sys.exit(1)

print("=== STEP 3: dummy vector ===", flush=True)
dummy_vector = [0.0] * 768  # adjust dim to match your model

print("=== STEP 4: entity_boosted retrieval ===", flush=True)
t0 = time.perf_counter()
try:
    result = run_entity_boosted_retrieval(
        client=client,
        collection="faqs_bge_base_en_v1_5",
        query_vector=dummy_vector,
        course_filter="machine-learning-zoomcamp",
        config={"boost_question": 5.0, "boost_text": 5.0, "rrf_k": 60},
        top_k=40,
        ner_category=None,
        ner_primary_entity=None,
    )
    print(f"=== retrieval done in {(time.perf_counter()-t0)*1000:.1f}ms ===", flush=True)
    print(f"hit_ids: {result.hit_ids[:5]}", flush=True)
except Exception as e:
    print(f"Retrieval failed: {e}")

print("=== STEP 5: reranker ===", flush=True)
try:
    candidates = [{"es_id": id_, "question": "test", "answer": "test"} for id_ in result.hit_ids[:10]]
    reranked, latency = run_reranking(
        reranker_config={"model": "BAAI/bge-reranker-base", "name": "bge-reranker-base"},
        query="test query",
        candidates=candidates,
        top_k=5,
    )
    print(f"=== reranking done in {latency:.1f}ms ===", flush=True)
    print(f"reranked_ids: {reranked}", flush=True)
except Exception as e:
    print(f"Reranking failed: {e}")

print("=== ALL STEPS OK ===", flush=True)

# Test ONNX Model Loading
print("=== STEP 6: ONNX Model Loading Test ===", flush=True)
try:
    from production_pipeline.p04_ingestion._onnx_model_loader import ONNXModelLoader
    from production_pipeline.p04_ingestion._onnx_cross_encoder import ONNXCrossEncoder
    
    print("=== STEP 7: Testing ONNX Model ===", flush=True)
    model_loader = None
    model_name = "BAAI/bge-reranker-base"
    provider = "CPUExecutionProvider"
    
    # Initialize model loader
    model_loader = ONNXModelLoader(model_name, provider)
    print("ONNX Model loaded successfully", flush=True)
    
    # Test dummy inference
    #from production_pipeline.p04_ingestion.onnx_cross_encoder import ONNXCrossEncoder
    encoder = ONNXCrossEncoder(model_name, max_length=512, provider=provider)
    
    # Dummy input pairs
    pairs = [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What is machine learning?", "Machine learning is a field of AI."),
    ]
    
    t0 = time.perf_counter()
    scores = encoder.predict(pairs, batch_size=32)
    print(f"ONNX Inference completed in {(time.perf_counter()-t0)*1000:.1f}ms", flush=True)
    print("ONNX test completed successfully", flush=True)
    
except Exception as e:
    print(f"ONNX test failed: {e}")

print("=== COMPLETE TEST DONE ===", flush=True)

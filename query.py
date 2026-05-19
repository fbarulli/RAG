# /workspaces/LLM/test_pipeline_probe.py
"""
Standalone probe — run directly to isolate where the hang occurs.
Usage: python test_pipeline_probe.py
"""
import logging
import sys
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)

print("=== STEP 1: imports ===", flush=True)
from production_pipeline.p04_ingestion._benchmark_metrics.retrievers import run_entity_boosted_retrieval
from production_pipeline.p04_ingestion._benchmark_reranker import evaluate_with_reranker
from production_pipeline.p04_ingestion._reranker_runner import run_reranking
print("=== imports OK ===", flush=True)

print("=== STEP 2: qdrant client ===", flush=True)
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
print(f"Collections: {client.get_collections()}", flush=True)

print("=== STEP 3: dummy vector ===", flush=True)
dummy_vector = [0.0] * 768  # adjust dim to match your model

print("=== STEP 4: entity_boosted retrieval ===", flush=True)
import time
t0 = time.perf_counter()
result = run_entity_boosted_retrieval(
    client=client,
    collection="faqs_bge_base_en_v1_5",

    query_vector=dummy_vector,
    course_filter="machine-learning-zoomcamp",  # or whichever course is in your test set

    config={"boost_question": 5.0, "boost_text": 5.0, "rrf_k": 60},
    top_k=40,
    ner_category=None,
    ner_primary_entity=None,
)
print(f"=== retrieval done in {(time.perf_counter()-t0)*1000:.1f}ms ===", flush=True)
print(f"hit_ids: {result.hit_ids[:5]}", flush=True)

print("=== STEP 5: reranker ===", flush=True)
candidates = [{"es_id": id_, "question": "test", "answer": "test"} for id_ in result.hit_ids[:10]]
reranked, latency = run_reranking(
    reranker_config={"model": "BAAI/bge-reranker-base", "name": "bge-reranker-base"},
    query="test query",
    candidates=candidates,
    top_k=5,
)
print(f"=== reranking done in {latency:.1f}ms ===", flush=True)
print(f"reranked_ids: {reranked}", flush=True)

print("=== ALL STEPS OK ===", flush=True)
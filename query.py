"""
scripts/test_fixed_retrieval.py
Test the fixed version of run_vector_retrieval WITHOUT changing any files yet.
"""

from qdrant_client import QdrantClient
from production_pipeline.p04_ingestion._benchmark_metrics.retrievers import run_vector_retrieval
import json
from rag_pipeline.paths import Paths
import time

# Load config
with open(Paths.base() / "configs" / "defaults.json") as f:
    defaults = json.load(f)

qdrant_cfg = defaults.get("qdrant", {})
client = QdrantClient(
    host=qdrant_cfg.get("host", "localhost"), 
    port=qdrant_cfg.get("port", 6333)
)

collection = "faqs_bge_base_en_v1_5"

print("Testing FIXED version...")

# Temporary fixed version for testing
def test_fixed_run_vector_retrieval():
    start = time.perf_counter()
    must_conditions = []
    # (we keep it minimal for this test)
    
    result = client.query_points(
        collection_name=collection,
        query=[0.1] * 768,   # dummy vector
        limit=3,
        with_payload=True,
        with_vectors=False,
    )
    
    points = result.points
    latency_ms = (time.perf_counter() - start) * 1000
    
    hit_ids = tuple(p.payload.get("es_id", "") for p in points)
    hit_courses = tuple(p.payload.get("course", "") for p in points)
    hit_scores = tuple(float(p.score) if p.score is not None else 0.0 for p in points)
    hit_answers = tuple(p.payload.get("answer", "") for p in points)      # ← key line
    top_answer = points[0].payload.get("answer") if points else None

    # Import here so we can test the class
    from production_pipeline.p04_ingestion._benchmark_types import SearchResult
    
    return SearchResult(
        hit_ids=hit_ids,
        hit_scores=hit_scores,
        hit_courses=hit_courses,
        top_answer=top_answer,
        latency_ms=latency_ms,
        hit_answers=hit_answers,
    )

# Run the test
result = test_fixed_run_vector_retrieval()

print("✅ Success!")
print("hit_ids count:", len(result.hit_ids))
print("hit_answers count:", len(result.hit_answers))
print("First hit_id:", result.hit_ids[0] if result.hit_ids else None)
print("First course:", result.hit_courses[0] if result.hit_courses else None)
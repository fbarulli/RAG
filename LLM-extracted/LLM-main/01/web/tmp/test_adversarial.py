"""Test adversarial queries — adjacent to course domain but not in FAQ."""
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
model = SentenceTransformer('BAAI/bge-base-en-v1.5')
client = QdrantClient('localhost', port=6333)
ADVERSARIAL = [
    # ML-adjacent but not covered
    "how do I deploy a model on AWS SageMaker",
    "what is gradient descent",
    "how do I use MLflow with Azure instead of AWS",
    "how do I implement a neural network in PyTorch",
    "what is the difference between bagging and boosting",
    # DE-adjacent
    "how do I use Apache Airflow instead of Kestra",
    "how do I connect to Snowflake instead of BigQuery",
    "what is the difference between OLAP and OLTP",
    # MLOps-adjacent
    "how do I use Weights & Biases instead of MLflow",
    "how do I deploy a model with BentoML",
    # LLM-adjacent
    "how do I fine-tune Llama 3 on my own data",
    "what is the difference between RAG and fine-tuning",
    # General tech
    "how do I use Kubernetes for model serving",
    "what is a vector database",
]
print("Scoring adversarial queries...\n")
vecs = model.encode(ADVERSARIAL, batch_size=32, show_progress_bar=True)
print(f"{'Score':<8} {'Query':<60} {'Top match':<60}")
print(f"{'-'*8} {'-'*60} {'-'*60}")
for q, vec in zip(ADVERSARIAL, vecs):
    hits = client.query_points(collection_name='faqs_bge_base_en_v1.5', query=vec.tolist(), limit=1, with_payload=True)
    if hits.points:
        top = hits.points[0]
        score = top.score
        match = top.payload.get('question', '')
        # Determine if this SHOULD be blocked (no FAQ covers it) or let through
        action = "BLOCK" if score < 0.65 else "PASS"
        print(f"{score:<8.3f} {q:<60} {match[:60]}")
    else:
        print(f"{0:<8.3f} {q:<60} {'NO RESULTS':<60}")
# Summary
scores = []
for vec in vecs:
    hits = client.query_points(collection_name='faqs_bge_base_en_v1.5', query=vec.tolist(), limit=1, with_payload=True)
    scores.append(hits.points[0].score if hits.points else 0)
scores = np.array(scores)
blocked = scores < 0.65
passed = ~blocked
print(f"\nThreshold 0.65:")
print(f"  Blocked: {blocked.sum()}/{len(scores)}")
print(f"  Passed:  {passed.sum()}/{len(scores)}")
print(f"  Score range: {scores.min():.3f} - {scores.max():.3f}")

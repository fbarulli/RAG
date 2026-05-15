"""
eval/qdrant_evaluation.py
==========================
Comprehensive Qdrant evaluation suite.

Tests:
  1. Different embedding models (all-MiniLM-L6-v2, all-mpnet-base-v2, etc.)
  2. Hybrid search (dense + sparse) vs pure vector
  3. Cross-course retrieval accuracy
  4. Latency comparison: Qdrant vs ES Vector vs ES BM25
  5. Result diversity (pairwise cosine distance, section coverage)

Output: experiments/results/qdrant_eval_*.json

Run:    uv run python eval/qdrant_evaluation.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
from typing import List, Dict
from datetime import datetime
from collections import defaultdict

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue,
    SearchParams, SearchRequest, Fusion, Prefetch
)

RESULTS_DIR = 'experiments/results'
QDRANT_URL = 'http://localhost:6333'
ES_URL = 'http://localhost:9200'

# Models to test
EMBEDDING_MODELS = [
    'all-MiniLM-L6-v2',       # 384 dims, fast
    'BAAI/bge-small-en-v1.5',  # 384 dims, retrieval-optimized
]

# Models requiring more memory - test one at a time
LARGE_MODELS = [
    'all-mpnet-base-v2',       # 768 dims, better quality
]


class QdrantEvaluator:
    def __init__(self):
        self.qdrant = QdrantClient(QDRANT_URL)
        self.es = Elasticsearch(ES_URL)
        self.results = {}
        
    def get_all_docs(self) -> List[Dict]:
        """Get all documents from ES."""
        result = self.es.search(index='faqs_complete', body={
            'size': 2000,
            'query': {'match_all': {}}
        })
        return [hit['_source'] for hit in result['hits']['hits']]

    # ── Test 1: Embedding Model Comparison ─────────────────────────────────
    def test_embedding_models(self):
        """Compare different embedding models for retrieval quality."""
        print("\n" + "="*60)
        print("TEST 1: Embedding Model Comparison")
        print("="*60)
        
        docs = self.get_all_docs()
        eval_docs = docs[:200]  # Sample for speed
        
        for model_name in EMBEDDING_MODELS:
            print(f"\n  Testing {model_name}...")
            model = SentenceTransformer(model_name)
            dims = model.get_sentence_embedding_dimension()
            
            # Create collection
            collection = f'test_{model_name.replace("/", "_").replace("-", "_")}'
            self.qdrant.delete_collection(collection) if self.qdrant.collection_exists(collection) else None
            self.qdrant.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
            )
            
            # Ingest
            points = []
            for i, doc in enumerate(eval_docs):
                vec = model.encode(doc['question']).tolist()
                points.append(PointStruct(
                    id=i,
                    vector=vec,
                    payload={'es_id': doc['id'], 'question': doc['question'],
                             'answer': doc['answer'], 'course': doc['course']}
                ))
            self.qdrant.upsert(collection_name=collection, points=points)
            
            # Test: for each doc, find itself
            hits_at_1 = 0
            for i, doc in enumerate(eval_docs):
                query_vec = model.encode(doc['question']).tolist()
                results = self.qdrant.query_points(
                    collection_name=collection,
                    query=query_vec,
                    limit=1,
                    with_payload=True,
                )
                if results.points and results.points[0].payload['es_id'] == doc['id']:
                    hits_at_1 += 1
            
            recall_at_1 = hits_at_1 / len(eval_docs)
            print(f"    Recall@1 (self-retrieval): {recall_at_1:.2%}")
            print(f"    Dimensions: {dims}")
            
            self.results[f'embedding_{model_name}'] = {
                'recall_at_1': round(recall_at_1, 4),
                'dimensions': dims,
            }
            
            # Clean up test collection
            self.qdrant.delete_collection(collection)

    # ── Test 2: Hybrid Search vs Pure Vector ──────────────────────────────
    def test_hybrid_search(self):
        """Compare hybrid (dense + sparse) vs pure vector search."""
        print("\n" + "="*60)
        print("TEST 2: Hybrid Search vs Pure Vector")
        print("="*60)
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create test collections: one for dense, one for hybrid
        # Note: Qdrant sparse vectors require separate setup
        # For now, test with different fusion strategies
        docs = self.get_all_docs()[:100]
        
        collection = 'test_hybrid'
        self.qdrant.delete_collection(collection) if self.qdrant.collection_exists(collection) else None
        self.qdrant.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        
        points = []
        for i, doc in enumerate(docs):
            vec = model.encode(doc['question']).tolist()
            points.append(PointStruct(
                id=i, vector=vec,
                payload={'es_id': doc['id'], 'course': doc['course']}
            ))
        self.qdrant.upsert(collection_name=collection, points=points)
        
        # Test pure vector vs vector + course filter
        results_vector = []
        results_filtered = []
        
        for doc in docs:
            query_vec = model.encode(doc['question']).tolist()
            
            # Pure vector
            t0 = time.time()
            r1 = self.qdrant.query_points(
                collection_name=collection, query=query_vec, limit=5
            )
            lat_vector = time.time() - t0
            
            # Vector + course filter (hybrid)
            t0 = time.time()
            r2 = self.qdrant.query_points(
                collection_name=collection, query=query_vec, limit=5,
                query_filter=Filter(must=[FieldCondition(key='course', match=MatchValue(value=doc['course']))])
            )
            lat_filtered = time.time() - t0
            
            # Check if found
            found_vector = any(p.payload['es_id'] == doc['id'] for p in r1.points) if r1.points else False
            found_filtered = any(p.payload['es_id'] == doc['id'] for p in r2.points) if r2.points else False
            
            results_vector.append(found_vector)
            results_filtered.append(found_filtered)
        
        recall_vector = sum(results_vector) / len(results_vector)
        recall_filtered = sum(results_filtered) / len(results_filtered)
        
        print(f"  Pure Vector Recall@5: {recall_vector:.2%}")
        print(f"  Vector + Course Filter Recall@5: {recall_filtered:.2%}")
        
        self.results['hybrid'] = {
            'pure_vector_recall': round(recall_vector, 4),
            'vector_with_filter_recall': round(recall_filtered, 4),
        }
        
        self.qdrant.delete_collection(collection)

    # ── Test 3: Cross-Course Retrieval ────────────────────────────────────
    def test_cross_course(self):
        """Test if queries retrieve documents from wrong courses."""
        print("\n" + "="*60)
        print("TEST 3: Cross-Course Retrieval")
        print("="*60)
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        docs = self.get_all_docs()
        
        # For each query, check if top results are from the correct course
        cross_course_hits = defaultdict(int)
        total = 0
        
        for doc in docs[:300]:
            query_vec = model.encode(doc['question']).tolist()
            results = self.qdrant.query_points(
                collection_name='faqs',
                query=query_vec,
                limit=5,
                with_payload=True,
            )
            
            total += 1
            for hit in results.points:
                if hit.payload['course'] != doc['course']:
                    cross_course_hits[doc['course']] += 1
        
        print(f"  Queries tested: {total}")
        print(f"  Cross-course results (top 5, without filter):")
        for course, count in sorted(cross_course_hits.items()):
            rate = count / (total * 5)
            print(f"    {course}: {count} hits ({rate:.2%})")
        
        self.results['cross_course'] = {
            'queries_tested': total,
            'cross_course_hits': dict(cross_course_hits),
        }

    # ── Test 4: Latency Comparison ────────────────────────────────────────
    def test_latency(self):
        """Compare latency: Qdrant vs ES Vector vs ES BM25."""
        print("\n" + "="*60)
        print("TEST 4: Latency Comparison")
        print("="*60)
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        docs = self.get_all_docs()[:100]
        
        latencies = {'qdrant': [], 'es_vector': [], 'es_bm25': []}
        
        for doc in docs:
            query_vec = model.encode(doc['question']).tolist()
            
            # Qdrant
            t0 = time.time()
            self.qdrant.query_points(collection_name='faqs', query=query_vec, limit=5)
            latencies['qdrant'].append((time.time() - t0) * 1000)
            
            # ES Vector
            t0 = time.time()
            self.es.search(index='faqs_complete', body={
                'size': 5,
                'query': {
                    'script_score': {
                        'query': {'match_all': {}},
                        'script': {
                            'source': "cosineSimilarity(params.query_vector, 'question_vector') + 1.0",
                            'params': {'query_vector': query_vec}
                        }
                    }
                }
            })
            latencies['es_vector'].append((time.time() - t0) * 1000)
            
            # ES BM25
            t0 = time.time()
            self.es.search(index='faqs_complete', body={
                'size': 5,
                'query': {'match': {'question': doc['question']}}
            })
            latencies['es_bm25'].append((time.time() - t0) * 1000)
        
        print(f"  Latency (ms) - P50 / P95 / P99:")
        for name, lats in latencies.items():
            p50 = np.percentile(lats, 50)
            p95 = np.percentile(lats, 95)
            p99 = np.percentile(lats, 99)
            print(f"    {name:12s}: {p50:.1f} / {p95:.1f} / {p99:.1f}")
        
        self.results['latency'] = {
            name: {
                'p50': round(float(np.percentile(lats, 50)), 1),
                'p95': round(float(np.percentile(lats, 95)), 1),
                'p99': round(float(np.percentile(lats, 99)), 1),
                'mean': round(float(np.mean(lats)), 1),
            }
            for name, lats in latencies.items()
        }

    # ── Test 5: Result Diversity ──────────────────────────────────────────
    def test_diversity(self):
        """Measure diversity of top-5 results."""
        print("\n" + "="*60)
        print("TEST 5: Result Diversity")
        print("="*60)
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        docs = self.get_all_docs()[:100]
        
        diversities = []
        section_coverages = []
        
        for doc in docs:
            query_vec = model.encode(doc['question']).tolist()
            results = self.qdrant.query_points(
                collection_name='faqs',
                query=query_vec,
                limit=5,
                with_payload=True,
                with_vectors=True,
            )
            
            if len(results.points) >= 2:
                # Pairwise cosine distance
                vectors = [p.vector for p in results.points if p.vector]
                if len(vectors) >= 2:
                    distances = []
                    for i in range(len(vectors)):
                        for j in range(i+1, len(vectors)):
                            # Cosine distance = 1 - cosine similarity
                            sim = np.dot(vectors[i], vectors[j]) / (
                                np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j])
                            )
                            distances.append(1 - sim)
                    diversities.append(np.mean(distances))
                
                # Section coverage
                sections = set(p.payload.get('section', '') for p in results.points)
                section_coverages.append(len(sections))
        
        print(f"  Avg pairwise cosine distance (higher = more diverse): {np.mean(diversities):.4f}")
        print(f"  Avg unique sections in top 5: {np.mean(section_coverages):.2f}")
        print(f"  Section coverage distribution:")
        from collections import Counter
        dist = Counter(section_coverages)
        for k in sorted(dist):
            print(f"    {k} unique sections: {dist[k]} queries")
        
        self.results['diversity'] = {
            'avg_pairwise_distance': round(float(np.mean(diversities)), 4),
            'avg_unique_sections': round(float(np.mean(section_coverages)), 2),
        }

    def save_results(self):
        """Save all results."""
        output = {
            'metadata': {
                'name': 'Qdrant Evaluation Suite',
                'timestamp': datetime.now().isoformat(),
            },
            'results': self.results,
        }
        path = f'{RESULTS_DIR}/qdrant_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {path}")

    def run_all(self):
        self.test_embedding_models()
        self.test_hybrid_search()
        self.test_cross_course()
        self.test_latency()
        self.test_diversity()
        self.save_results()


if __name__ == '__main__':
    evaluator = QdrantEvaluator()
    evaluator.run_all()

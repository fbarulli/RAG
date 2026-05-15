"""
eval/benchmarks/smart_routing.py
=================================
Compares: open search, course filter, smart routing.
Uses BM25 + Qdrant (bge-base 768d) with RRF fusion.

Run:    uv run python eval/benchmarks/smart_routing.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json, time, numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

MODEL_NAME = 'BAAI/bge-base-en-v1.5'
ES_INDEX = 'faqs_complete'
QDRANT_COLLECTION = 'faqs_bge_base_en_v1.5'
TOP_K = 5
RRF_K = 60

COURSE_SIGNALS = {
    'de-zoomcamp':    ['bigquery', 'kafka', 'dbt', 'terraform', 'kestra', 'gcp',
                       'postgres', 'pgadmin', 'spark', 'parquet', 'data warehouse',
                       'dataproc', 'wget', 'leaderboard', 'dlt', 'mage'],
    'mlops-zoomcamp': ['mlflow', 'grafana', 'prefect', 'evidently', 'experiment tracking',
                       'model registry', 'model deployment', 'ml monitoring',
                       'production model', 'anaconda environment', 'vs code',
                       'bashrc', 'pytest', 'mage version', 'conda environment'],
    'ml-zoomcamp':    ['regression model', 'classification model', 'xgboost', 'homework', 
                       'scikit-learn', 'decision tree', 'random forest', 'gradient boosting',
                       'cross-validation', 'linear regression', 'logistic regression',
                       'svizor', '3.10.12-slim', 'xgb'],
    'llm-zoomcamp':   ['rag pipeline', 'ollama', 'openai api', 'prompt engineering', 
                       'embedding model', 'vector store', 'llm agent', 'langchain',
                       'tokenizer', 'chat completion', 'semantic search', 'ground truth set'],
}


def detect_course(query: str) -> str | None:
    query_lower = query.lower()
    matches = {course: sum(1 for kw in signals if kw in query_lower)
               for course, signals in COURSE_SIGNALS.items()}
    sorted_courses = sorted(matches.items(), key=lambda x: x[1], reverse=True)
    best_course, best_score = sorted_courses[0]
    second_score = sorted_courses[1][1] if len(sorted_courses) > 1 else 0
    
    # Require score >= 2 OR clear margin (at least 1 and no competition)
    if best_score >= 2 or (best_score >= 1 and second_score == 0):
        return best_course
    return None


def load_queries():
    with open('experiments/eval_queries.json') as f:
        data = json.load(f)
    queries = []
    for doc in data['queries']:
        for strategy, variations in doc['prompt_results'].items():
            for query in variations:
                queries.append({
                    'query': query, 'expected_id': doc['expected_id'],
                    'strategy': strategy, 'course': doc['course'],
                })
    return queries


def rrf_score(ranks, k=RRF_K):
    return sum(1.0 / (k + r) for r in ranks)


def hybrid_rrf(bm25_ids, qdrant_ids):
    combined = []
    for b_ids, q_ids in zip(bm25_ids, qdrant_ids):
        candidates = set(b_ids) | set(q_ids)
        rank_b = {doc_id: r+1 for r, doc_id in enumerate(b_ids)}
        rank_q = {doc_id: r+1 for r, doc_id in enumerate(q_ids)}
        scores = {}
        for doc_id in candidates:
            ranks = []
            if doc_id in rank_b: ranks.append(rank_b[doc_id])
            if doc_id in rank_q: ranks.append(rank_q[doc_id])
            scores[doc_id] = rrf_score(ranks)
        combined.append(sorted(scores, key=scores.__getitem__, reverse=True)[:TOP_K])
    return combined


def search_bm25(es, query, course=None):
    q = {'multi_match': {'query': query, 'fields': ['question^5', 'answer^5'], 'type': 'best_fields'}}
    if course:
        q = {'bool': {'must': q, 'filter': {'term': {'course': course}}}}
    return [h['_id'] for h in es.search(index=ES_INDEX, size=TOP_K, query=q)['hits']['hits']]


def search_qdrant(client, vec, course=None):
    qfilter = Filter(must=[FieldCondition(key='course', match=MatchValue(value=course))]) if course else None
    results = client.query_points(collection_name=QDRANT_COLLECTION, query=vec, limit=TOP_K, query_filter=qfilter, with_payload=True)
    return [h.payload.get('es_id', '') for h in results.points]


# ── Init ─────────────────────────────────────────────────────────────────────
es = Elasticsearch('http://localhost:9200')
client = QdrantClient('localhost', port=6333)
model = SentenceTransformer(MODEL_NAME)
test_queries = load_queries()
total = len(test_queries)

print("Encoding queries...")
query_texts = [tq['query'] for tq in test_queries]
query_vecs = model.encode(query_texts, batch_size=64, show_progress_bar=False)
courses = [tq['course'] for tq in test_queries]
expected_ids = [tq['expected_id'] for tq in test_queries]
strategies = [tq['strategy'] for tq in test_queries]

detected_courses = [detect_course(q) for q in query_texts]
signal_count = sum(1 for d in detected_courses if d)
correct = sum(1 for d, c in zip(detected_courses, courses) if d == c)
print(f"Signal coverage: {signal_count}/{total} ({signal_count/total:.1%})")
print(f"Signal accuracy: {correct}/{signal_count} ({correct/signal_count:.1%})\n")

# ── Run benchmarks ──────────────────────────────────────────────────────────
modes = {
    'Open search':   lambda i: None,
    'Course filter': lambda i: courses[i],
    'Smart routing': lambda i: detected_courses[i],
}

all_results = {}
all_lats = {}
all_per_query = {}

for name, course_fn in modes.items():
    bm25_ids = []
    qdrant_ids = []
    per_query_found = []
    lats = []
    
    for i in range(total):
        course = course_fn(i)
        t0 = time.time()
        b_ids = search_bm25(es, query_texts[i], course)
        q_ids = search_qdrant(client, query_vecs[i].tolist(), course)
        hybrid_ids = hybrid_rrf([b_ids], [q_ids])[0]
        lats.append((time.time() - t0) * 1000)
        
        found = expected_ids[i] in hybrid_ids
        per_query_found.append(found)
        bm25_ids.append(b_ids)
        qdrant_ids.append(q_ids)
    
    all_results[name] = sum(per_query_found)
    all_lats[name] = lats
    all_per_query[name] = per_query_found
    
    print(f"  {name:<20}: R@5={sum(per_query_found)/total:.1%} P50={np.percentile(lats, 50):.1f}ms")

# ── Comparison table ─────────────────────────────────────────────────────────
baseline = all_results['Open search'] / total
print(f"\n{'Strategy':<20} {'R@5':>8} {'vs Open':>8} {'P50ms':>8}")
print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8}")
for name in modes:
    r5 = all_results[name] / total
    gain = r5 - baseline
    print(f"{name:<20} {r5:>7.1%} {gain:>+7.1%} {np.percentile(all_lats[name], 50):>7.1f}")

# ── Per-strategy breakdown for all modes ─────────────────────────────────────
for name in ['Smart routing', 'Course filter', 'Open search']:
    print(f"\n{'='*50}")
    print(f"PER-STRATEGY ({name})")
    by_strat = defaultdict(lambda: {'found': 0, 'total': 0})
    for i, strat in enumerate(strategies):
        by_strat[strat]['total'] += 1
        by_strat[strat]['found'] += all_per_query[name][i]
    for s in sorted(by_strat):
        m = by_strat[s]
        print(f"  {s:<25}: {m['found']/m['total']:.1%}")

# ── Signal quality ──────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("SIGNAL QUALITY")
detected_map = defaultdict(lambda: {'correct': 0, 'wrong': 0, 'missed': 0})
for actual, detected in zip(courses, detected_courses):
    if detected is None:
        detected_map[actual]['missed'] += 1
    elif detected == actual:
        detected_map[actual]['correct'] += 1
    else:
        detected_map[actual]['wrong'] += 1

for course in sorted(detected_map):
    d = detected_map[course]
    total_course = d['correct'] + d['wrong'] + d['missed']
    print(f"  {course:<20}: {d['correct']} correct, {d['missed']} missed, {d['wrong']} wrong")

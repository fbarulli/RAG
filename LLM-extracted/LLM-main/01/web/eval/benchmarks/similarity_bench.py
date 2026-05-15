"""
eval/benchmarks/similarity_bench.py
====================================
Tests embedding model similarity across multiple hard queries.
Finds queries that failed across all configs and measures which
model best captures their semantic relationship to the correct FAQ.

Run:    uv run python eval/benchmarks/similarity_bench.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json, time, gc
import numpy as np
from sentence_transformers import SentenceTransformer

MODELS = [
    'BAAI/bge-small-en-v1.5',
    'intfloat/e5-small-v2',
    'BAAI/bge-base-en-v1.5',
    'intfloat/e5-base-v2',
]

# Load failure queries from variations results
import glob
files = sorted(glob.glob('experiments/results/variations_*.json'))
with open(files[-1]) as f:
    data = json.load(f)

# Find queries that failed in hybrid_70_30_vec (or the best config)
hybrid = next(c for c in data['configs'] if 'hybrid_70_30_vec' in c['name'])

# Get actual failed queries from eval_queries.json
with open('experiments/eval_queries.json') as f:
    qdata = json.load(f)

# Build query → expected_question mapping
query_map = {}
for doc in qdata['queries']:
    for strategy, variations in doc['prompt_results'].items():
        for v in variations:
            query_map[v] = doc['original_question']

# Collect failure queries from the failure sample
failure_queries = []
if hybrid.get('failures_sample'):
    for f in hybrid['failures_sample']:
        expected_q = query_map.get(f['query'], f['query'])
        failure_queries.append({
            'query': f['query'],
            'expected_question': expected_q,
            'strategy': f['strategy'],
        })

# Also add the jupyter query and a few known hard ones
hard_queries = [
    ("required packages for jupyter notebook", "Other packages needed but not listed"),
    ("what dependencies do i need for the starter notebook", "Other packages needed but not listed"),
    ("csv file has weird bytes in it", "UnicodeDecodeError: 'utf-8' codec can't decode byte"),
    ("my containerization software is broken", "Docker: ERRO[0000] error waiting for container"),
    ("assignment scores aren't adding up", "I may end up submitting the assignment late"),
]

# Combine: take up to 5 from failures + all hard queries (deduplicated)
test_pairs = []
seen = set()
for q, eq in hard_queries:
    key = q[:60]
    if key not in seen:
        seen.add(key)
        test_pairs.append((q, eq))

for f in failure_queries[:5]:
    key = f['query'][:60]
    if key not in seen:
        seen.add(key)
        test_pairs.append((f['query'], f['expected_question']))

print(f"Testing {len(test_pairs)} query pairs across {len(MODELS)} models\n")
print(f"{'Model':<35} {'AvgSim':>8} {'MinSim':>8} {'MaxSim':>8} {'Enc(ms)':>8}")
print(f"{'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for model_name in MODELS:
    short = model_name.split('/')[-1]
    
    model = SentenceTransformer(model_name)
    
    # Per-query encoding time
    t0 = time.time()
    model.encode(["test"], show_progress_bar=False)
    enc_ms = (time.time() - t0) * 1000
    
    # Encode all pairs
    similarities = []
    for query, expected in test_pairs:
        q_vec = model.encode(query)
        e_vec = model.encode(expected)
        sim = np.dot(q_vec, e_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(e_vec))
        similarities.append(sim)
    
    avg_sim = np.mean(similarities)
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)
    
    print(f"{short:<35} {avg_sim:>8.4f} {min_sim:>8.4f} {max_sim:>8.4f} {enc_ms:>7.1f}")
    
    del model
    gc.collect()

# Show per-query detail for the best model
print(f"\n{'='*70}")
print("PER-QUERY DETAIL (best model)")
print(f"{'='*70}")
best_model = SentenceTransformer(MODELS[0])  # Will be replaced below
best_name = MODELS[0]

for model_name in MODELS:
    model = SentenceTransformer(model_name)
    sims = []
    for query, expected in test_pairs:
        q_vec = model.encode(query)
        e_vec = model.encode(expected)
        sims.append(np.dot(q_vec, e_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(e_vec)))
    if np.mean(sims) > np.mean([np.dot(best_model.encode(q), best_model.encode(e)) / (np.linalg.norm(best_model.encode(q)) * np.linalg.norm(best_model.encode(e))) for q, e in test_pairs]):
        best_model = model
        best_name = model_name

print(f"\nBest: {best_name}\n")
for (query, expected), sim in zip(test_pairs, sims):
    print(f"  Sim: {sim:.4f}")
    print(f"    Query:    {query[:80]}")
    print(f"    Expected: {expected[:80]}")
    print()

del best_model
gc.collect()

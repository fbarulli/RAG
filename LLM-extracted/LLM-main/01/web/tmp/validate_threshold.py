"""
Validate LOW_CONFIDENCE_THRESHOLD against full eval set.
Uses course boosting (matching service.py logic) to find the optimal threshold.
Shows precision vs recall tradeoff at each threshold.
"""
import json, numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

model = SentenceTransformer('BAAI/bge-base-en-v1.5')
client = QdrantClient('localhost', port=6333)

with open('experiments/eval_queries.json') as f:
    qdata = json.load(f)

test_queries = []
for doc in qdata['queries']:
    for strategy, variations in doc['prompt_results'].items():
        for v in variations:
            test_queries.append({
                'query': v, 
                'expected_id': doc['expected_id'], 
                'course': doc['course'],
                'strategy': strategy,
            })

# Batch encode
print("Encoding...")
all_queries = [tq['query'] for tq in test_queries]
all_vecs = model.encode(all_queries, batch_size=32, show_progress_bar=True)

# Get top boosted scores for all queries (matching service.py logic)
print("Scoring...")
scores = []
found = []
for tq, vec in zip(test_queries, all_vecs):
    results = client.query_points(
        collection_name='faqs_bge_base_en_v1.5', 
        query=vec.tolist(), 
        limit=20, 
        with_payload=True,
    )
    if results.points:
        # Apply course boost to match service.py logic
        top = max(results.points, key=lambda h: h.score * (1.2 if h.payload.get('course') == tq['course'] else 1.0))
        boosted = top.score * (1.2 if top.payload.get('course') == tq['course'] else 1.0)
        scores.append(boosted)
        found.append(top.payload.get('es_id') == tq['expected_id'])
    else:
        scores.append(0)
        found.append(False)

scores = np.array(scores)
found = np.array(found)

# Per-strategy breakdown
strategies = [tq['strategy'] for tq in test_queries]
strat_names = sorted(set(strategies))

print(f"\nTotal queries: {len(scores)}")
print(f"Score range: {scores.min():.3f} - {scores.max():.3f}")
print(f"Mean score: {scores.mean():.3f}")
print(f"Overall precision (R@1): {found.mean():.1%}")

# Score distribution by found/missed
print(f"\nScore distribution:")
print(f"  Found (n={found.sum()}): mean={scores[found].mean():.3f}, p10={np.percentile(scores[found],10):.3f}, p25={np.percentile(scores[found],25):.3f}")
print(f"  Missed (n={(~found).sum()}): mean={scores[~found].mean():.3f}, p50={np.percentile(scores[~found],50):.3f}, p90={np.percentile(scores[~found],90):.3f}")

print(f"\n{'Threshold':<12} {'Blocked':<8} {'LetThrough':<10} {'CorrectLost':<12} {'Precision':<10} {'Recall':<10}")
print(f"{'-'*12} {'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")

for threshold in [0.5, 0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75, 0.80]:
    blocked = scores < threshold
    let_through = ~blocked
    blocked_count = blocked.sum()
    let_count = let_through.sum()
    correct_lost = (blocked & found).sum()
    precision = found[let_through].mean() if let_count > 0 else 0
    recall = found[let_through].sum() / found.sum() if found.sum() > 0 else 0
    
    marker = " ← current" if threshold == 0.62 else ""
    print(f"{threshold:<12.2f} {blocked_count:<8} {let_count:<10} {correct_lost:<12} {precision:<10.1%} {recall:<10.1%}{marker}")

# Per-strategy at current threshold
print(f"\nPer-strategy at 0.62:")
for s in strat_names:
    mask = np.array([st == s for st in strategies])
    s_scores = scores[mask]
    s_found = found[mask]
    blocked = s_scores < 0.62
    let_through = ~blocked
    precision = s_found[let_through].mean() if let_through.sum() > 0 else 0
    correct_lost = (blocked & s_found).sum()
    print(f"  {s:<25}: {blocked.sum()} blocked, {correct_lost} correct lost, precision={precision:.1%}")

# Show what gets blocked at 0.62
blocked_mask = scores < 0.62
blocked_queries = [test_queries[i] for i in range(len(test_queries)) if blocked_mask[i]]
blocked_found = found[blocked_mask]

print(f"\nBlocked at 0.62: {len(blocked_queries)} queries")
print(f"  Correct answers lost: {blocked_found.sum()}")
if blocked_found.sum() > 0:
    print(f"  Examples of lost correct answers:")
    lost = [blocked_queries[i] for i in range(len(blocked_queries)) if blocked_found[i]]
    for q in lost[:5]:
        print(f"    [{q['course']}] {q['query'][:80]}")
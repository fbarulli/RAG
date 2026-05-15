"""
Section Boosting Validation
============================
Tests whether section-weighted retrieval improves R@5 for homework/module queries.

What we're testing:
- Baseline: raw cosine similarity scores
- Boosted: scores × 1.15 for homework/leaderboard sections, × 1.10 for module sections
- Boost only applied when the query contains section-intent keywords (homework, module, etc.)

Why conditional boosting:
- "turn words into numbers" shouldn't boost homework sections — it's an embeddings question
- "homework q3 help" should boost homework sections — the student wants a specific assignment answer

Output: R@5 comparison, rank improvements, regressions, per-strategy breakdown.
Run:    uv run python /tmp/test_section_boost.py
"""
import json, numpy as np
from collections import Counter
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

SECTION_BOOST = {'homework': 1.15, 'leaderboard': 1.15, 'module': 1.10}
QUERY_SECTION_KW = {'homework', 'leaderboard', 'module', 'evaluation', 'grading', 'score'}

def boosted_score(hit, query_has_section_kw=False):
    """Only boost if query signals section intent."""
    if not query_has_section_kw:
        return hit.score
    section = hit.payload.get('section', '').lower()
    for kw, b in SECTION_BOOST.items():
        if kw in section:
            return hit.score * b
    return hit.score

def query_has_section_intent(query):
    q = query.lower()
    return any(kw in q for kw in QUERY_SECTION_KW)

# Batch encode
print("Encoding...")
all_queries = [tq['query'] for tq in test_queries]
all_vecs = model.encode(all_queries, batch_size=32, show_progress_bar=True)

# Build section lookup
print("Building lookups...")
section_lookup = {}
all_points, offset = [], None
while True:
    batch, offset = client.scroll(collection_name='faqs_bge_base_en_v1.5', limit=500, with_payload=True, offset=offset)
    all_points.extend(batch)
    if offset is None:
        break
for point in all_points:
    section_lookup[point.payload.get('es_id','')] = point.payload.get('section','')
print(f"  {len(section_lookup)} docs indexed")

# Track detailed metrics
baseline_found = 0
boost_found = 0
baseline_ranks = []
boost_ranks = []
regressions = []
improvements = []

section_intent_queries = 0
section_intent_baseline = 0
section_intent_boost = 0

# Per-strategy tracking
strat_baseline = Counter()
strat_boost = Counter()
strat_total = Counter()

for tq, vec in zip(test_queries, all_vecs):
    has_intent = query_has_section_intent(tq['query'])
    strat_total[tq['strategy']] += 1
    
    search_results = client.query_points(
        collection_name='faqs_bge_base_en_v1.5', 
        query=vec.tolist(), 
        limit=20, 
        with_payload=True
    )
    
    # Baseline: top 5 by raw score
    baseline_ids = [h.payload.get('es_id','') for h in search_results.points[:5]]
    baseline_hit = tq['expected_id'] in baseline_ids
    if baseline_hit:
        baseline_found += 1
        baseline_ranks.append(baseline_ids.index(tq['expected_id']) + 1)
        strat_baseline[tq['strategy']] += 1
    
    # Boosted: re-rank by section-boosted score (conditional on query intent)
    scored = [(h, boosted_score(h, has_intent)) for h in search_results.points]
    scored.sort(key=lambda x: x[1], reverse=True)
    boosted_ids = [h.payload.get('es_id','') for h, _ in scored[:5]]
    boost_hit = tq['expected_id'] in boosted_ids
    if boost_hit:
        boost_found += 1
        boost_ranks.append(boosted_ids.index(tq['expected_id']) + 1)
        strat_boost[tq['strategy']] += 1
    
    # Delta classification
    if not baseline_hit and boost_hit:
        improvements.append({
            'query': tq['query'], 'strategy': tq['strategy'],
            'from_rank': '>5', 'to_rank': boost_ranks[-1],
        })
    elif baseline_hit and not boost_hit:
        regressions.append({
            'query': tq['query'], 'strategy': tq['strategy'],
            'from_rank': baseline_ranks[-1], 'to_rank': '>5',
        })
    elif baseline_hit and boost_hit:
        old_rank = baseline_ids.index(tq['expected_id']) + 1
        new_rank = boosted_ids.index(tq['expected_id']) + 1
        if new_rank < old_rank:
            improvements.append({
                'query': tq['query'], 'strategy': tq['strategy'],
                'from_rank': old_rank, 'to_rank': new_rank,
            })
        elif new_rank > old_rank:
            regressions.append({
                'query': tq['query'], 'strategy': tq['strategy'],
                'from_rank': old_rank, 'to_rank': new_rank,
            })
    
    # Section-intent query stats
    if has_intent:
        section_intent_queries += 1
        if baseline_hit:
            section_intent_baseline += 1
        if boost_hit:
            section_intent_boost += 1

total = len(test_queries)

print(f"\n{'='*50}")
print(f"OVERALL ({total} queries)")
print(f"  Baseline R@5: {baseline_found}/{total} = {baseline_found/total:.1%}")
print(f"  Boosted R@5:  {boost_found}/{total} = {boost_found/total:.1%}")
print(f"  Absolute delta: {boost_found - baseline_found:+d}")
if baseline_ranks:
    print(f"  Baseline mean rank (when found): {np.mean(baseline_ranks):.2f}")
if boost_ranks:
    print(f"  Boosted mean rank (when found):  {np.mean(boost_ranks):.2f}")

print(f"\nSECTION-INTENT QUERIES ({section_intent_queries} queries)")
print(f"  Baseline: {section_intent_baseline}/{section_intent_queries} = {section_intent_baseline/max(section_intent_queries,1):.1%}")
print(f"  Boosted:  {section_intent_boost}/{section_intent_queries} = {section_intent_boost/max(section_intent_queries,1):.1%}")
print(f"  Delta:    {section_intent_boost - section_intent_baseline:+d}")

print(f"\nIMPROVEMENTS: {len(improvements)}")
for imp in improvements[:5]:
    print(f"  [{imp['strategy']}] {imp['query'][:60]:60s}  rank {imp['from_rank']} → {imp['to_rank']}")

print(f"\nREGRESSIONS: {len(regressions)}")
for reg in regressions[:5]:
    print(f"  [{reg['strategy']}] {reg['query'][:60]:60s}  rank {reg['from_rank']} → {reg['to_rank']}")

print(f"\nPER-STRATEGY R@5:")
for s in sorted(strat_total.keys()):
    b = strat_baseline[s] / strat_total[s] if strat_total[s] else 0
    bo = strat_boost[s] / strat_total[s] if strat_total[s] else 0
    print(f"  {s:<25}: baseline={b:.1%}  boosted={bo:.1%}  delta={bo-b:+.1%}")
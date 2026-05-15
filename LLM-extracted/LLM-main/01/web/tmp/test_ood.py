"""Test OOD detection: clean OOD queries vs FAQ and generated queries."""
import json, numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

model = SentenceTransformer('BAAI/bge-base-en-v1.5')
client = QdrantClient('localhost', port=6333)

# ── Clean OOD queries (nothing close to the 4 courses) ──────────────────────
OOD_QUERIES = [
    "what is the capital of Burkina Faso",
    "how do I make sourdough bread from scratch",
    "what are the best hiking trails in Patagonia",
    "how do I file taxes as a freelancer in Germany",
    "what is the airspeed velocity of an unladen swallow",
    "best way to learn piano as an adult beginner",
    "how do I train for a marathon in 6 months",
    "what's the difference between a Roth IRA and a 401k",
    "how do I fix a leaking kitchen faucet",
    "what are the symptoms of vitamin D deficiency",
    "how do I grow tomatoes in a small apartment",
    "what is the plot of Don Quixote",
    "how do I get a visa for visiting Japan",
    "best restaurants in Mexico City for street tacos",
    "how do I change a car tire on the highway",
    "what is the difference between yoga and pilates",
    "how do I knit a scarf for beginners",
    "what are the rules of cricket",
    "how do I meditate for anxiety relief",
    "what is the best way to learn Spanish as an adult",
]

# ── FAQ originals (stratified across courses) ────────────────────────────────
with open('data_cleaning/data/processed/clean.jsonl') as f:
    faq_docs = [json.loads(line) for line in f]

# Stratify by course
from collections import defaultdict
by_course = defaultdict(list)
for d in faq_docs:
    by_course[d['course']].append(d)

FAQ_QUERIES = []
for course, docs in by_course.items():
    FAQ_QUERIES.extend([d['question'] for d in docs[:15]])  # 15 per course = 60 total

# ── Generated queries (include chaos_monkey specifically) ────────────────────
with open('experiments/eval_queries.json') as f:
    qdata = json.load(f)

GEN_QUERIES = []
chaos_count = 0
for doc in qdata['queries']:
    for strategy, variations in doc['prompt_results'].items():
        for v in variations:
            GEN_QUERIES.append({'query': v, 'strategy': strategy})
            if strategy == 'chaos_monkey':
                chaos_count += 1

# ── Score all ────────────────────────────────────────────────────────────────
all_queries = OOD_QUERIES + FAQ_QUERIES + [g['query'] for g in GEN_QUERIES]
labels = (['OOD'] * len(OOD_QUERIES) + 
          ['FAQ'] * len(FAQ_QUERIES) + 
          ['GEN'] * len(GEN_QUERIES))

print(f"Scoring {len(all_queries)} queries...")
all_vecs = model.encode(all_queries, batch_size=32, show_progress_bar=True)

results = {label: [] for label in ['OOD', 'FAQ', 'GEN']}
ood_matches = []

for i, (label, vec) in enumerate(zip(labels, all_vecs)):
    hits = client.query_points(collection_name='faqs_bge_base_en_v1.5', query=vec.tolist(), limit=5, with_payload=True)
    top_score = hits.points[0].score if hits.points else 0
    results[label].append(top_score)
    
    if label == 'OOD':
        ood_matches.append({
            'query': all_queries[i],
            'top_score': top_score,
            'top_match': hits.points[0].payload.get('question','') if hits.points else '',
            'top_course': hits.points[0].payload.get('course','') if hits.points else '',
        })

# ── Score distributions ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"SCORE DISTRIBUTIONS")
print(f"{'='*60}")
print(f"{'Type':<8} {'Count':<8} {'Min':<8} {'P10':<8} {'P25':<8} {'P50':<8} {'P75':<8} {'P90':<8} {'Max':<8}")
print(f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for label in ['OOD', 'FAQ', 'GEN']:
    scores = np.array(results[label])
    print(f"{label:<8} {len(scores):<8} {scores.min():<8.3f} {np.percentile(scores,10):<8.3f} "
          f"{np.percentile(scores,25):<8.3f} {np.percentile(scores,50):<8.3f} "
          f"{np.percentile(scores,75):<8.3f} {np.percentile(scores,90):<8.3f} {scores.max():<8.3f}")

ood_scores = np.array(results['OOD'])
faq_scores = np.array(results['FAQ'])
gen_scores = np.array(results['GEN'])

# ── Bidirectional threshold analysis ─────────────────────────────────────────
print(f"\n{'='*60}")
print(f"THRESHOLD ANALYSIS")
print(f"{'='*60}")

# Direction 1: catch 90% of OOD
t_ood90 = np.percentile(ood_scores, 90)
faq_lost = (faq_scores < t_ood90).sum()
gen_lost = (gen_scores < t_ood90).sum()
print(f"\nTo catch 90% of OOD (threshold={t_ood90:.3f}):")
print(f"  FAQ below: {faq_lost}/{len(faq_scores)} ({faq_lost/len(faq_scores):.1%})")
print(f"  GEN below: {gen_lost}/{len(gen_scores)} ({gen_lost/len(gen_scores):.1%})")

# Direction 2: preserve 95% of FAQ
t_faq95 = np.percentile(faq_scores, 5)
ood_through = (ood_scores >= t_faq95).sum()
print(f"\nTo preserve 95% of FAQ (threshold={t_faq95:.3f}):")
print(f"  OOD passing through: {ood_through}/{len(ood_scores)} ({ood_through/len(ood_scores):.1%})")

# ── What do OOD queries match? ──────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"OOD QUERIES — TOP MATCHES")
print(f"{'='*60}")
for m in ood_matches:
    print(f"  [{m['top_score']:.3f}] {m['query'][:60]}")
    print(f"         → {m['top_match'][:70]} ({m['top_course']})\n")
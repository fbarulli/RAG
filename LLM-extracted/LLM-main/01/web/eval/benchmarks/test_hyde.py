"""
HyDE: Hypothetical Document Embeddings test.
Generates a fake answer, embeds it, searches — compares to direct embedding.

Run:    uv run python eval/benchmarks/test_hyde.py
"""
import sys, os, json, time, logging
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from litellm import completion

load_dotenv('configs/.env')
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

MODEL = "nvidia_nim/meta/llama-3.1-8b-instruct"
EMBEDDING = 'BAAI/bge-base-en-v1.5'
QDRANT_COLLECTION = 'faqs_bge_base_en_v1.5'
TOP_K = 5

HYDE_PROMPT = """Write a short FAQ answer (2-3 sentences) that would answer this student question:

Question: {query}

Write only the answer, nothing else. Answer:"""

# Load the 72 Topic 0 test queries
with open('experiments/topic0_queries.json') as f:
    t0_data = json.load(f)

test_queries = []
for doc in t0_data['queries']:
    for v in doc['variations']:
        test_queries.append({
            'query': v,
            'expected_id': doc['expected_id'],
            'course': doc['course'],
            'sub_topic': doc['sub_topic'],
        })

logger.info(f"Testing HyDE on {len(test_queries)} queries\n")

model = SentenceTransformer(EMBEDDING)
client = QdrantClient('localhost', port=6333)

sub_names = {0: "Vector search", 1: "Python errors", 2: "Homework/data", 3: "Hardware/Mac", 4: "Leaderboard"}

# Track per-subtopic
by_sub = {}
for s in set(tq['sub_topic'] for tq in test_queries):
    by_sub[s] = {'total': 0, 'baseline': 0, 'hyde': 0, 'hyde_attempted': 0}

baseline_found = 0
hyde_found = 0
hyde_attempted = 0
hyde_skipped = 0

for i, tq in enumerate(test_queries):
    s = tq['sub_topic']
    by_sub[s]['total'] += 1
    
    qfilter = Filter(must=[FieldCondition(key='course', match=MatchValue(value=tq['course']))])
    
    # ── Baseline: direct query embedding ─────────────────────────────────────
    query_vec = model.encode(tq['query']).tolist()
    hits = client.query_points(collection_name=QDRANT_COLLECTION, query=query_vec, limit=TOP_K, query_filter=qfilter, with_payload=True)
    hit_ids = [h.payload.get('es_id', '') for h in hits.points]
    if tq['expected_id'] in hit_ids:
        baseline_found += 1
        by_sub[s]['baseline'] += 1
    
    # ── HyDE: generate fake answer, embed that ───────────────────────────────
    fake_answer = ""
    for attempt in range(3):
        try:
            response = completion(model=MODEL, messages=[{"role":"user","content":HYDE_PROMPT.format(query=tq['query'])}], temperature=0.3, max_tokens=150)
            fake_answer = response.choices[0].message.content.strip()
            break
        except Exception as e:
            msg = str(e)
            if '429' in msg or '502' in msg or '504' in msg:
                wait = 60 * (attempt + 1)
                logger.warning(f"  [{i+1}] Rate limit, waiting {wait}s...")
                time.sleep(wait)
            else:
                logger.warning(f"  [{i+1}] HyDE failed: {e}")
                break
    
    if not fake_answer:
        hyde_skipped += 1
    
    if fake_answer:
        if i < 3: logger.info(f"  HyDE sample: '{fake_answer[:100]}...'")
        hyde_attempted += 1
        by_sub[s]['hyde_attempted'] += 1
        
        hyde_vec = model.encode(fake_answer).tolist()
        hyde_hits = client.query_points(collection_name=QDRANT_COLLECTION, query=hyde_vec, limit=TOP_K, query_filter=qfilter, with_payload=True)
        hyde_ids = [h.payload.get('es_id', '') for h in hyde_hits.points]
        if tq['expected_id'] in hyde_ids:
            hyde_found += 1
            by_sub[s]['hyde'] += 1
    
    if (i+1) % 10 == 0:
        logger.info(f"  {i+1}/{len(test_queries)} — baseline: {baseline_found}, hyde: {hyde_found}")
        with open('experiments/hyde_results.json', 'w') as f:
            json.dump({'baseline_r5': baseline_found/(i+1), 'hyde_r5': hyde_found/hyde_attempted if hyde_attempted > 0 else 0, 'by_subtopic': {str(k): v for k, v in by_sub.items()}, 'timestamp': datetime.now().isoformat()}, f, indent=2)
    
    time.sleep(1)

# ── Results ──────────────────────────────────────────────────────────────────
total = len(test_queries)
hyde_r5 = hyde_found/hyde_attempted if hyde_attempted > 0 else 0
print(f"\n{'='*55}")
print(f"HyDE RESULTS ({total} queries)")
print(f"{'='*55}")
print(f"  Baseline (direct embed): {baseline_found}/{total} = {baseline_found/total:.1%}")
print(f"  HyDE (attempted):        {hyde_found}/{hyde_attempted} = {hyde_r5:.1%}")
print(f"  HyDE (all queries):      {hyde_found}/{total} = {hyde_found/total:.1%}")
print(f"  Improvement:             {hyde_found - baseline_found:+d} queries")
print(f"  Skipped (API error):     {hyde_skipped}")

print(f"\n{'='*55}")
print("PER SUB-TOPIC")
print(f"{'='*55}")
for s in sorted(by_sub.keys()):
    v = by_sub[s]
    name = sub_names.get(s, f"Topic {s}")
    b = v['baseline'] / v['total'] if v['total'] > 0 else 0
    h = v['hyde'] / v['hyde_attempted'] if v['hyde_attempted'] > 0 else 0
    print(f"  {name:<20}: baseline={b:.0%}  hyde={h:.0%}  ({v['hyde_attempted']}/{v['total']} attempted)")

# Save
results = {
    'baseline_r5': baseline_found / total,
    'hyde_r5': hyde_r5,
    'hyde_all_r5': hyde_found / total if total > 0 else 0,
    'improvement': hyde_found - baseline_found,
    'skipped': hyde_skipped,
    'by_subtopic': {str(k): v for k, v in by_sub.items()},
    'timestamp': datetime.now().isoformat(),
}
with open('experiments/hyde_results.json', 'w') as f:
    json.dump(results, f, indent=2)
logger.info(f"Saved: experiments/hyde_results.json")
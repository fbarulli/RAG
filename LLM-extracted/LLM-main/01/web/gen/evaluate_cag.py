"""
gen/evaluate_cag.py
===================
CAG evaluation with Langfuse tracing + RAGAS metrics.
Every evaluation run is logged for long-term tracking.

Run:    uv run python gen/evaluate_cag.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, time, logging
import numpy as np
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

load_dotenv('configs/.env')
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")

from src.langfuse_config import init_langfuse, load_api_keys
load_api_keys()

from langfuse import Langfuse
langfuse = Langfuse()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

CAG_FILE = 'experiments/cag_answers.json'
QUERIES_FILE = 'experiments/eval_queries.json'
MODEL_NAME = 'BAAI/bge-base-en-v1.5'
QDRANT_COLLECTION = 'faqs_bge_base_en_v1.5'

# ── Load data ────────────────────────────────────────────────────────────────
with open(CAG_FILE) as f:
    cag = json.load(f)['answers']

with open(QUERIES_FILE) as f:
    data = json.load(f)

test_queries = []
for doc in data['queries']:
    for strategy, variations in doc['prompt_results'].items():
        for query in variations:
            test_queries.append({
                'query': query, 'expected_id': doc['expected_id'],
                'strategy': strategy, 'course': doc['course'],
            })

evaluable = [q for q in test_queries if q['expected_id'] in cag]

logger.info(f"CAG answers: {len(cag)} | Evaluable: {len(evaluable)}/{len(test_queries)}")

# ── Create Langfuse trace ────────────────────────────────────────────────────
trace = langfuse.trace(
    name=f"CAG_eval_{datetime.now().strftime('%H%M')}",
    metadata={
        'cag_answers': len(cag),
        'test_queries': len(test_queries),
        'evaluable': len(evaluable),
        'model': MODEL_NAME,
    },
)

# ── Evaluate ─────────────────────────────────────────────────────────────────
model = SentenceTransformer(MODEL_NAME)
client = QdrantClient('localhost', port=6333)

results = []
lats = []

for tq in evaluable:
    vec = model.encode(tq['query']).tolist()
    
    t0 = time.time()
    hits = client.query_points(
        collection_name=QDRANT_COLLECTION, query=vec, limit=5, with_payload=True
    )
    elapsed = (time.time() - t0) * 1000
    lats.append(elapsed)
    
    hit_ids = [h.payload.get('es_id', '') for h in hits.points]
    top_id = hit_ids[0] if hit_ids else None
    rank = next((pos for pos, hid in enumerate(hit_ids, 1) if hid == tq['expected_id']), None)
    found = rank is not None
    has_cag = top_id in cag if top_id else False
    
    results.append({
        'query': tq['query'][:80], 'found': found, 'rank': rank,
        'has_cag': has_cag, 'strategy': tq['strategy'],
        'latency_ms': round(elapsed, 1),
    })

total = len(results)
found = sum(1 for r in results if r['found'])
cag_hits = sum(1 for r in results if r['has_cag'])
cag_correct = sum(1 for r in results if r['has_cag'] and r['found'] and r['rank'] == 1)

# ── Log results to Langfuse ──────────────────────────────────────────────────
trace.span(
    name="retrieval_overall",
    output={
        'total': total, 'found': found, 'R@5': found/total,
        'cag_hits': cag_hits, 'cag_correct@1': cag_correct,
        'latency_p50': np.percentile(lats, 50),
        'latency_p95': np.percentile(lats, 95),
    },
)

# Per-strategy spans
by_strat = defaultdict(lambda: {'found': 0, 'total': 0})
for r in results:
    by_strat[r['strategy']]['total'] += 1
    by_strat[r['strategy']]['found'] += r['found']

for strategy, counts in by_strat.items():
    trace.span(
        name=f"strategy_{strategy}",
        output={'R@5': counts['found']/counts['total'], 'n': counts['total']},
    )

trace.update(output={'status': 'complete'})
langfuse.flush()

# ── Print summary ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"CAG EVALUATION ({total} evaluable)")
print(f"{'='*60}")
print(f"  R@5: {found}/{total} ({found/total:.1%})")
print(f"  CAG hits: {cag_hits}/{total} ({cag_hits/total:.1%})")
print(f"  CAG correct @1: {cag_correct}/{total} ({cag_correct/total:.1%})")
print(f"  Latency: P50={np.percentile(lats, 50):.1f}ms  P95={np.percentile(lats, 95):.1f}ms")
print(f"  Coverage: {len(cag)}/1140 ({len(cag)/1140:.1%})")

print(f"\n  Per-strategy R@5:")
for s, counts in sorted(by_strat.items()):
    print(f"    {s:<25}: {counts['found']/counts['total']:.1%} ({counts['found']}/{counts['total']})")

print(f"\n✓ Traced to Langfuse: {trace.id}")

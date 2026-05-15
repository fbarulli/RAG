"""
Compare HyDE approaches for short queries:
A) Full answer embedding (what we just generated)
B) Query rewriting (generate 3 rephrased search queries, embed each)
C) Hybrid: combine answer + rewritten queries with reciprocal rank fusion

Tests which approach closes the vocabulary gap better.

Run:    uv run python eval/benchmarks/hyde_compare.py
"""
import sys, os, json, time, logging, re, asyncio
from datetime import datetime
from typing import List, Dict, Any
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from litellm import completion

load_dotenv('configs/.env')
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
MODEL = "nvidia_nim/meta/llama-3.1-8b-instruct"
EMBEDDING = 'BAAI/bge-base-en-v1.5'
COLLECTION = 'faqs_bge_base_en_v1.5'
TOP_K = 5
MAX_TEST = 72
RETRIES = 2  # Retry failed LLM calls
TIMEOUT = 30  # Seconds for LLM call

# RRF hyperparameter (higher = more weight to top ranks)
RRF_K = 60

REWRITE_PROMPT = """Rewrite this student question into 3 specific search queries they might type.
Make each query more detailed and technical than the original.
Include specific terms, tool names, and error messages.

Short query: {query}

Return ONLY a JSON array: ["query1", "query2", "query3"]"""

ANSWER_PROMPT = """Write a short FAQ answer (2-3 sentences) that would answer this question.
Use technical terms and specific details.

Question: {query}

Answer:"""

# ── Data Loading ─────────────────────────────────────────────────────────────
with open('experiments/topic0_queries.json') as f:
    t0_data = json.load(f)

test_queries = []
for doc in t0_data['queries']:
    for v in doc['variations']:
        test_queries.append({
            'query': v,
            'expected_id': doc['expected_id'],
            'course': doc['course'],
        })

# Filter to short queries, take first MAX_TEST
short = [q for q in test_queries if len(q['query'].split()) < 8]
short = short[:MAX_TEST]

logger.info(f"Comparing on {len(short)} short queries\n")

# ── Model Loading ────────────────────────────────────────────────────────────
model = SentenceTransformer(EMBEDDING)
client = QdrantClient('localhost', port=6333)

# ── Helper Functions ───────────────────────────────────────────────────────

def safe_llm_call(prompt: str, temperature: float = 0.3, max_tokens: int = 150) -> str:
    """Call LLM with retries and error handling."""
    for attempt in range(RETRIES):
        try:
            resp = completion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=TIMEOUT
            )
            content = resp.choices[0].message.content.strip()
            if content:
                return content
        except Exception as e:
            logger.warning(f"LLM call failed (attempt {attempt + 1}/{RETRIES}): {e}")
            if attempt < RETRIES - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    return ""


def parse_json_array(raw: str) -> List[str]:
    """Robustly parse LLM JSON output."""
    # Remove code fences
    raw = re.sub(r'```(?:json)?\s*', '', raw).strip()
    raw = raw.replace('```', '').strip()

    # Try direct parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if x]
    except json.JSONDecodeError:
        pass

    # Try extracting array with regex
    match = re.search(r'\[(.*?)\]', raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(f"[{match.group(1)}]")
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if x]
        except json.JSONDecodeError:
            pass

    # Fallback: split by newlines/numbers
    lines = [re.sub(r'^\s*\d+[.)]\s*', '', line).strip() 
             for line in raw.split('\n') if line.strip()]
    return [l for l in lines if l and (l[0] == '"' or l[0].isalnum())]


def search_qdrant(query_vec: List[float], course: str, limit: int = TOP_K) -> List[Dict[str, Any]]:
    """Execute Qdrant search with course filter."""
    qfilter = Filter(must=[FieldCondition(key='course', match=MatchValue(value=course))])
    hits = client.query_points(
        collection_name=COLLECTION,
        query=query_vec,
        limit=limit,
        query_filter=qfilter,
        with_payload=True
    )
    return [
        {
            'id': h.payload.get('es_id', ''),
            'score': h.score,
            'payload': h.payload
        }
        for h in hits.points
    ]


def reciprocal_rank_fusion(rankings: List[List[str]], k: int = RRF_K) -> List[str]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.
    score = sum(1 / (k + rank)) for each document across all lists.
    """
    scores: Dict[str, float] = {}

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            if doc_id:
                scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)

    # Sort by score descending
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs]


def check_hit(doc_id: str, hits: List[Dict[str, Any]]) -> bool:
    """Check if expected document is in hits."""
    return any(h['id'] == doc_id for h in hits)


# ── Main Evaluation ──────────────────────────────────────────────────────────
results = {
    'baseline': [],
    'answer_embed': [],
    'query_rewrite': [],
    'hybrid_rrf': [],
    'latency_ms': {
        'baseline': [],
        'answer_embed': [],
        'query_rewrite': [],
        'hybrid_rrf': [],
    }
}

for i, sq in enumerate(short):
    expected_id = sq['expected_id']
    course = sq['course']

    # ── Baseline: direct embedding ─────────────────────────────────────────
    t0 = time.perf_counter()
    vec = model.encode(sq['query']).tolist()
    hits_baseline = search_qdrant(vec, course)
    baseline_found = check_hit(expected_id, hits_baseline)
    results['baseline'].append(baseline_found)
    results['latency_ms']['baseline'].append((time.perf_counter() - t0) * 1000)

    # ── Approach A: Full answer embedding ──────────────────────────────────
    t0 = time.perf_counter()
    answer = safe_llm_call(ANSWER_PROMPT.format(query=sq['query']), temperature=0.3, max_tokens=150)

    if answer:
        vec_a = model.encode(answer).tolist()
        hits_a = search_qdrant(vec_a, course)
        answer_found = check_hit(expected_id, hits_a)
    else:
        hits_a = []
        answer_found = False

    results['answer_embed'].append(answer_found)
    results['latency_ms']['answer_embed'].append((time.perf_counter() - t0) * 1000)

    # ── Approach B: Query rewriting ──────────────────────────────────────
    t0 = time.perf_counter()
    raw = safe_llm_call(REWRITE_PROMPT.format(query=sq['query']), temperature=0.5, max_tokens=200)
    rewritten = parse_json_array(raw) if raw else []

    rewrite_found = False
    all_rewrite_hits = []
    for rq in rewritten[:3]:  # Limit to 3 queries
        vec_b = model.encode(rq).tolist()
        hits_b = search_qdrant(vec_b, course)
        all_rewrite_hits.append([h['id'] for h in hits_b])
        if check_hit(expected_id, hits_b):
            rewrite_found = True
            # Don't break - collect all for hybrid

    results['query_rewrite'].append(rewrite_found)
    results['latency_ms']['query_rewrite'].append((time.perf_counter() - t0) * 1000)

    # ── Approach C: Hybrid RRF ───────────────────────────────────────────
    t0 = time.perf_counter()
    # Combine: baseline + answer_embed + all rewritten queries
    all_rankings = [
        [h['id'] for h in hits_baseline],
        [h['id'] for h in hits_a],
    ] + all_rewrite_hits

    fused = reciprocal_rank_fusion(all_rankings)[:TOP_K]
    hybrid_found = expected_id in fused
    results['hybrid_rrf'].append(hybrid_found)
    results['latency_ms']['hybrid_rrf'].append((time.perf_counter() - t0) * 1000)

    # ── Logging ────────────────────────────────────────────────────────────
    if i < 3:
        logger.info(f"  [{i+1}] Query: '{sq['query']}'")
        logger.info(f"    Baseline: {'✓' if baseline_found else '✗'}")
        logger.info(f"    Answer:   {'✓' if answer_found else '✗'} (len={len(answer) if answer else 0})")
        logger.info(f"    Rewrite:  {'✓' if rewrite_found else '✗'} ({len(rewritten)} queries)")
        logger.info(f"    Hybrid:   {'✓' if hybrid_found else '✗'} (fused top-{len(fused)})")

    time.sleep(0.5)  # Reduced from 1s

# ── Results ─────────────────────────────────────────────────────────────────
n = len(short)
print(f"\n{'='*60}")
print(f"HyDE APPROACH COMPARISON ({n} short queries)")
print(f"{'='*60}")

for method, scores in results.items():
    if method == 'latency_ms':
        continue
    found = sum(scores)
    pct = found / n if n > 0 else 0
    print(f"  {method:<20}: {found}/{n} = {pct:.1%}")

print(f"\n{'='*60}")
print("LATENCY COMPARISON (median ms)")
print(f"{'='*60}")
for method, times in results['latency_ms'].items():
    if times:
        median_ms = np.median(times)
        p95_ms = np.percentile(times, 95)
        print(f"  {method:<20}: median={median_ms:.1f}ms  p95={p95_ms:.1f}ms")

# ── Per-query analysis for failure patterns ─────────────────────────────────
print(f"\n{'='*60}")
print("PER-QUERY BREAKDOWN (failures only)")
print(f"{'='*60}")
for i, sq in enumerate(short):
    if not any([
        results['baseline'][i],
        results['answer_embed'][i],
        results['query_rewrite'][i],
        results['hybrid_rrf'][i]
    ]):
        print(f"  [{i+1}] ALL FAILED: '{sq['query']}'")
    elif not results['hybrid_rrf'][i] and any([
        results['baseline'][i],
        results['answer_embed'][i],
        results['query_rewrite'][i]
    ]):
        print(f"  [{i+1}] Hybrid failed (others passed): '{sq['query']}'")

# ── Statistical significance test ────────────────────────────────────────────
from scipy.stats import fisher_exact

print(f"\n{'='*60}")
print("STATISTICAL SIGNIFICANCE (vs Baseline)")
print(f"{'='*60}")
for method in ['answer_embed', 'query_rewrite', 'hybrid_rrf']:
    # Build 2x2 contingency table
    a = sum(1 for i in range(n) if results['baseline'][i] and results[method][i])  # Both pass
    b = sum(1 for i in range(n) if results['baseline'][i] and not results[method][i])  # Baseline only
    c = sum(1 for i in range(n) if not results['baseline'][i] and results[method][i])  # Method only
    d = sum(1 for i in range(n) if not results['baseline'][i] and not results[method][i])  # Both fail

    if a + b + c + d > 0:
        oddsratio, pvalue = fisher_exact([[a, b], [c, d]])
        print(f"  {method:<20}: OR={oddsratio:.2f}, p={pvalue:.3f} {'*' if pvalue < 0.05 else ''}")
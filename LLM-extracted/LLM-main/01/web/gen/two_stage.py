"""
gen/two_stage.py
=================
Two-stage retrieval: CAG (fast cache) → RAG (LLM fallback).
Evaluates on the 420 test queries with confidence-based routing.

Run:    uv run python gen/two_stage.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, time
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

MODEL_NAME = 'BAAI/bge-base-en-v1.5'
QDRANT_COLLECTION = 'faqs_bge_base_en_v1.5'
CAG_FILE = 'experiments/cag_answers.json'
QUERIES_FILE = 'experiments/eval_queries.json'
SIMILARITY_THRESHOLD = 0.70
TOP_K = 5


class TwoStagePipeline:
    def __init__(self):
        print("Loading models...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.client = QdrantClient('localhost', port=6333)
        self.cag = self._load_cag()
        self.stats = {'cag_hit': 0, 'cag_miss': 0, 'rag_would_fallback': 0,
                      'cag_correct': 0, 'rag_would_be_needed': 0}
    
    def _load_cag(self) -> dict:
        if os.path.exists(CAG_FILE):
            with open(CAG_FILE) as f:
                return json.load(f)['answers']
        return {}
    
    def query(self, question: str) -> dict:
        """Two-stage query: CAG first, then RAG fallback if confidence low."""
        vec = self.model.encode(question).tolist()
        
        # Stage 1: CAG lookup
        results = self.client.query_points(
            collection_name=QDRANT_COLLECTION, query=vec, limit=1, with_payload=True
        )
        
        if not results.points:
            return {'stage': 'rag_fallback', 'reason': 'no_results'}
        
        top_hit = results.points[0]
        top_score = top_hit.score
        top_id = top_hit.payload.get('es_id', '')
        top_question = top_hit.payload.get('question', '')
        
        # Check if cached answer exists
        if top_id in self.cag:
            cached = self.cag[top_id]
            
            if top_score >= SIMILARITY_THRESHOLD:
                # Stage 1: CAG hit — fast, cached answer
                return {
                    'stage': 'cag',
                    'similarity': round(top_score, 4),
                    'faq_question': top_question,
                    'cached_answer': cached['generated_answer'],
                    'course': cached.get('course', ''),
                }
            else:
                # Stage 2 needed: low confidence
                return {
                    'stage': 'rag_fallback',
                    'reason': f'low_similarity ({top_score:.3f} < {SIMILARITY_THRESHOLD})',
                    'faq_question': top_question,
                    'similarity': round(top_score, 4),
                    'has_cached': True,
                }
        else:
            # No cached answer exists yet
            return {
                'stage': 'rag_fallback',
                'reason': 'no_cached_answer',
                'faq_question': top_question,
                'similarity': round(top_score, 4),
                'has_cached': False,
            }


def evaluate():
    pipeline = TwoStagePipeline()
    
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
    
    print(f"CAG answers: {len(pipeline.cag)}")
    print(f"Test queries: {len(test_queries)}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}\n")
    
    results = []
    for tq in test_queries:
        t0 = time.time()
        result = pipeline.query(tq['query'])
        result['latency_ms'] = round((time.time() - t0) * 1000, 1)
        result['expected_id'] = tq['expected_id']
        result['strategy'] = tq['strategy']
        result['course'] = tq['course']
        results.append(result)
    
    # Stats
    cag_hits = [r for r in results if r['stage'] == 'cag']
    rag_falls = [r for r in results if r['stage'] == 'rag_fallback']
    
    print(f"{'='*60}")
    print(f"RESULTS ({len(test_queries)} queries)")
    print(f"{'='*60}")
    print(f"  CAG hits:      {len(cag_hits)} ({len(cag_hits)/len(results):.1%})")
    print(f"  RAG fallbacks: {len(rag_falls)} ({len(rag_falls)/len(results):.1%})")
    
    if cag_hits:
        cag_lat = [r['latency_ms'] for r in cag_hits]
        print(f"\n  CAG latency: P50={np.percentile(cag_lat, 50):.1f}ms  P95={np.percentile(cag_lat, 95):.1f}ms")
        # CAG accuracy: did the cached FAQ answer match the expected FAQ?
        cag_correct = sum(1 for r in cag_hits if r.get('expected_id') == r.get('faq_question'))
    
    if rag_falls:
        rag_lat = [r['latency_ms'] for r in rag_falls]
        print(f"  RAG lookup latency: P50={np.percentile(rag_lat, 50):.1f}ms (LLM call not included)")
        
        # Breakdown of fallback reasons
        from collections import Counter
        reasons = Counter(r['reason'].split('(')[0].strip() for r in rag_falls)
        print(f"\n  Fallback reasons:")
        for reason, count in reasons.most_common():
            print(f"    {reason}: {count}")
        
        # How many fallbacks have cached answers?
        has_cache = sum(1 for r in rag_falls if r.get('has_cached'))
        print(f"\n  Fallbacks with existing CAG answer: {has_cache}/{len(rag_falls)}")
    
    # Per-strategy
    print(f"\n{'='*60}")
    print(f"PER-STRATEGY")
    from collections import defaultdict
    by_strat = defaultdict(lambda: {'cag': 0, 'rag': 0, 'total': 0})
    for r in results:
        by_strat[r['strategy']][r['stage'].replace('rag_fallback', 'rag')] += 1
        by_strat[r['strategy']]['total'] += 1
    for s in sorted(by_strat):
        m = by_strat[s]
        print(f"  {s:<25}: {m['cag']} CAG / {m['rag']} RAG ({m['cag']/m['total']:.0%} cached)")


if __name__ == '__main__':
    evaluate()

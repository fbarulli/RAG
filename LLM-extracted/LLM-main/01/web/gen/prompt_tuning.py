"""
gen/prompt_tuning.py
=====================
Tests prompt variants for CAG answer generation.
Shared rate-limit backoff across concurrent calls.
Saves incrementally after every document.

Run:    uv run python gen/prompt_tuning.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, time, asyncio, random, logging, re
from collections import defaultdict
from dotenv import load_dotenv
from litellm import acompletion
from qdrant_client import QdrantClient

load_dotenv('configs/.env')
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

MODEL = "nvidia_nim/meta/llama-3.1-70b-instruct"
TEST_SIZE = 10
GAP = 3.0
PROMPTS_FILE = 'eval/generation/prompts_cag.json'
OUTPUT = 'experiments/prompt_tuning.json'

JUDGE_PROMPT = """Score this answer on three criteria from 1-5:

Question: {question}
Source material: {source}
Generated answer: {answer}

1. COMPLETENESS (1-5): Does it include the key details from the source?
2. ACCURACY (1-5): Is all information factually correct vs the source?
3. TONE (1-5): Is it helpful, natural, and appropriate for a course TA?

Return ONLY a JSON object: {{"completeness": N, "accuracy": N, "tone": N}}"""


def load_prompts():
    with open(PROMPTS_FILE) as f:
        return json.load(f)


def get_sample_docs(n=TEST_SIZE):
    client = QdrantClient('localhost', port=6333)
    scroll_result = client.scroll(collection_name='faqs', limit=200, with_payload=True)
    all_docs = [p.payload for p in scroll_result[0]]
    
    by_course = defaultdict(list)
    for d in all_docs:
        by_course[d['course']].append(d)
    
    random.seed(42)
    sampled = []
    per_course = max(1, n // len(by_course))
    for course, docs in by_course.items():
        sampled.extend(random.sample(docs, min(per_course, len(docs))))
    sampled = sampled[:n]
    course_counts = dict((c, sum(1 for d in sampled if d['course']==c)) for c in by_course)
    logger.info(f"Sampled {len(sampled)} docs: {course_counts}")
    return sampled


class RateGate:
    """Shared rate-limit gate across all concurrent calls."""
    def __init__(self):
        self.limited_until = 0
    
    async def wait_if_needed(self):
        now = time.monotonic()
        if now < self.limited_until:
            wait = self.limited_until - now
            logger.warning(f"Rate gate: all calls waiting {wait:.0f}s...")
            await asyncio.sleep(wait)
    
    def hit_limit(self):
        self.limited_until = time.monotonic() + 60
        logger.warning(f"Rate limit hit! Gate closed for 60s.")


rate_gate = RateGate()


async def generate_answer(doc: dict, template: str) -> tuple[str, float]:
    prompt = template.format(question=doc['question'], answer=doc['answer'][:1000], course=doc.get('course', 'this'))
    
    await rate_gate.wait_if_needed()
    
    t0 = time.time()
    for attempt in range(3):
        try:
            response = await acompletion(model=MODEL, messages=[{"role":"user","content":prompt}], temperature=0.3, max_tokens=300)
            return response.choices[0].message.content.strip(), time.time() - t0
        except Exception as e:
            if any(c in str(e) for c in ['429','502','504']):
                rate_gate.hit_limit()
                await rate_gate.wait_if_needed()
            else:
                await asyncio.sleep(5)
    return "", time.time() - t0


async def score_answer(question: str, source: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, source=source[:500], answer=answer)
    try:
        response = await acompletion(model=MODEL, messages=[{"role":"user","content":prompt}], temperature=0, max_tokens=100)
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'```(?:json)?|```', '', raw).strip()
        return json.loads(raw)
    except:
        return {"completeness": 0, "accuracy": 0, "tone": 0}


async def test_doc(doc: dict, prompts: dict) -> dict:
    """Run all prompts concurrently for one document."""
    tasks = [
        generate_answer(doc, info['template'])
        for name, info in prompts.items()
    ]
    answers = await asyncio.gather(*tasks)
    
    result = {'question': doc['question'], 'course': doc['course']}
    for (name, info), (answer, elapsed) in zip(prompts.items(), answers):
        result[name] = {'answer': answer, 'latency': round(elapsed, 1)}
    return result


def save_progress(results: list, winner: str = None):
    prompts = load_prompts()
    scores = {}
    if results:
        for name in prompts:
            scored = []
            for r in results:
                if name in r and r[name].get('score'):
                    scored.append(r[name]['score'])
            if scored:
                scores[name] = {
                    'avg': {k: sum(s[k] for s in scored)/len(scored) for k in ['completeness','accuracy','tone']}
                }
    
    output = {
        'metadata': {'model': MODEL, 'test_size': TEST_SIZE, 'winner': winner, 'count': len(results)},
        'scores': scores,
        'results': results,
    }
    with open(OUTPUT, 'w') as f:
        json.dump(output, f, indent=2)


async def main():
    prompts = load_prompts()
    docs = get_sample_docs(TEST_SIZE)
    
    logger.info(f"Testing {len(prompts)} prompts on {len(docs)} questions\n")
    
    # Load existing progress
    existing_results = []
    existing_questions = set()
    if os.path.exists(OUTPUT):
        with open(OUTPUT) as f:
            existing = json.load(f)
            existing_results = existing.get('results', [])
            existing_questions = {r['question'] for r in existing_results}
            logger.info(f"Resuming: {len(existing_results)} already done")
    
    all_results = existing_results.copy()
    
    for i, doc in enumerate(docs):
        if doc['question'] in existing_questions:
            continue
        
        result = await test_doc(doc, prompts)
        all_results.append(result)
        
        # Show what we got
        for name in prompts:
            ans = result[name]['answer']
            logger.info(f"  [{name}] {ans[:60]}...")
        
        # Save incrementally
        save_progress(all_results)
        
        if i < len(docs) - 1:
            await asyncio.sleep(GAP)
    
    # Score all answers
    logger.info(f"\nScoring {len(all_results)} answers...")
    for result in all_results:
        if any(result[name].get('score') for name in prompts):
            continue  # Already scored
        
        # Find source
        source = ""
        for d in docs:
            if d['question'] == result['question']:
                source = d['answer'][:500]
                break
        
        for name in prompts:
            if result[name]['answer'] and not result[name].get('score'):
                score = await score_answer(result['question'], source, result[name]['answer'])
                result[name]['score'] = score
                await asyncio.sleep(0.5)
        
        save_progress(all_results)
    
    # Determine winner
    totals = {}
    for name in prompts:
        scored = [r[name]['score'] for r in all_results if r[name].get('score')]
        if scored:
            totals[name] = sum(
                s['completeness'] + s['accuracy'] + s['tone']
                for s in scored
            ) / len(scored)
    
    winner = max(totals, key=totals.get) if totals else None
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"PROMPT COMPARISON (LLM-judged, 1-5 scale)")
    print(f"{'='*70}")
    print(f"{'Prompt':<25} {'Complete':>10} {'Accurate':>10} {'Tone':>8} {'Total':>8}")
    print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    
    for name in prompts:
        scored = [r[name]['score'] for r in all_results if r[name].get('score')]
        if scored:
            c = sum(s['completeness'] for s in scored) / len(scored)
            a = sum(s['accuracy'] for s in scored) / len(scored)
            t = sum(s['tone'] for s in scored) / len(scored)
            total = c + a + t
            marker = " ← WINNER" if name == winner else ""
            print(f"{name:<25} {c:>9.1f} {a:>10.1f} {t:>8.1f} {total:>7.1f}{marker}")
    
    print(f"\nWinner: {winner}")
    save_progress(all_results, winner)
    logger.info(f"Saved: {OUTPUT}")


if __name__ == '__main__':
    asyncio.run(main())

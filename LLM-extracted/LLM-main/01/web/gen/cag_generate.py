"""
gen/cag_generate.py
===================
Cache-Augmented Generation: Pre-compute answers for FAQ questions.
Stores answers in JSON keyed by FAQ ID.

Run:    uv run python gen/cag_generate.py
        uv run python gen/cag_generate.py --all    # Generate all 1150
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, time, asyncio, logging, random, argparse
from collections import defaultdict
from dotenv import load_dotenv
from litellm import acompletion
from qdrant_client import QdrantClient

load_dotenv('configs/.env')
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

MODEL = "nvidia_nim/meta/llama-3.1-70b-instruct"
GAP = 3.0
OUTPUT = 'experiments/cag_answers.json'

PROMPT = """You are a course teaching assistant. Answer the question based on the FAQ context below.
Use only the information provided. Keep your answer clear and concise.

FAQ Question: {question}
FAQ Answer: {answer}

Your answer:"""


def load_all_docs():
    """Load all documents from Qdrant."""
    client = QdrantClient('localhost', port=6333)
    all_docs = []
    offset = 0
    while True:
        results = client.scroll(collection_name='faqs_bge_base_en_v1.5', limit=200, offset=offset, with_payload=True)
        points, next_offset = results
        if not points:
            break
        for p in points:
            all_docs.append(p.payload)
        offset = next_offset
        if len(all_docs) > 2000:
            break
    return all_docs


async def generate_answer(doc: dict) -> str:
    prompt = PROMPT.format(question=doc['question'], answer=doc['answer'][:1000])
    
    for attempt in range(3):
        try:
            response = await acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e)
            if any(code in msg for code in ['429', '502', '504']):
                wait = 60 * (attempt + 1)
                logger.warning(f"Rate limit (attempt {attempt+1}), waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                logger.error(f"Error: {e}")
                await asyncio.sleep(5)
    return ""


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Generate for all 1150 FAQs')
    parser.add_argument('--n', type=int, default=25, help='Number of FAQs (default: 25)')
    args = parser.parse_args()
    
    all_docs = load_all_docs()
    logger.info(f"Loaded {len(all_docs)} docs from Qdrant")
    
    # Load existing answers to skip completed ones
    existing = {}
    if os.path.exists(OUTPUT):
        with open(OUTPUT) as f:
            existing = json.load(f).get('answers', {})
        logger.info(f"Found {len(existing)} existing answers")
    
    # Filter to unprocessed
    pending = [d for d in all_docs if d['es_id'] not in existing]
    
    if not args.all:
        # Sample evenly across courses
        random.seed(42)
        by_course = defaultdict(list)
        for d in pending:
            by_course[d['course']].append(d)
        
        sampled = []
        per_course = max(1, args.n // len(by_course))
        for course, docs in by_course.items():
            sampled.extend(random.sample(docs, min(per_course, len(docs))))
        pending = sampled[:args.n]
    
    if not pending:
        logger.info("All answers already generated!")
        return
    
    logger.info(f"Generating {len(pending)} answers")
    logger.info(f"Model: {MODEL} | Gap: {GAP}s\n")
    
    generated = 0
    failed = 0
    answers = dict(existing)
    
    for i, doc in enumerate(pending):
        t0 = time.time()
        answer = await generate_answer(doc)
        elapsed = time.time() - t0
        
        if answer:
            answers[doc['es_id']] = {
                'question': doc['question'],
                'original_answer': doc['answer'],
                'generated_answer': answer,
                'course': doc['course'],
            }
            generated += 1
            logger.info(f"  [{i+1}/{len(pending)}] ✓ {elapsed:.1f}s | {doc['course']} | {doc['question'][:50]}...")
        else:
            failed += 1
            logger.warning(f"  [{i+1}/{len(pending)}] ✗ FAILED | {doc['question'][:50]}...")
        
        # Save incrementally
        with open(OUTPUT, 'w') as f:
            json.dump({'metadata': {'model': MODEL, 'total': len(answers), 'generated': generated, 'failed': failed}, 'answers': answers}, f, indent=2)
        
        if i < len(pending) - 1:
            await asyncio.sleep(GAP)
    
    logger.info(f"\nDone: {generated} generated, {failed} failed → {OUTPUT}")


if __name__ == '__main__':
    asyncio.run(main())

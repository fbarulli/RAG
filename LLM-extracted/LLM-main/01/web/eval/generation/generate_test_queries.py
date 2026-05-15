"""
eval/generation/generate_test_queries.py
=========================================
Generates test queries using 3 prompt strategies.
Logs progress, timing, and errors without showing raw model output.

Output: experiments/eval_queries.json

Run:    uv run python eval/generation/generate_test_queries.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
import json
import asyncio
import random
import time
import logging
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv
from litellm import acompletion
from qdrant_client import QdrantClient

load_dotenv('configs/.env')
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

MODEL = "nvidia_nim/meta/llama-3.1-70b-instruct"
PROMPTS_FILE = 'eval/generation/prompts.json'
OUTPUT = 'experiments/eval_queries.json'
BATCH_SIZE = 5
TOTAL_DOCS = 50
GAP_BETWEEN_CALLS = 3.0
MAX_RETRIES = 3


def load_prompts() -> dict:
    with open(PROMPTS_FILE) as f:
        return json.load(f)


def get_diverse_sample(client, n=TOTAL_DOCS) -> list:
    logger.info("Sampling documents from Qdrant...")
    all_docs = []
    offset = 0
    max_docs = 1200  # Safety limit
    while len(all_docs) < max_docs:
        results = client.scroll(collection_name='faqs', limit=100, offset=offset, with_payload=True)
        points, next_offset = results
        if not points:
            break
        for p in points:
            all_docs.append(p.payload)
        offset = next_offset
        if next_offset is None:
            break

    groups = defaultdict(list)
    for doc in all_docs:
        key = f"{doc['course']}|{doc.get('section', 'general')}"
        groups[key].append(doc)

    sampled = []
    random.seed(42)
    group_keys = list(groups.keys())
    random.shuffle(group_keys)
    for key in group_keys:
        if len(sampled) >= n:
            break
        if groups[key]:
            sampled.append(random.choice(groups[key]))

    logger.info(f"Sampled {len(sampled)} docs from {len(groups)} courses")
    return sampled[:n]


def build_qa_pairs(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(
            f"FAQ {i}:\n"
            f"QUESTION: {doc['question']}\n"
            f"ANSWER: {doc['answer'][:400]}\n"
        )
    return "\n".join(parts)


def clean_json(raw: str) -> dict:
    text = re.sub(r'```(?:json)?|```', '', raw).strip()
    depth = 0
    start = text.find('{')
    if start == -1:
        return {}
    for i, c in enumerate(text[start:], start):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except:
                    break
    try:
        return json.loads(text)
    except:
        return {}


async def call_llm_with_retry(prompt, max_retries=MAX_RETRIES, temperature=0.7):
    """Call LLM with retry logic for rate limits."""
    for attempt in range(max_retries):
        try:
            response = await acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e)
            if any(code in msg for code in ['429', '502', '504']):
                wait = 60 * (attempt + 1)
                logger.warning(f"Rate limit (attempt {attempt+1}/{max_retries}), waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                logger.error(f"LLM error: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(10)
    return "{}"


async def generate_batch(docs: list, prompt_name: str, info: dict) -> list:
    """Generate queries for one batch using one prompt strategy."""
    template = info['template']
    temperature = info.get('temperature', 0.7)
    qa_pairs = build_qa_pairs(docs)
    prompt = template.format(qa_pairs=qa_pairs)
    
    try:
        raw = await call_llm_with_retry(prompt, temperature=temperature)
        result = clean_json(raw)

        output = []
        for i, doc in enumerate(docs, 1):
            variations = result.get(str(i), [])
            if not isinstance(variations, list):
                variations = []
            output.append({
                'original_question': doc['question'],
                'expected_id': doc['es_id'],
                'course': doc['course'],
                'section': doc.get('section', ''),
                'prompt_strategy': prompt_name,
                'variations': variations[:3],
            })
        return output

    except Exception as e:
        logger.error(f"[{prompt_name}] Generation failed: {e}")
        return [
            {'original_question': doc['question'], 'expected_id': doc['es_id'],
             'course': doc['course'], 'section': doc.get('section', ''),
             'prompt_strategy': prompt_name, 'variations': []}
            for doc in docs
        ]


async def main():
    start_time = time.time()
    
    prompts = load_prompts()
    client = QdrantClient('localhost', port=6333)
    docs = get_diverse_sample(client, TOTAL_DOCS)
    
    total_batches = len(docs) // BATCH_SIZE
    total_calls = len(prompts) * total_batches
    
    logger.info(f"Starting generation:")
    logger.info(f"  Docs: {len(docs)} | Batch size: {BATCH_SIZE} | Batches: {total_batches}")
    logger.info(f"  Strategies: {list(prompts.keys())} | Total LLM calls: {total_calls}")
    logger.info(f"  Gap between calls: {GAP_BETWEEN_CALLS}s")
    logger.info(f"  Estimated time: ~{total_calls * (GAP_BETWEEN_CALLS + 5) / 60:.0f} min")

    all_results = []
    total_queries_generated = 0
    errors = 0
    
    for batch_num in range(0, len(docs), BATCH_SIZE):
        batch = docs[batch_num:batch_num + BATCH_SIZE]
        if len(batch) < BATCH_SIZE:
            continue
        
        bn = batch_num // BATCH_SIZE + 1
        courses = set(d['course'] for d in batch)
        batch_start = time.time()
        
        logger.info(f"[Batch {bn}/{total_batches}] courses: {courses}")
        
        for prompt_name, info in prompts.items():
            call_start = time.time()
            results = await generate_batch(batch, prompt_name, info)
            call_elapsed = time.time() - call_start
            
            all_results.extend(results)
            count = sum(len(r['variations']) for r in results)
            total_queries_generated += count
            
            if count == 0:
                errors += 1
                logger.warning(f"  [{prompt_name}]: 0 queries generated ({call_elapsed:.1f}s)")
            else:
                logger.info(f"  [{prompt_name}]: {count} queries ({call_elapsed:.1f}s)")
            
            # Gap between calls
            await asyncio.sleep(GAP_BETWEEN_CALLS)
        
        batch_elapsed = time.time() - batch_start
        batch_total = sum(len(r['variations']) for r in all_results[-(len(prompts)*BATCH_SIZE):])
        logger.info(f"  Batch {bn} done: {batch_total} queries in {batch_elapsed:.1f}s")
    
    # ── Group by document ─────────────────────────────────────────────────────
    by_doc = defaultdict(lambda: {'prompt_results': {}})
    for r in all_results:
        key = r['original_question']
        by_doc[key].update({
            'original_question': r['original_question'],
            'expected_id': r['expected_id'],
            'course': r['course'],
            'section': r['section'],
        })
        by_doc[key]['prompt_results'][r['prompt_strategy']] = r['variations']

    queries = list(by_doc.values())
    total_queries = sum(len(v) for doc in queries for v in doc['prompt_results'].values())

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        'metadata': {
            'description': 'Test queries for retrieval evaluation',
            'model': MODEL,
            'generated_at': datetime.now().isoformat(),
            'total_documents': len(queries),
            'total_queries': total_queries,
            'batch_size': BATCH_SIZE,
            'prompt_strategies': list(prompts.keys()),
            'generation_time_seconds': round(time.time() - start_time, 1),
            'errors': errors,
        },
        'queries': queries,
    }

    os.makedirs('experiments', exist_ok=True)
    with open(OUTPUT, 'w') as f:
        json.dump(output, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*50}")
    logger.info(f"GENERATION COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"  Documents: {len(queries)}")
    logger.info(f"  Total queries: {total_queries}")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    
    # Per-strategy stats
    for strategy in prompts.keys():
        count = sum(len(doc['prompt_results'].get(strategy, [])) for doc in queries)
        logger.info(f"  [{strategy}]: {count} queries")
    
    logger.info(f"  Output: {OUTPUT}")


if __name__ == '__main__':
    asyncio.run(main())

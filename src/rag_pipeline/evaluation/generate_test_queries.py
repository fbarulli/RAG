"""
generate_test_queries.py
============================
Generates test queries from a stratified document sample using 3 prompt strategies.
Saves incrementally — safe to resume after interruption.

Output: rag_pipeline/p01_data_cleaning/data/processed/eval_queries.json

Run:
    uv run python -m rag_pipeline.evaluation.generate_test_queries
    uv run python -m rag_pipeline.evaluation.generate_test_queries --n-docs 308 --batch-size 5
"""
import argparse
import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from rag_pipeline.logging import get_logger
from rag_pipeline.core.paths import Paths
logger = get_logger(__name__)
DEFAULT_N_DOCS = 308
DEFAULT_BATCH_SIZE = 5
DEFAULT_OUTPUT = Paths.processed_dir() / 'eval_queries.json'
DEFAULT_PROMPTS = Path('rag_pipeline/evaluation/prompts.json')
DEFAULT_TOPIC_ASSIGNMENTS = Path('rag_pipeline/p02_eda/experiments/topic_assignments_all.json')
DEFAULT_CLEAN = Paths.processed_dir() / 'clean.jsonl'
DEFAULT_MODEL = 'BAAI/bge-base-en-v1.5'
JUDGE_LLM = 'nvidia_nim/meta/llama-3.1-70b-instruct'
GAP_BETWEEN_CALLS = 3.0
MAX_RETRIES = 3

def load_documents_with_topics(clean_path: Path, topic_assignments_path: Path, model: str=DEFAULT_MODEL) -> list[dict]:
    assignments = json.load(open(topic_assignments_path))
    topic_map = {a['id']: a for a in assignments['results'][model]['assignments']}
    docs = []
    for line in open(clean_path, encoding='utf-8'):
        if not line.strip():
            continue
        doc = json.loads(line)
        topic_info = topic_map.get(doc['id'], {})
        doc['topic'] = topic_info.get('topic', -1)
        doc['ner_category'] = topic_info.get('ner_category', 'OTHER')
        doc['ner_primary_entity'] = topic_info.get('ner_primary_entity')
        docs.append(doc)
    logger.info(f'Loaded {len(docs)} documents with topic assignments')
    return docs

def stratified_sample(docs: list[dict], n: int, seed: int=42, exclude_outliers: bool=True) -> list[dict]:
    import random
    from collections import defaultdict
    random.seed(seed)
    if exclude_outliers:
        docs = [d for d in docs if d.get('topic', -1) != -1]
        logger.info(f'After outlier exclusion: {len(docs)} docs')
    groups: dict[str, list[dict]] = defaultdict(list)
    for doc in docs:
        key = f"{doc['course']}::{doc.get('topic', -1)}"
        groups[key].append(doc)
    total = len(docs)
    sampled = []
    for key, pool in groups.items():
        target = max(1, round(len(pool) / total * n))
        target = min(target, len(pool))
        sampled.extend(random.sample(pool, target))
    random.shuffle(sampled)
    sampled = sampled[:n]
    logger.info(f'Stratified sample: {len(sampled)} docs from {len(groups)} groups')
    return sampled

def _init_env():
    load_dotenv('.env')
    os.environ['NVIDIA_NIM_API_KEY'] = os.getenv('NVIDIA_API_KEY', '')

async def call_llm_async(prompt: str, temperature: float=0.7, max_tokens: int=500) -> str:
    from litellm import acompletion
    _init_env()
    for attempt in range(MAX_RETRIES):
        try:
            resp = await acompletion(model=JUDGE_LLM, messages=[{'role': 'user', 'content': prompt}], temperature=temperature, max_tokens=max_tokens)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e)
            is_rl = any((code in msg for code in ['429', '502', '504', 'RateLimitError']))
            wait = 60 * (attempt + 1) if is_rl else 10
            if is_rl:
                logger.warning(f'Rate limit (attempt {attempt + 1}/{MAX_RETRIES}) — waiting {wait}s')
                for remaining in range(wait, 0, -10):
                    logger.info(f'  Resume in {remaining}s...')
                    await asyncio.sleep(min(10, remaining))
            else:
                logger.warning(f'LLM error (attempt {attempt + 1}/{MAX_RETRIES}): {e} — retrying in {wait}s')
                if attempt == MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(wait)
    return '{}'

def build_qa_pairs(docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"FAQ {i}:\nQUESTION: {doc['question']}\nANSWER: {doc['answer'][:400]}\n")
    return '\n'.join(parts)

def clean_json(raw: str) -> dict:
    text = re.sub('```(?:json)?|```', '', raw).strip()
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
                    return json.loads(text[start:i + 1])
                except Exception:
                    break
    try:
        return json.loads(text)
    except Exception:
        return {}

async def generate_batch(docs: list[dict], strategy_name: str, strategy_info: dict) -> list[dict]:
    template = strategy_info['template']
    temperature = strategy_info.get('temperature', 0.7)
    qa_pairs = build_qa_pairs(docs)
    prompt = template.format(qa_pairs=qa_pairs)
    try:
        t0 = time.monotonic()
        raw = await call_llm_async(prompt, temperature=temperature)
        elapsed = time.monotonic() - t0
        result = clean_json(raw)
        output = []
        for i, doc in enumerate(docs, 1):
            variations = result.get(str(i), [])
            if not isinstance(variations, list):
                variations = []
            count = len([v for v in variations if v])
            output.append({'original_question': doc['question'], 'expected_id': doc['id'], 'course': doc['course'], 'section': doc.get('section', ''), 'topic': doc.get('topic', -1), 'ner_category': doc.get('ner_category', 'OTHER'), 'ner_primary_entity': doc.get('ner_primary_entity'), 'prompt_strategy': strategy_name, 'variations': [v for v in variations[:3] if v]})
        total = sum((len(r['variations']) for r in output))
        logger.info(f'  [{strategy_name}] {total} queries in {elapsed:.1f}s')
        return output
    except Exception as e:
        logger.error(f'  [{strategy_name}] Batch failed: {e}')
        return [{'original_question': doc['question'], 'expected_id': doc['id'], 'course': doc['course'], 'section': doc.get('section', ''), 'topic': doc.get('topic', -1), 'ner_category': doc.get('ner_category', 'OTHER'), 'ner_primary_entity': doc.get('ner_primary_entity'), 'prompt_strategy': strategy_name, 'variations': []} for doc in docs]

def _load_existing(output_path: Path) -> set[str]:
    """Return set of (expected_id, strategy) already generated."""
    if not output_path.exists():
        return set()
    try:
        data = json.load(open(output_path))
        done = set()
        for q in data.get('queries', []):
            for strategy, variations in q.get('prompt_results', {}).items():
                if variations:
                    done.add(f"{q['expected_id']}::{strategy}")
        logger.info(f'Resuming — {len(done)} (doc, strategy) pairs already done')
        return done
    except Exception:
        return set()

def _save(queries: list[dict], output_path: Path, metadata: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_doc: dict[str, dict] = {}
    for r in queries:
        key = r['expected_id']
        if key not in by_doc:
            by_doc[key] = {'original_question': r['original_question'], 'expected_id': r['expected_id'], 'course': r['course'], 'section': r['section'], 'topic': r.get('topic', -1), 'ner_category': r.get('ner_category', 'OTHER'), 'ner_primary_entity': r.get('ner_primary_entity'), 'prompt_results': {}}
        by_doc[key]['prompt_results'][r['prompt_strategy']] = r['variations']
    total_queries = sum((len(v) for doc in by_doc.values() for v in doc['prompt_results'].values()))
    metadata['total_queries'] = total_queries
    metadata['total_documents'] = len(by_doc)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'metadata': metadata, 'queries': list(by_doc.values())}, f, indent=2)
    logger.info(f'Saved {total_queries} queries across {len(by_doc)} docs to {output_path}')

async def main(n_docs: int=DEFAULT_N_DOCS, batch_size: int=DEFAULT_BATCH_SIZE, output_path: Path=DEFAULT_OUTPUT, prompts_path: Path=DEFAULT_PROMPTS, seed: int=42) -> None:
    start = time.monotonic()
    _init_env()
    prompts = json.load(open(prompts_path))
    docs = load_documents_with_topics(DEFAULT_CLEAN, DEFAULT_TOPIC_ASSIGNMENTS)
    sampled = stratified_sample(docs, n_docs, seed=seed)
    total_batches = len(sampled) // batch_size
    total_calls = len(prompts) * total_batches
    logger.info(f'Generating queries: {len(sampled)} docs | batch_size={batch_size} | strategies={list(prompts.keys())} | ~{total_calls} LLM calls | ~{total_calls * (GAP_BETWEEN_CALLS + 5) / 60:.0f} min estimated')
    done = _load_existing(output_path)
    all_results = []
    errors = 0
    metadata = {'description': 'Stratified test queries for retrieval evaluation', 'model': JUDGE_LLM, 'generated_at': datetime.now().isoformat(), 'n_docs': n_docs, 'batch_size': batch_size, 'prompt_strategies': list(prompts.keys()), 'seed': seed, 'total_queries': 0, 'total_documents': 0, 'errors': 0}
    for batch_start in range(0, len(sampled) - batch_size + 1, batch_size):
        batch = sampled[batch_start:batch_start + batch_size]
        bn = batch_start // batch_size + 1
        courses = set((d['course'] for d in batch))
        logger.info(f'Batch {bn}/{total_batches} | courses: {courses}')
        for strategy_name, strategy_info in prompts.items():
            skip = all((f"{doc['id']}::{strategy_name}" in done for doc in batch))
            if skip:
                logger.info(f'  [{strategy_name}] already done — skipping')
                continue
            results = await generate_batch(batch, strategy_name, strategy_info)
            all_results.extend(results)
            if all((len(r['variations']) == 0 for r in results)):
                errors += 1
            await asyncio.sleep(GAP_BETWEEN_CALLS)
        metadata['errors'] = errors
        _save(all_results, output_path, metadata)
    elapsed = time.monotonic() - start
    metadata['generation_time_seconds'] = round(elapsed, 1)
    metadata['errors'] = errors
    _save(all_results, output_path, metadata)
    logger.info(f"Done — {metadata['total_queries']} queries | {metadata['total_documents']} docs | {errors} errors | {elapsed:.0f}s")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate stratified test queries')
    parser.add_argument('--n-docs', type=int, default=DEFAULT_N_DOCS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main(n_docs=args.n_docs, batch_size=args.batch_size, output_path=args.output, seed=args.seed))
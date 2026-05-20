"""
p01_generate_diverse_test_queries.py
=====================================
Generates a structured, multi-tiered evaluation dataset across three explicit zones:
1. global_health       : Stratified, mixed cross-course sampling (macro metrics)
2. cross_course_clash  : Hyper-focused on identical technical tools across different courses
3. intra_topic_battle  : Concentrated within high-density BERTopic clusters (micro separation)

Output: flat JSONL rows — one per query variation — matching _benchmark_loader.load_test_set()
exactly. Required fields per row: id, question, answer, course, expected_id, query_type, section.

Saves incrementally via append; crash-safe via signature cache rebuilt from existing rows.
"""
import argparse
import asyncio
import json
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from production_pipeline.p05_evaluation.p01_generate_test_queries import DEFAULT_CLEAN, DEFAULT_PROMPTS, DEFAULT_TOPIC_ASSIGNMENTS, GAP_BETWEEN_CALLS, JUDGE_LLM, _init_env, generate_batch, load_documents_with_topics, stratified_sample
from rag_pipeline.logging import get_logger
from rag_pipeline.core.paths import Paths
logger = get_logger(__name__)
DIVERSE_OUTPUT = Paths.processed_dir() / 'eval_queries_tiered.jsonl'

def sample_cross_course_clash(docs: list[dict], target_docs: int=40) -> list[dict]:
    """Selects docs matching highly shared universal infrastructure across different courses."""
    universal_tools = ['docker', 'git', 'python', 'aws', 'terraform', 'bash', 'linux']
    doc_texts = [(d, (d['question'] + ' ' + d['answer']).lower()) for d in docs]
    tool_groups: dict[str, list[dict]] = defaultdict(list)
    for d, text in doc_texts:
        for tool in universal_tools:
            if tool in text:
                tool_groups[tool].append(d)
    if not tool_groups:
        logger.warning('⚠️ No universal tool matches found in corpus')
        return []
    per_tool_target = max(1, target_docs // len(tool_groups))
    sampled: list[dict] = []
    for tool, group in tool_groups.items():
        size = min(len(group), per_tool_target)
        sampled.extend(random.sample(group, size))
    sampled = list({d['id']: d for d in sampled}.values())
    logger.info(f'🎯 [Tier: cross_course_clash] Sampled {len(sampled)} unique docs targeting universal tool chains.')
    return sampled

def sample_intra_topic_battle(docs: list[dict], target_docs: int=50) -> list[dict]:
    """Concentrates sampling within the highest-density BERTopic clusters."""
    topic_counts: dict[int, int] = defaultdict(int)
    for d in docs:
        if d.get('topic', -1) != -1:
            topic_counts[d['topic']] += 1
    dense_topics = [t for t, _ in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
    pool = [d for d in docs if d.get('topic', -1) in dense_topics]
    if len(pool) <= target_docs:
        sampled = pool
    else:
        sampled = random.sample(pool, target_docs)
    logger.info(f'⚔️ [Tier: intra_topic_battle] Sampled {len(sampled)} docs strictly inside dense clusters: {dense_topics}')
    return sampled

def _load_existing_cache(output_path: Path) -> set[str]:
    """
    Rebuilds completed-signature set from existing JSONL rows.

    Signature format: {expected_id}::{strategy}::{query_type}
    Derived from row id format:  query__{strategy}__{expected_id}_{variation_idx}
    """
    if not output_path.exists():
        return set()
    completed: set[str] = set()
    try:
        with open(output_path, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row_id = row.get('id', '')
                parts = row_id.split('__')
                if len(parts) != 3:
                    continue
                _, strategy, tail = parts
                expected_id = '_'.join(tail.split('_')[:-1])
                query_type = row.get('query_type', '')
                if expected_id and strategy and query_type:
                    completed.add(f'{expected_id}::{strategy}::{query_type}')
        logger.info(f'🔄 Cache hit: {len(completed)} completed query signatures found on disk.')
    except Exception as e:
        logger.warning(f'Failed to parse existing JSONL cache ({e}). Starting fresh.')
        return set()
    return completed

async def run_tier_generation(sampled_docs: list[dict], tier_name: str, prompts: dict, batch_size: int, done_cache: set[str], output_path: Path) -> None:
    """
    Generates query variations for one tier and appends flat JSONL rows to output_path.

    Each written row matches the schema expected by _benchmark_loader.load_test_set():
        id          -> query__{strategy}__{expected_id}_{variation_idx}
        question    -> generated query variation text
        answer      -> original source question (reference context)
        course      -> course identifier
        expected_id -> source document id for retrieval ground truth
        query_type  -> tier name (global_health | cross_course_clash | intra_topic_battle)
        section     -> section string (may be empty)
    """
    if not sampled_docs:
        return
    total_batches = (len(sampled_docs) + batch_size - 1) // batch_size
    for batch_start in range(0, len(sampled_docs), batch_size):
        batch = sampled_docs[batch_start:batch_start + batch_size]
        if not batch:
            continue
        bn = batch_start // batch_size + 1
        logger.info(f'🚀 [{tier_name.upper()}] Batch {bn}/{total_batches} | Processing {len(batch)} source docs')
        for strategy_name, strategy_info in prompts.items():
            needed = [d for d in batch if f"{d['id']}::{strategy_name}::{tier_name}" not in done_cache]
            if not needed:
                logger.debug(f'⏭️ [{tier_name}] {strategy_name} — all cached, skipping')
                continue
            results = await generate_batch(needed, strategy_name, strategy_info)
            with open(output_path, 'a', encoding='utf-8') as f:
                for r in results:
                    expected_id = r['expected_id']
                    for idx, query_text in enumerate(r.get('variations', []), 1):
                        if not query_text.strip():
                            continue
                        row = {'id': f'query__{strategy_name}__{expected_id}_{idx}', 'question': query_text, 'answer': r['original_question'], 'course': r['course'], 'expected_id': expected_id, 'query_type': tier_name, 'section': r.get('section', '')}
                        f.write(json.dumps(row, ensure_ascii=False) + '\n')
            for d in needed:
                done_cache.add(f"{d['id']}::{strategy_name}::{tier_name}")
            await asyncio.sleep(GAP_BETWEEN_CALLS)

async def main() -> None:
    parser = argparse.ArgumentParser(description='Generate Multi-Tiered Evaluation Datasets')
    parser.add_argument('--n-global', type=int, default=150)
    parser.add_argument('--n-clash', type=int, default=40)
    parser.add_argument('--n-battle', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--output', type=Path, default=DIVERSE_OUTPUT)
    args = parser.parse_args()
    start_time = time.monotonic()
    _init_env()
    prompts = json.load(open(DEFAULT_PROMPTS, encoding='utf-8'))
    all_docs = load_documents_with_topics(DEFAULT_CLEAN, DEFAULT_TOPIC_ASSIGNMENTS)
    done_cache = _load_existing_cache(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    random.shuffle(all_docs)
    global_docs = stratified_sample(all_docs, args.n_global)
    used_ids = {d['id'] for d in global_docs}
    remaining = [d for d in all_docs if d['id'] not in used_ids]
    clash_docs = sample_cross_course_clash(remaining, args.n_clash)
    clash_ids = {d['id'] for d in clash_docs}
    battle_pool = [d for d in remaining if d['id'] not in clash_ids]
    battle_docs = sample_intra_topic_battle(battle_pool, args.n_battle)
    logger.info(f'📊 Tier split | Global: {len(global_docs)} | Clash: {len(clash_docs)} | Battle: {len(battle_docs)}')
    await run_tier_generation(global_docs, 'global_health', prompts, args.batch_size, done_cache, args.output)
    await run_tier_generation(clash_docs, 'cross_course_clash', prompts, args.batch_size, done_cache, args.output)
    await run_tier_generation(battle_docs, 'intra_topic_battle', prompts, args.batch_size, done_cache, args.output)
    elapsed = time.monotonic() - start_time
    logger.info(f'✓ Done in {elapsed:.0f}s. Benchmark-ready dataset written to: {args.output}')
if __name__ == '__main__':
    asyncio.run(main())
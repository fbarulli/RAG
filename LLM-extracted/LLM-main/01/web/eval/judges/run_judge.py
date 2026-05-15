"""
eval/run_judge.py
=================
Evaluates retrieved contexts using an LLM judge.

Every single API call is a real judgment that gets saved to CSV.
No calls are wasted on probing - we start judging immediately with a 
conservative gap and adapt based on actual rate limit responses.

Progress is saved after every single result. On restart, previous runs for
the same subset are detected automatically and already-completed queries are
skipped — no work is ever repeated.

Runs on two slices by default:
  • vector_default failures  (k=5, success=False)
  • bm25_default successes   (k=5, success=True,  50-query sample)

Output: experiments/judge/<subset>_progress.csv   (live, appended per result)
        experiments/judge/<timestamp>_judge_results.csv  (final combined)

Usage:
    uv run eval/run_judge.py
    uv run eval/run_judge.py --model 8b --sample 100
    uv run eval/run_judge.py --config hybrid_default --success --model 70b
"""
import os
import csv
import glob
import asyncio
import argparse
from datetime import datetime

from shared import (
    llm_call, run_sequential,
    parse_verdicts, sanitize_query, extract_reasoning,
    load_results, sample_results,
    JUDGE_MODEL_8B, JUDGE_MODEL_70B, LIMITS_BY_MODEL,
)

OUTPUT_DIR = 'experiments/judge'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIELDNAMES = [
    'subset', 'query', 'expected_id', 'found_id',
    'judge_any_yes', 'judge_verdicts', 'judged_rank', 'reasoning',
    'raw_response', 'raw_len', 'verdict_count_raw',
]


# ── Prompt (no truncation, with note about truncated queries) ───────────────
def build_judge_prompt(query: str, contexts: list[str]) -> str:
    """
    Uses all contexts (up to 5) with full content, no truncation.
    Includes a note that questions may be truncated.
    """
    clean  = sanitize_query(query)
    n      = min(len(contexts), 5)
    
    prompt = f"Question: {clean}\n\nContexts:\n"
    for i, ctx in enumerate(contexts[:n], 1):
        prompt += f"<context_{i}>\n{ctx}\n</context_{i}>\n\n"
    
    prompt += (
        f"Task: Does each context directly answer the question?\n"
        f"- 'YES' = contains the factual answer to the question\n"
        f"- 'NO' = only discusses the topic, doesn't answer\n\n"
        f"Note: Some questions may be truncated. Judge based on the topic, not the exact wording.\n\n"
        f"Examples:\n"
        f"Q: How do I install Docker on Ubuntu?\n"
        f"C1: To install Docker on Ubuntu, run: sudo apt-get update && sudo apt-get install docker.io\n"
        f"C2: Docker is a platform for developing and running applications in containers.\n"
        f"Answer: ['YES', 'NO']\n\n"
        f"Q: What is the course schedule?\n"
        f"C1: The course covers machine learning topics including supervised and unsupervised learning.\n"
        f"C2: You can find the syllabus on the course website under the Materials section.\n"
        f"Answer: ['NO', 'NO']\n\n"
        f"Now evaluate the {n} contexts above.\n"
        f"For each context, write ONE short sentence explaining why.\n"
        f"Then on the final line, output ONLY a JSON array like: ['YES', 'NO', 'NO']\n"
        f"Answer:"
    )
    return prompt


# ── Progress tracker (unchanged) ───────────────────────────────────────────
class ProgressTracker:
    def __init__(self, subset_name: str):
        self.subset_name = subset_name
        self.path        = f'{OUTPUT_DIR}/{subset_name}_progress.csv'
        self.done        = {}

        # Load previous progress (same as original, unchanged)
        if os.path.exists(self.path):
            with open(self.path, newline='') as f:
                for row in csv.DictReader(f):
                    self.done[row['query']] = row
        # Also check older timestamped files
        pattern = f'{OUTPUT_DIR}/*_judge_results.csv'
        for fpath in sorted(glob.glob(pattern)):
            with open(fpath, newline='') as f:
                for row in csv.DictReader(f):
                    if row.get('subset') == subset_name:
                        if row['query'] not in self.done:
                            self.done[row['query']] = row

        write_header = not os.path.exists(self.path)
        self._fh     = open(self.path, 'a', newline='')
        self._writer = csv.DictWriter(self._fh, fieldnames=FIELDNAMES)
        if write_header:
            self._writer.writeheader()
            self._fh.flush()

    def is_done(self, query: str) -> bool:
        return query in self.done

    def completed_rows(self) -> list[dict]:
        return list(self.done.values())

    def save(self, row: dict):
        self.done[row['query']] = row
        self._writer.writerow(row)
        self._fh.flush()

    def close(self):
        self._fh.close()


# ── Single judge call (with reasoning and rank) ─────────────────────────────
async def judge_item(item: dict, idx: int,
                     model: str) -> dict:
    if not item['contexts']:
        return {
            'query':             item['query'],
            'expected_id':       item['expected_id'],
            'found_id':          item['found_id'],
            'judge_any_yes':     False,
            'judge_verdicts':    '[]',
            'judged_rank':       None,
            'reasoning':         '',
            'raw_response':      'NO_CONTEXT',
            'raw_len':           0,
            'verdict_count_raw': 0,
        }

    prompt = build_judge_prompt(item['query'], item['contexts'])
    raw = await llm_call(prompt, model=model, max_tokens=2000)

    n         = min(len(item['contexts']), 5)
    verdicts  = parse_verdicts(raw, n, query_preview=item['query'])
    any_yes   = any(v == 'YES' for v in verdicts)
    first_yes = next((i+1 for i, v in enumerate(verdicts) if v == 'YES'), None)
    reasoning = extract_reasoning(raw)

    raw_count = len([x for x in raw.upper().split() if 'YES' in x or 'NO' in x])

    return {
        'query':             item['query'],
        'expected_id':       item['expected_id'],
        'found_id':          item['found_id'],
        'judge_any_yes':     any_yes,
        'judge_verdicts':    str(verdicts),
        'judged_rank':       first_yes,
        'reasoning':         reasoning[:500],
        'raw_response':      raw[:500],
        'raw_len':           len(raw),
        'verdict_count_raw': raw_count,
        '_rate_limited':     False,
    }


# ── Main (with fixed error preservation in run_sequential) ─────────────────
async def main(args):
    model     = JUDGE_MODEL_8B if args.model == '8b' else JUDGE_MODEL_70B
    limits    = LIMITS_BY_MODEL.get(model, {})
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(
        f"Model : {model}\n"
        f"  Starting gap : {limits.get('gap')}s  "
        f"RPM cap : {limits.get('rpm')}  "
        f"TPM cap : {limits.get('tpm')}"
    )

    # ── Load slices ──────────────────────────────────────────────────────────
    if args.config:
        items   = sample_results(args.config, k=5, success=args.success, n=args.sample)
        subsets = [(args.config + ('_success' if args.success else '_failure'), items)]
    else:
        failures  = load_results('vector_default', k=5, success=False)
        successes = sample_results('bm25_default',  k=5, success=True, n=args.sample)
        subsets   = [('vector_failure', failures), ('bm25_success', successes)]

    # ── Start judging immediately (no probing) ───────────────────────────────
    start_gap = limits.get('gap', 12.0)
    all_rows = []
    
    for subset_name, items in subsets:
        tracker = ProgressTracker(subset_name)

        pending  = [it for it in items if not tracker.is_done(it['query'])]
        skipped  = len(items) - len(pending)
        n_total  = len(items)
        eta_min  = (len(pending) * start_gap) / 60.0

        print(f"\nJudging {n_total} items  [{subset_name}]  "
              f"starting gap={start_gap}s  (~{eta_min:.1f} min)")
        if skipped:
            print(f"  [resume] skipping {skipped} already-completed queries, "
                  f"{len(pending)} remaining")

        prev_results = tracker.completed_rows()

        if pending:
            # Wrap judge_item to save progress immediately and preserve identity on error
            async def judge_and_save(item, idx, tracker=tracker):
                try:
                    result = await judge_item(item, idx, model)
                except Exception as e:
                    result = {
                        'query':             item['query'],
                        'expected_id':       item['expected_id'],
                        'found_id':          item['found_id'],
                        'judge_any_yes':     False,
                        'judge_verdicts':    '[]',
                        'judged_rank':       None,
                        'reasoning':         f'ERROR: {type(e).__name__}',
                        'raw_response':      f'ERROR: {type(e).__name__}',
                        'raw_len':           0,
                        'verdict_count_raw': 0,
                        '_rate_limited':     True if '429' in str(e) else False,
                    }
                row = {**result, 'subset': subset_name}
                # Remove internal field not meant for CSV
                if '_rate_limited' in row:
                    del row['_rate_limited']
                tracker.save(row)
                return result

            done_so_far = skipped
            def progress(done, total, _base=skipped, _total=n_total):
                print(f"  [{_base + done}/{_total}]", end='\r', flush=True)

            coros = [
                judge_and_save(item, skipped + i + 1)
                for i, item in enumerate(pending)
            ]
            new_results = await run_sequential(coros, gap=start_gap, progress_cb=progress, model=model)
            print()  # newline after \r
        else:
            new_results = []
            print("  [resume] all queries already complete — nothing to run")

        tracker.close()

        all_subset_rows = prev_results + new_results
        all_rows.extend(all_subset_rows)

        yes   = sum(1 for r in all_subset_rows if str(r.get('judge_any_yes')).lower() in ('true', '1', 'yes'))
        total = len(all_subset_rows)
        pct   = yes / total if total else 0
        label = 'judged precision@5' if 'success' in subset_name else 'any context answers?'
        print(f"  {label}: {yes}/{total} = {pct:.1%}")

        over = [r for r in all_subset_rows if int(r.get('verdict_count_raw', 0)) > 5]
        if over:
            print(f"  ⚠️  {len(over)} rows with extra verdicts")

    # ── Save final combined CSV ───────────────────────────────────────────────
    csv_path = f'{OUTPUT_DIR}/{timestamp}_judge_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in all_rows:
            # Ensure all keys exist
            out_row = {k: row.get(k, '') for k in FIELDNAMES}
            writer.writerow(out_row)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    for subset_name, _ in subsets:
        rows  = [r for r in all_rows if r.get('subset') == subset_name]
        yes   = sum(1 for r in rows if str(r.get('judge_any_yes')).lower() in ('true', '1', 'yes'))
        total = len(rows)
        pct   = yes / total if total else 0
        label = 'judged precision@5' if 'success' in subset_name else 'any context answers?'
        print(f"{subset_name}  →  {label}: {yes}/{total} = {pct:.1%}")

    total_judged = len(all_rows)
    print(f"\nTotal queries judged: {total_judged}")
    print(f"Progress files : {OUTPUT_DIR}/<subset>_progress.csv")
    print(f"Final combined : {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   choices=['8b', '70b'], default='70b')
    parser.add_argument('--sample',  type=int, default=50,
                        help='Max rows to sample (default 50)')
    parser.add_argument('--config',  type=str, default=None,
                        help='Single config name instead of default two-slice run')
    parser.add_argument('--success', action='store_true',
                        help='With --config: load successes instead of failures')
    args = parser.parse_args()
    asyncio.run(main(args))
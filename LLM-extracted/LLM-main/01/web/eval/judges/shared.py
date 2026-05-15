"""
eval/shared.py  shared utilities for all judge tools
"""
import os
import re
import json
import time
import random
import asyncio
from dotenv import load_dotenv
from litellm import acompletion

load_dotenv('/home/admin/LLM/LLM/01/web/configs/.env')
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")

# ── Models ────────────────────────────────────────────────────────────────────
JUDGE_MODEL_8B  = "nvidia_nim/meta/llama-3.1-8b-instruct"
JUDGE_MODEL_70B = "nvidia_nim/meta/llama-3.1-70b-instruct"
DEFAULT_JUDGE   = JUDGE_MODEL_70B

# Fallback limits used as starting point, then adjusted dynamically
LIMITS_BY_MODEL = {
    JUDGE_MODEL_70B: dict(rpm=10,  tpm=4_000,  gap=3.0),
    JUDGE_MODEL_8B:  dict(rpm=30,  tpm=15_000, gap=3.0),
}

PROBE_SAFETY = 0.80
_RATE_WINDOW = 60.0


# ── Token estimator ───────────────────────────────────────────────────────────
def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.5))


def _usage(response) -> tuple[int, int]:
    usage = getattr(response, 'usage', None)
    if usage and getattr(usage, 'prompt_tokens', None):
        return usage.prompt_tokens, (usage.completion_tokens or 1)
    content = response.choices[0].message.content.strip()
    return _estimate_tokens("Reply with the single word OK."), _estimate_tokens(content)


# ── Adaptive sequential runner ────────────────────────────────────────────────
async def run_sequential(coros: list, gap: float,
                         progress_cb=None,
                         model: str = DEFAULT_JUDGE) -> list:
    from litellm.exceptions import RateLimitError, Timeout
    
    results = []
    total   = len(coros)
    ratelimit_count = 0
    timeout_count = 0
    
    for i, coro in enumerate(coros):
        t0 = time.monotonic()
        try:
            result = await coro
            status = "✓"
        except RateLimitError:
            ratelimit_count += 1
            status = "✗ 429"
            result = {
                'query':             'unknown',   # Placeholder – will be overwritten by caller
                'expected_id':       '',
                'found_id':          '',
                'judge_any_yes':     False,
                'judge_verdicts':    '[]',
                'judged_rank':       None,
                'reasoning':         '',
                'raw_response':      'RATE_LIMITED',
                'raw_len':           0,
                'verdict_count_raw': 0,
                '_rate_limited':     True,
            }
        except Timeout:
            timeout_count += 1
            status = "✗ 504"
            result = {
                'query':             'unknown',
                'expected_id':       '',
                'found_id':          '',
                'judge_any_yes':     False,
                'judge_verdicts':    '[]',
                'judged_rank':       None,
                'reasoning':         '',
                'raw_response':      'TIMEOUT',
                'raw_len':           0,
                'verdict_count_raw': 0,
                '_rate_limited':     False,
            }
        
        results.append(result)
        elapsed = time.monotonic() - t0
        print(f"  [{i+1}/{total}] {status} gap={gap}s elapsed={elapsed:.1f}s")
        
        if progress_cb:
            progress_cb(i + 1, total)
        
        if '✗' in status:
            print(f"  [{status.strip()}] waiting 60s — resuming after")
            await asyncio.sleep(60)
            print(f"  [{status.strip()}] wait complete — continuing")
        
        wait = max(0.0, gap - elapsed)
        if wait > 0 and i < total - 1:
            await asyncio.sleep(wait)
    
    if ratelimit_count or timeout_count:
        print(f"\n  [stats] 429s: {ratelimit_count}  504s: {timeout_count}  "
              f"total wait: {(ratelimit_count + timeout_count) * 60}s")
    
    return results


# ── LLM call wrapper (no truncation, extended max_tokens) ─────────────────────
async def llm_call(prompt: str,
                   model: str = DEFAULT_JUDGE,
                   max_tokens: int = 2000) -> str:
    """
    Single LLM call. No retry logic - rate limits and timeouts
    are handled by run_sequential which controls pacing and waits.
    """
    response = await acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ── Verdict helpers ───────────────────────────────────────────────────────────
def parse_verdicts(raw: str, n: int, query_preview: str = "") -> list[str]:
    match = re.search(r'\[(.*?)\]', raw, re.DOTALL)
    if match:
        items  = [i.strip().strip('\'"') for i in match.group(1).split(',')]
        parsed = [i.upper() if i.upper() in ('YES', 'NO') else 'NO' for i in items]
    else:
        tokens = re.findall(r'\b(YES|NO)\b', raw, re.IGNORECASE)
        parsed = [t.upper() for t in tokens]

    if len(parsed) != n and query_preview:
        print(f"  [WARN] {len(parsed)} verdicts for {n} contexts: "
              f"{query_preview[:60]}")

    return (parsed + ['NO'] * n)[:n]


def extract_reasoning(raw: str) -> str:
    """
    Extract the explanation text before the final verdict list.
    Assumes the last line contains the JSON array.
    """
    # Find the last occurrence of a JSON list pattern
    match = re.search(r'\[(.*?)\]', raw, re.DOTALL)
    if match:
        reasoning = raw[:match.start()].strip()
        return reasoning
    return ""


def parse_json_response(raw: str) -> dict | None:
    clean = re.sub(r'```(?:json)?|```', '', raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return None


# ── Text helpers (no query truncation) ───────────────────────────────────────
def sanitize_query(query: str, max_len: int = 10000) -> str:
    """
    Remove leading punctuation but do NOT truncate the query.
    """
    q = query.lstrip("'\"': ")
    return q


# ── Result loaders (unchanged) ────────────────────────────────────────────────
RESULTS_DIR = 'experiments/results'

def load_results(config_name: str, k: int = 5, success: bool = True) -> list[dict]:
    path = f'{RESULTS_DIR}/{config_name}.json'
    with open(path) as f:
        data = json.load(f)
    seen, out = set(), []
    for r in data['results']:
        if r.get('k') == k and r.get('success') == success:
            q = r['query']
            if q not in seen:
                seen.add(q)
                out.append({
                    'query':       q,
                    'expected_id': r['expected_id'],
                    'found_id':    r.get('found_id', 'NONE'),
                    'contexts':    r.get('contexts', []),
                })
    return out


def sample_results(config_name: str, k: int = 5, success: bool = True,
                   n: int = 50, seed: int = 42) -> list[dict]:
    rows = load_results(config_name, k=k, success=success)
    if len(rows) > n:
        random.seed(seed)
        rows = random.sample(rows, n)
    return rows
"""
helpers/nvidia_call.py
=======================
Centralized NVIDIA NIM API calls with rate limiting, retries, and logging.
Thread-safe RateGate for concurrent async calls.

max_tokens is REQUIRED — no default that silently truncates.
Every call site must think about response length.

Usage:
    from helpers.nvidia_call import call_llm, call_llm_async, RateGate, LLMResult
"""
import os, time, asyncio, logging
from dataclasses import dataclass
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "nvidia_nim/meta/llama-3.1-70b-instruct"
MAX_RETRIES = 3
RATE_LIMIT_WAIT = 60


def init():
    """Load API keys. Safe to call multiple times."""
    load_dotenv('configs/.env')
    os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")


@dataclass
class LLMResult:
    content: str
    latency_ms: float
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _is_rate_limit(exc: Exception) -> bool:
    """Check if exception is a rate limit or service unavailable error."""
    try:
        from litellm.exceptions import RateLimitError, ServiceUnavailableError
        if isinstance(exc, (RateLimitError, ServiceUnavailableError)):
            return True
    except ImportError:
        pass
    msg = str(exc)
    return any(code in msg for code in ['429', '502', '504', 'RateLimitError'])


def _backoff(attempt: int, is_rate_limit: bool) -> float:
    return RATE_LIMIT_WAIT * (attempt + 1) if is_rate_limit else 5.0


class RateGate:
    """Async-safe rate-limit gate for concurrent calls within a single event loop."""
    def __init__(self):
        self._limited_until = 0
        self._lock = asyncio.Lock()

    async def wait_if_needed(self):
        async with self._lock:
            now = time.monotonic()
            wait = self._limited_until - now
            if wait > 0:
                logger.warning(f"Rate gate: waiting {wait:.0f}s...")
                await asyncio.sleep(wait)

    async def hit_limit(self):
        async with self._lock:
            self._limited_until = max(self._limited_until, time.monotonic() + RATE_LIMIT_WAIT)
            logger.warning(f"Rate limit hit — gate closed for {RATE_LIMIT_WAIT}s")


def call_llm(prompt: str, max_tokens: int, model: str = DEFAULT_MODEL,
             temperature: float = 0.3) -> LLMResult:
    """Synchronous LLM call with retry. max_tokens is REQUIRED. Raises on failure."""
    from litellm import completion
    init()
    
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.monotonic()
            resp = completion(model=model, messages=[{"role":"user","content":prompt}],
                            temperature=temperature, max_tokens=max_tokens)
            elapsed = (time.monotonic() - t0) * 1000
            content = resp.choices[0].message.content
            if content is None:
                content = ""
            usage = getattr(resp, 'usage', None)
            return LLMResult(
                content=content.strip(),
                latency_ms=elapsed,
                model=model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            )
        except Exception as e:
            is_rl = _is_rate_limit(e)
            wait = _backoff(attempt, is_rl)
            logger.warning(f"LLM error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(wait)


async def call_llm_async(prompt: str, max_tokens: int, model: str = DEFAULT_MODEL,
                         temperature: float = 0.3, rate_gate: RateGate = None) -> LLMResult:
    """Async LLM call with retry and optional shared rate gate. max_tokens is REQUIRED. Raises on failure."""
    from litellm import acompletion
    init()
    
    for attempt in range(MAX_RETRIES):
        if rate_gate:
            await rate_gate.wait_if_needed()
        try:
            t0 = time.monotonic()
            resp = await acompletion(model=model, messages=[{"role":"user","content":prompt}],
                                     temperature=temperature, max_tokens=max_tokens)
            elapsed = (time.monotonic() - t0) * 1000
            content = resp.choices[0].message.content
            if content is None:
                content = ""
            usage = getattr(resp, 'usage', None)
            return LLMResult(
                content=content.strip(),
                latency_ms=elapsed,
                model=model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            )
        except Exception as e:
            is_rl = _is_rate_limit(e)
            if is_rl and rate_gate:
                await rate_gate.hit_limit()
                await rate_gate.wait_if_needed()
            else:
                wait = _backoff(attempt, is_rl)
                logger.warning(f"LLM error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(wait)
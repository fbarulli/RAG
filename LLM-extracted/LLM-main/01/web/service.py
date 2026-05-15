"""
Two-stage FAQ Service: CAG (fast cache) → RAG (LLM fallback)

Behavior:
1. Query → glossary expansion → async embed → async Qdrant search (top 50)
2. Course boost (×1.2) → re-rank → top 5
3. Guardrail: raw top-1 score < 0.67 → "I don't know"
4. CAG: top-1 score ≥ 0.75 and cached → return pre-computed answer
5. RAG: retrieve top-5 contexts (uses CAG answers when available) → generate with 70B

Fully async — embedding runs in thread pool, Qdrant via AsyncQdrantClient.
CAG is pre-computed offline. No runtime cache growth.
Guardrail validated against OOD (0.41-0.59), adversarial (0.61-0.65), FAQ (0.70+).
expansion_hits protected by asyncio.Lock (thread-safe under concurrency).

Run:    uv run python service.py
Test:   curl -X POST http://localhost:7870/ask -H 'Content-Type: application/json' -d '{"question":"how do I install Docker?"}'
"""
import sys, os, json, asyncio
from pathlib import Path
from collections import Counter

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from dotenv import load_dotenv
load_dotenv(BASE / 'configs/.env')
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from litellm import acompletion

# ── Config ───────────────────────────────────────────────────────────────────
MODEL                  = "nvidia_nim/meta/llama-3.1-70b-instruct"
EMBEDDING              = 'BAAI/bge-base-en-v1.5'
COLLECTION             = 'faqs_bge_base_en_v1.5'
CAG_PATH               = BASE / 'experiments/cag_answers_v2.json'
CAG_THRESHOLD          = 0.75
LOW_CONFIDENCE_THRESHOLD = 0.67   # Validated: OOD 0.41-0.59, adversarial 0.61-0.65, FAQ 0.70+
TOP_K                  = 5
OVERFETCH              = 50
CONTEXT_CHAR_LIMIT     = 1500
RAG_MAX_TOKENS         = 1024     # Matches CAG generation quality (300 tokens → poor FC)

GLOSSARY = {
    'embedding':         ['turn words into numbers', 'word vectors', 'text to numbers'],
    'encoding error':    ['csv weird bytes', 'weird characters'],
    'wget':              ['downloader thing', 'download tool', 'install downloader'],
    'correlation matrix': ['correlating numerical and categorical'],
    'multicollinearity': ['prevent multicollinearity'],
    'docker build cache': ['docker model not updating', 'docker same result'],
}

# ── Load once ────────────────────────────────────────────────────────────────
print("Loading...")
embed_model    = SentenceTransformer(EMBEDDING)
client         = AsyncQdrantClient('localhost', port=6333)
expansion_hits = Counter()
expansion_lock = asyncio.Lock()

def load_cag() -> dict:
    if CAG_PATH.exists():
        with open(CAG_PATH) as f:
            return json.load(f)['answers']
    return {}

cag = load_cag()
print(f"Loaded {len(cag)} CAG answers")

RAG_PROMPT = """You are a course teaching assistant. Answer the question using ONLY the FAQ contexts below.
If the contexts don't contain enough information, say exactly: "I don't have enough information to answer this question."

Question: {question}

Contexts:
{contexts}

Answer:"""

# ── Helpers ──────────────────────────────────────────────────────────────────
async def expand_query(query: str) -> str:
    """Glossary expansion with thread-safe hit tracking."""
    query_lower = query.lower()
    expansions  = []
    for technical_term, colloquial_terms in GLOSSARY.items():
        if any(phrase in query_lower for phrase in colloquial_terms):
            expansions.append(technical_term)
    if expansions:
        async with expansion_lock:
            for term in expansions:
                expansion_hits[term] += 1
        return query + ' ' + ' '.join(expansions)
    return query

def compute_boosted_score(hit, course: str = None) -> float:
    score = hit.score
    if course and hit.payload.get('course') == course:
        score *= 1.2
    return score

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="FAQ Service")

class Query(BaseModel):
    question: str
    course:   str = None

class Response(BaseModel):
    answer:          str
    stage:           str
    reason:          str   = None
    retrieval_score: float = None
    sources:         list  = []

@app.post("/ask", response_model=Response)
async def ask(q: Query):
    expanded = await expand_query(q.question)

    # Embed in thread pool — SentenceTransformer is synchronous/CPU-bound
    vec = (await asyncio.to_thread(embed_model.encode, expanded)).tolist()

    results = await client.query_points(
        collection_name=COLLECTION, query=vec, limit=OVERFETCH, with_payload=True,
    )

    if not results.points:
        return Response(
            answer="I couldn't find any relevant answers. Try rephrasing.",
            stage="none", reason="no_results", retrieval_score=0.0,
        )

    # Re-rank by boosted score, keep top K
    results.points.sort(key=lambda h: compute_boosted_score(h, q.course), reverse=True)
    results.points = results.points[:TOP_K]

    top_hit   = results.points[0]
    top_score = compute_boosted_score(top_hit, q.course)
    top_id    = top_hit.payload.get('es_id', '')

    # Guardrail on raw score (before course boost) to preserve validated thresholds
    raw_score = top_hit.score
    if raw_score < LOW_CONFIDENCE_THRESHOLD:
        return Response(
            answer="I don't have enough information to answer this question. "
                   "Try asking in a course-specific channel or rephrasing your question.",
            stage="none", reason="low_confidence", retrieval_score=raw_score,
        )

    # ── Stage 1: CAG ─────────────────────────────────────────────────────────
    if top_id in cag and top_score >= CAG_THRESHOLD:
        cached = cag[top_id]
        return Response(
            answer=cached['generated_answer'],
            stage="cag",
            reason="cached",
            retrieval_score=top_score,
            sources=[{
                'question': top_hit.payload.get('question', ''),
                'course':   top_hit.payload.get('course', ''),
                'score':    top_score,
            }],
        )

    # ── Stage 2: RAG ─────────────────────────────────────────────────────────
    # Prefer CAG-generated answers as context (better FC than raw FAQ text)
    contexts, sources = [], []
    for hit in results.points:
        p       = hit.payload
        hit_id  = p.get('es_id', '')
        answer_text = (
            cag.get(hit_id, {}).get('generated_answer')
            or p.get('answer', '')
        )
        contexts.append(f"Q: {p.get('question', '')}\nA: {answer_text[:CONTEXT_CHAR_LIMIT]}")
        sources.append({
            'question': p.get('question', ''),
            'course':   p.get('course', ''),
            'score':    compute_boosted_score(hit, q.course),
        })

    prompt = RAG_PROMPT.format(question=q.question, contexts="\n\n".join(contexts))

    try:
        resp = await acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=RAG_MAX_TOKENS,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"RAG error: {e}")
        answer = "Error generating answer. Please try again."

    insufficient = "don't have enough" in answer.lower()
    return Response(
        answer=answer,
        stage="rag",
        reason="insufficient_context" if insufficient else "generated",
        retrieval_score=top_score,
        sources=sources,
    )

# ── Monitoring ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    try:
        await client.count(collection_name=COLLECTION)
        qdrant_ok = True
    except Exception:
        qdrant_ok = False
    return {
        "status":     "ok" if qdrant_ok else "degraded",
        "qdrant":     qdrant_ok,
        "cag_loaded": len(cag) > 0,
        "cag_count":  len(cag),
    }

@app.get("/stats")
async def stats():
    async with expansion_lock:
        hits = dict(expansion_hits)
    return {
        "cag_size":       len(cag),
        "expansion_hits": hits,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7870)
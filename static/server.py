"""
RAG-a-muffin FastAPI server.
Usage: uv run uvicorn static.server:app --reload
"""
import json
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag_pipeline.core.paths import Paths
from rag_pipeline.logging import get_logger
from rag_pipeline.answer_generation.runner import QueryRetriever
from rag_pipeline.answer_generation.retriever import ContextRetriever
from rag_pipeline.answer_generation.generator import AnswerGenerator
from rag_pipeline.answer_generation.config import GenerationConfig, PROMPT_CONFIGS

logger = get_logger(__name__)
app = FastAPI(title="RAG-a-muffin")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── load defaults ──────────────────────────────────────────────────
with open(Paths.base() / "configs" / "defaults.json") as f:
    _defaults = json.load(f)

_model      = _defaults["production_model"]
_llm_model  = _defaults["static_llm_model"]
_qdrant     = _defaults["qdrant"]

# ── lazy singletons ────────────────────────────────────────────────
_retriever: Optional[QueryRetriever]    = None
_ctx:       Optional[ContextRetriever]  = None
_generator: Optional[AnswerGenerator]   = None

def get_retriever() -> QueryRetriever:
    global _retriever
    if _retriever is None:
        logger.info(f"Loading retriever: {_model}")
        _retriever = QueryRetriever(model_name=_model)
    return _retriever

def get_context_retriever() -> ContextRetriever:
    global _ctx
    if _ctx is None:
        _ctx = ContextRetriever(
            host=_qdrant["host"],
            port=_qdrant["port"],
            model_name=_model,
        )
    return _ctx

def get_generator(prompt_style: str) -> AnswerGenerator:
    global _generator
    if _generator is None:
        config = GenerationConfig(llm_model=_llm_model, prompt_style=prompt_style)
        _generator = AnswerGenerator.from_config(config)
    return _generator


# ── request / response ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    course: Optional[str] = None
    prompt_style: str = "strict"
    top_k: int = 3
    rerank: bool = True

class QueryResponse(BaseModel):
    answer: str
    doc_ids: list[str]
    context: str
    latency_ms: int


# ── endpoint ───────────────────────────────────────────────────────
@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if req.prompt_style not in PROMPT_CONFIGS:
        raise HTTPException(400, f"Unknown prompt_style. Choose from: {list(PROMPT_CONFIGS)}")
    if not 1 <= req.top_k <= 10:
        raise HTTPException(400, "top_k must be between 1 and 10")

    t0 = time.perf_counter()

    retriever = get_retriever()
    ctx       = get_context_retriever()
    generator = get_generator(req.prompt_style)

    pool = 10 if req.rerank else req.top_k
    candidate_ids = retriever.retrieve(req.query, top_k=pool, course=req.course)

    if req.rerank and len(candidate_ids) > 1:
        final_ids = ctx.rerank(req.query, candidate_ids, req.top_k)
    else:
        final_ids = candidate_ids[: req.top_k]

    context_text = ctx.get_context(final_ids)
    result = generator.generate(
        query_id="web",
        query=req.query,
        context_doc_ids=final_ids,
        context_text=context_text,
        prompt_style=req.prompt_style,
    )

    if not result.success:
        raise HTTPException(500, result.error or "Generation failed")

    latency_ms = int((time.perf_counter() - t0) * 1000)
    return QueryResponse(answer=result.generated_answer, doc_ids=final_ids, context=context_text, latency_ms=latency_ms)


@app.get("/health")
async def health():
    return {"status": "ok", "model": _model}


# ── serve frontend ─────────────────────────────────────────────────
from fastapi.responses import FileResponse
_static = Paths.base() / "static"

@app.get("/")
async def index():
    return FileResponse(str(_static / "index.html"))
# ── embeddings / UMAP ──────────────────────────────────────────────
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=1)
def get_umap_projection():
    """Scroll all vectors from Qdrant, fit UMAP, return (reducer, points)."""
    import umap
    client = get_context_retriever().client
    collection = Paths.collection_for_model(_model)

    vectors, ids, courses = [], [], []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=100, offset=offset,
            with_payload=True, with_vectors=True
        )
        for p in points:
            vectors.append(p.vector)
            ids.append(p.payload.get('es_id', str(p.id)))
            courses.append(p.payload.get('course', 'unknown'))
        if offset is None:
            break

    X = np.array(vectors)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(X)
    return reducer, coords, ids, courses

@app.get("/api/embeddings")
async def embeddings():
    _, coords, ids, courses = get_umap_projection()
    return {
        "points": [
            {"x": float(coords[i,0]), "y": float(coords[i,1]), "id": ids[i], "course": courses[i]}
            for i in range(len(ids))
        ]
    }

@app.post("/api/embed-query")
async def embed_query(req: QueryRequest):
    from sentence_transformers import SentenceTransformer
    reducer, _, _, _ = get_umap_projection()
    model = SentenceTransformer(_model)
    vec = model.encode(req.query)
    coord = reducer.transform([vec])[0]
    return {"x": float(coord[0]), "y": float(coord[1])}

# RAG Service Module — Design Plan

## What I've Observed

### Core Infrastructure (must always use)
| Class / Function | Location | Purpose |
|---|---|---|
| `Paths` | `core/paths.py` | All file paths — reads `configs/paths.json` |
| `get_logger(__name__)` | `rag_pipeline/logging.py` | Centralised logging |
| `FAQDocument` | `core/schemas.py` | Frozen dataclass for FAQ docs |
| `SearchResult` | `ingestion/benchmark_types.py` | Frozen dataclass returned by all retrievers |
| `GeneratedAnswer` | `p06/generator.py` | Typed LLM output |
| `EvaluationResult` | `p06/evaluator.py` | Typed judge output |
| `PipelineResult` | `p06/runner.py` | Typed end-to-end result |

### Winning Configuration
| Setting | Value | Source |
|---|---|---|
| Retrieval model | `BAAI/bge-base-en-v1.5` | `models.json` `_winner` + `defaults.json` `production_model` |
| Qdrant collection | `faqs_bge_base_en_v1_5` | derived from model name |
| Retrieval strategy | `entity_boosted` | `defaults.json` `production_config` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `ContextRetriever` default |
| LLM | `nvidia_nim/meta/llama-3.1-70b-instruct` | `defaults.json` `llm_model` |
| Prompt style | `strict` | only style in `retrieval_prompts.json` |
| top_k | 3 | best from generation sweep |
| rerank_pool | 10 | `runner.py` `RERANK_POOL` constant |

### Known Gaps (confirmed from code)
1. `Paths.get()` — called in `config.py` but **does not exist** on `Paths`
2. `Paths.configs_dir()` — called in `config.py` but **does not exist** on `Paths`
3. `configs/paths.json` has no `configs_dir` key
4. `runner.py` / `QueryRetriever` reads `defaults.json` inline — bypasses `Paths`
5. No `service.json` exists
6. No service module exists — pipeline is eval-only, not request/response ready
7. NER (`ner_category`, `ner_primary_entity`) not extracted for live queries — soft boost means `None` is a safe graceful fallback

---

## What We Do NOT Change
- `retrievers.py` — working, untouched
- `generator.py` / `evaluator.py` — working, untouched  
- `runner.py` — eval pipeline stays separate, untouched
- `retrieval_configs.json` — untouched
- `benchmark_types.py` — untouched

---

## Plan

### Step 1 — Patch `paths.json` + `Paths`
Add `"configs_dir": "configs"` to `paths.json`.  
Add two methods to `Paths`:
```python
@classmethod
def configs_dir(cls) -> Path:
    return cls._resolve("configs_dir")

@classmethod
def service_config(cls) -> Path:
    return cls.configs_dir() / "service.json"
```
**Test**: import `Paths`, assert both paths resolve without error.

---

### Step 2 — `configs/service.json` (new)
Single source of truth for the service. All values come from existing configs — no duplication.

```json
{
  "_comment": "Production service configuration. Locks in the winning retrieval + generation setup.",
  "retrieval_model": "BAAI/bge-base-en-v1.5",
  "collection": "faqs_bge_base_en_v1_5",
  "retrieval_config": "entity_boosted",
  "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "rerank": true,
  "rerank_pool": 10,
  "top_k": 3,
  "prompt_style": "strict",
  "llm_model": "nvidia_nim/meta/llama-3.1-70b-instruct",
  "qdrant_host": "localhost",
  "qdrant_port": 6333
}
```

---

### Step 3 — New module `src/rag_pipeline/service/`

```
service/
  __init__.py     # exports RAGService, ServiceConfig, ServiceRequest, ServiceResult
  config.py       # ServiceConfig dataclass — loads service.json via Paths
  schema.py       # ServiceRequest / ServiceResult frozen dataclasses
  pipeline.py     # RAGService — initializes once, exposes .answer()
```

#### `schema.py`
```python
@dataclass(frozen=True)
class ServiceRequest:
    query: str
    course: Optional[str] = None  # used as Qdrant course_filter

@dataclass(frozen=True)
class ServiceResult:
    query: str
    answer: str
    doc_ids: tuple[str, ...]
    latency_ms: float           # total wall time
    retrieval_latency_ms: float
    generation_latency_ms: float
    success: bool
    error: Optional[str] = None
```

#### `config.py`
```python
@dataclass(frozen=True)
class ServiceConfig:
    retrieval_model: str
    collection: str
    retrieval_config: str
    reranker_model: str
    rerank: bool
    rerank_pool: int
    top_k: int
    prompt_style: str
    llm_model: str
    qdrant_host: str
    qdrant_port: int

    @classmethod
    def from_service_json(cls) -> "ServiceConfig":
        # loads via Paths.service_config()
```

#### `pipeline.py` — `RAGService`
Initializes once (heavy objects loaded at startup, not per request):
- `SentenceTransformer(config.retrieval_model)` 
- `QdrantClient(host, port)`
- `ContextRetriever` (lazy-loads reranker + payload map on first call)
- `AnswerGenerator.from_config(...)`

Exposes one public method:
```python
def answer(self, request: ServiceRequest) -> ServiceResult:
```

Internal flow:
```
encode query
  → run_entity_boosted_retrieval(ner_category=None, ner_primary_entity=None)
  → ContextRetriever.rerank(query, candidate_ids, top_k)   [if rerank=True]
  → ContextRetriever.get_context(final_ids)
  → AnswerGenerator.generate(...)
  → return ServiceResult
```

NER note: live queries pass `ner_category=None`, `ner_primary_entity=None`.  
Entity boosting degrades gracefully to pure vector search — no NER pipeline needed at serve time.

---

### Step 4 — Smoke Test
```python
# test_service_smoke.py
from rag_pipeline.service import RAGService
from rag_pipeline.service.schema import ServiceRequest

svc = RAGService.from_service_json()
result = svc.answer(ServiceRequest(query="how do I fix a docker error", course="ml-zoomcamp"))
assert result.success
assert len(result.answer) > 0
print(result)
```

---

## Order of Execution
1. Patch `paths.json` + `Paths` → run import test
2. Create `configs/service.json`
3. Write `service/schema.py`
4. Write `service/config.py` → run load test
5. Write `service/pipeline.py` → run smoke test
6. Fix `GenerationConfig` `Paths.get()` calls (minimal patch, keep changes isolated)

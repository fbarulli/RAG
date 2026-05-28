"""
Public Functions for Answer Generation Document Retrieval and Reranking:

def collection_name_for_model(model_name: str) -> str:
    Derive Qdrant collection name from model name.
    I/O: model_name (str) -> str

def rerank(self, query: str, doc_ids: list[str], top_k: int) -> list[str]:
    Rerank doc_ids by cross-encoder score against the query.
    I/O: query (str), doc_ids (list[str]), top_k (int) -> list[str]

def get_context(self, doc_ids: list[str], max_chars_per_doc: int = _DEFAULT_MAX_CHARS_PER_DOC, max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS) -> str:
    Retrieve full answers for document IDs and combine as context.
    I/O: doc_ids (list[str]), max_chars_per_doc (int), max_context_chars (int) -> str.
"""
from typing import Optional
from qdrant_client import QdrantClient
from rag_pipeline.logging import get_logger
logger = get_logger(__name__)
_DEFAULT_MAX_CHARS_PER_DOC = 1000
_DEFAULT_MAX_CONTEXT_CHARS = 3000

def collection_name_for_model(model_name: str) -> str:
    """Derive Qdrant collection name from model name."""
    short = model_name.split('/')[-1].replace('-', '_').replace('.', '_')
    return f'faqs_{short}'

def _clean_answer(text: str) -> str:
    """Remove common introductory phrases."""
    import re
    text = re.sub('^To resolve this issue:\\s*\\n?', '', text)
    text = re.sub('^To fix this:\\s*\\n?', '', text)
    text = re.sub('^Solution:\\s*\\n?', '', text)
    return text.strip()

class ContextRetriever:
    """Retrieve and optionally rerank documents from Qdrant for answer generation."""

    def __init__(self, host: str | None=None, port: int | None=None, model_name: str | None=None, reranker_model: str='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        _d = Paths.defaults()
        host = host or _d["qdrant"]["host"]
        port = port or _d["qdrant"]["port"]
        model_name = model_name or _d["production_model"]
        self.client = QdrantClient(host=host, port=port)
        self.model_name = model_name
        self.reranker_model = reranker_model
        self._collection: Optional[str] = None
        self._payload_map: Optional[dict[str, dict]] = None
        self._reranker = None

    @property
    def collection(self) -> str:
        if self._collection is None:
            self._collection = collection_name_for_model(self.model_name)
        return self._collection

    def _load_payload_map(self) -> dict[str, dict]:
        """
        Load answer + question for all docs from Qdrant into memory.

        Extends the old _answer_map to also cache 'question', which the
        cross-encoder needs to score (query, question + answer) pairs.
        """
        if self._payload_map is not None:
            return self._payload_map
        logger.info(f'Loading payload map from {self.collection}...')
        payload_map: dict[str, dict] = {}
        offset = None
        try:
            while True:
                points, offset = self.client.scroll(collection_name=self.collection, limit=100, offset=offset, with_payload=True, with_vectors=False)
                for point in points:
                    es_id = point.payload.get('es_id', '')
                    if es_id:
                        payload_map[es_id] = {'answer': point.payload.get('answer', ''), 'question': point.payload.get('question', '')}
                if offset is None:
                    break
        except Exception as e:
            logger.error(f'Failed to load payload map from {self.collection}: {e}')
            raise
        logger.info(f'Loaded {len(payload_map)} documents from {self.collection}')
        self._payload_map = payload_map
        return payload_map

    def _get_reranker(self):
        """Lazy-load cross-encoder reranker."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            logger.info(f'Loading reranker: {self.reranker_model}')
            self._reranker = CrossEncoder(self.reranker_model)
        return self._reranker

    def rerank(self, query: str, doc_ids: list[str], top_k: int) -> list[str]:
        """
        Rerank doc_ids by cross-encoder score against the query.

        Scores (query, question + answer) pairs — combining both fields
        gives the cross-encoder more signal than answer alone, since FAQ
        questions often contain the key entity/error name.

        Args:
            query: The user query
            doc_ids: Candidate doc IDs from first-stage retrieval
            top_k: How many to return after reranking

        Returns:
            Reranked doc IDs, best first, sliced to top_k
        """
        if not doc_ids:
            return []
        payload_map = self._load_payload_map()
        reranker = self._get_reranker()
        pairs = []
        valid_ids = []
        for doc_id in doc_ids:
            payload = payload_map.get(doc_id)
            if not payload:
                logger.warning(f'No payload found for doc {doc_id} during reranking')
                continue
            doc_text = f"{payload['question']}\n{payload['answer']}"
            pairs.append([query, doc_text])
            valid_ids.append(doc_id)
        if not pairs:
            return doc_ids[:top_k]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
        reranked_ids = [doc_id for doc_id, _ in ranked[:top_k]]
        logger.debug(f'Reranked {len(valid_ids)} docs → top {top_k}: ' + ', '.join((f'{d}({s:.3f})' for d, s in ranked[:top_k])))
        return reranked_ids

    def get_context(self, doc_ids: list[str], max_chars_per_doc: int=_DEFAULT_MAX_CHARS_PER_DOC, max_context_chars: int=_DEFAULT_MAX_CONTEXT_CHARS) -> str:
        """Retrieve full answers for document IDs and combine as context."""
        payload_map = self._load_payload_map()
        contexts = []
        total_chars = 0
        for doc_id in doc_ids:
            payload = payload_map.get(doc_id)
            if not payload:
                logger.warning(f'No answer found for document {doc_id}')
                continue
            answer = _clean_answer(payload['answer'])
            if len(answer) > max_chars_per_doc:
                answer = answer[:max_chars_per_doc].rsplit(' ', 1)[0] + '...'
            if total_chars + len(answer) > max_context_chars:
                logger.debug(f'Context limit reached at {total_chars} chars')
                break
            contexts.append(answer)
            total_chars += len(answer)
        if not contexts:
            return 'No relevant documents found.'
        return '\n\n---\n\n'.join(contexts)
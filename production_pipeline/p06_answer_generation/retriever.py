"""Document retrieval for answer generation."""

from typing import Optional
from qdrant_client import QdrantClient
from rag_pipeline.logging import get_logger
from rag_pipeline.paths import Paths

logger = get_logger(__name__)

_DEFAULT_MAX_CHARS_PER_DOC = 1000
_DEFAULT_MAX_CONTEXT_CHARS = 3000


def collection_name_for_model(model_name: str) -> str:
    """Derive Qdrant collection name from model name."""
    short = model_name.split("/")[-1].replace("-", "_").replace(".", "_")
    return f"faqs_{short}"


class ContextRetriever:
    """Retrieve relevant documents from Qdrant for answer generation."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        model_name: str = "BAAI/bge-base-en-v1.5",
    ):
        self.client = QdrantClient(host=host, port=port)
        self.model_name = model_name
        self._collection: Optional[str] = None
        self._answer_map: Optional[dict[str, str]] = None

    @property
    def collection(self) -> str:
        if self._collection is None:
            self._collection = collection_name_for_model(self.model_name)
        return self._collection

    def _load_answer_map(self) -> dict[str, str]:
        """Load all answers from Qdrant into memory."""
        if self._answer_map is not None:
            return self._answer_map

        logger.info(f"Loading answer map from {self.collection}...")
        answer_map: dict[str, str] = {}
        offset = None

        try:
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    es_id = point.payload.get("es_id", "")
                    answer = point.payload.get("answer", "")
                    if es_id:
                        answer_map[es_id] = answer

                if offset is None:
                    break
        except Exception as e:
            logger.error(f"Failed to load answer map from {self.collection}: {e}")
            raise

        logger.info(f"Loaded {len(answer_map)} answers from {self.collection}")
        self._answer_map = answer_map
        return answer_map

    def get_context(
        self,
        doc_ids: list[str],
        max_chars_per_doc: int = _DEFAULT_MAX_CHARS_PER_DOC,
        max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
    ) -> str:
        """
        Retrieve full answers for document IDs and combine as context.

        Args:
            doc_ids: List of document IDs (es_id values)
            max_chars_per_doc: Max chars per individual document
            max_context_chars: Max total chars across all documents

        Returns:
            Combined context string, or fallback message if nothing found.
        """
        answer_map = self._load_answer_map()
        contexts = []
        total_chars = 0

        for doc_id in doc_ids:
            answer = answer_map.get(doc_id, "")
            if not answer:
                logger.warning(f"No answer found for document {doc_id}")
                continue

            if len(answer) > max_chars_per_doc:
                answer = answer[:max_chars_per_doc].rsplit(" ", 1)[0] + "..."

            if total_chars + len(answer) > max_context_chars:
                logger.debug(f"Context limit reached at {total_chars} chars, skipping remaining docs")
                break

            contexts.append(f"[Document {doc_id}]:\n{answer}")
            total_chars += len(answer)

        if not contexts:
            return "No relevant documents found."

        return "\n\n---\n\n".join(contexts)
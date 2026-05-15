import os
import json
from typing import Optional, List
from llama_index.core import VectorStoreIndex, Settings, load_index_from_storage
from llama_index.core.schema import Document
from llama_index.core.storage import StorageContext
from llama_index.llms.litellm import LiteLLM
from src.logger_config import logger

class LlamaIndexService:
    def __init__(self, settings: dict):
        self.settings = settings
        self.index = None
        self.query_engine = None
        self._initialize()
    
    def _initialize(self):
        # Set LLM if configured
        if self.settings.get("use_llama_llm", False):
            llm_model = self.settings.get("llama_llm_model", "nvidia_nim/meta/llama-3.1-8b-instruct")
            Settings.llm = LiteLLM(model=llm_model)
        
        if self.settings.get("build_llama_index", False):
            self._build()
        else:
            self._load()
    
    def _build(self):
        docs_path = self.settings.get("documents_json_path", "documents.json")
        if not os.path.exists(docs_path):
            logger.warning(f"Document file {docs_path} not found; cannot build LlamaIndex.")
            return
        with open(docs_path, "r") as f:
            data = json.load(f)
        documents = []
        for course in data:
            course_name = course["course"]
            for doc in course["documents"]:
                text = doc['text']
                documents.append(Document(
                    text=text,
                    metadata={
                        "course": course_name,
                        "original_question": doc.get("question", ""),
                        "id": doc.get("id", "")
                    }
                ))
        self.index = VectorStoreIndex.from_documents(documents)
        self.query_engine = self.index.as_query_engine()
        persist_dir = self.settings.get("llama_index_persist_dir", "storage/llama_index")
        os.makedirs(persist_dir, exist_ok=True)
        self.index.storage_context.persist(persist_dir=persist_dir)
        logger.info(f"Built LlamaIndex with {len(documents)} documents, persisted to {persist_dir}")
    
    def _load(self):
        persist_dir = self.settings.get("llama_index_persist_dir", "storage/llama_index")
        if os.path.exists(persist_dir) and os.path.isdir(persist_dir):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
                self.index = load_index_from_storage(storage_context)
                self.query_engine = self.index.as_query_engine()
                logger.info(f"Loaded persisted LlamaIndex from {persist_dir}")
            except Exception as e:
                raise RuntimeError(f"Failed to load persisted index: {e}")
        else:
            logger.info("No persisted LlamaIndex found; will build when needed.")
    
    def query(self, question: str) -> str:
        if not self.query_engine:
            raise RuntimeError("LlamaIndex not initialized")
        response = self.query_engine.query(question)
        return str(response)
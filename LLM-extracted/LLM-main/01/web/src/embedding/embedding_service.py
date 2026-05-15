from typing import List
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.logger_config import logger

class EmbeddingService:
    def __init__(self, model_name: str):
        self.model = HuggingFaceEmbedding(model_name=model_name)
        # warmup
        _ = self.model.get_text_embedding("warmup")
        logger.info(f"Loaded embedding model: {model_name}")
    
    def get_embedding(self, text: str) -> List[float]:
        return self.model.get_text_embedding(text)
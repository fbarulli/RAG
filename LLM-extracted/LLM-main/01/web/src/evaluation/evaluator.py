from typing import List, Dict, Any
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.litellm import LiteLLM
from src.logger_config import logger

class RAGEvaluator:
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        llm_model = settings.get("llama_llm_model", "nvidia_nim/meta/llama-3.1-8b-instruct")
        self.llm = LiteLLM(model=llm_model)
        self.faithfulness = FaithfulnessEvaluator(llm=self.llm)
        self.relevancy = RelevancyEvaluator(llm=self.llm)
        logger.info(f"Initialized RAG evaluators with LLM: {llm_model}")
    
    def evaluate(self, query: str, response_text: str, contexts: List[str]) -> Dict[str, Any]:
        try:
            faith_result = self.faithfulness.evaluate(query=query, response=response_text, contexts=contexts)
            rel_result = self.relevancy.evaluate(query=query, response=response_text, contexts=contexts)
            return {
                "faithful": faith_result.passing if faith_result else None,
                "faithfulness_score": getattr(faith_result, 'score', None),
                "relevant": rel_result.passing if rel_result else None,
                "relevancy_score": getattr(rel_result, 'score', None),
            }
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": str(e)}
    
    def evaluate_triad(self, query: str, response_text: str, contexts: List[str]) -> Dict[str, Any]:
        # No naive substring check; just faithfulness and relevancy.
        base = self.evaluate(query, response_text, contexts)
        base["contexts_provided"] = len(contexts)
        return base
import time
import traceback
import hashlib
from typing import Optional, Tuple, Dict, Any
from litellm import completion
from src.logger_config import logger, time_logger
from langfuse.decorators import observe, langfuse_context

def generate_document_id(doc: Dict[str, Any]) -> str:
    """
    Creates a deterministic hash. 
    Uses ONLY course, question, and a text snippet to ensure 100% match stability.
    """
    try:
        # Strip and lower to eliminate mismatches from hidden characters or casing
        course = str(doc['course']).strip().lower()
        question = str(doc['question']).strip().lower()
        text_snippet = str(doc['text']).strip().lower()[:50]
        
        combined = f"{course}-{question}-{text_snippet}"
        return hashlib.md5(combined.encode()).hexdigest()
    except KeyError as e:
        logger.error(f"CRITICAL: Missing field {e} for ID generation:\n{traceback.format_exc()}")
        raise

@observe()
@time_logger
def query_llm_provider(prompt: str, model_name: str, provider_prefix: str, metadata: dict, tags: list) -> Optional[str]:
    short_name = model_name.split('/')[-1][:20]
    
    langfuse_context.update_current_trace(
        name=f"LLM_{provider_prefix}_{short_name}",
        metadata={**metadata, "model": model_name, "provider": provider_prefix, "timestamp": time.time()},
        tags=tags
    )

    try:
        response = completion(
            model=f"{provider_prefix}/{model_name}",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
            timeout=60
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Provider {provider_prefix}/{model_name} failed: {e}")
        return None

@observe()
@time_logger
def query_llm(prompt: str, settings: dict, metadata: dict, tags: list) -> Tuple[str, str]:
    from src.langfuse_config import get_providers
    
    providers = get_providers(settings)
    
    if not providers:
        return "Error: No providers configured.", "NONE"
    
    for model, prefix, label, cost in providers:
        meta = {**metadata, "cost_per_1k_tokens": cost}
        logger.info(f"Requesting {label} ({model})...")
        answer = query_llm_provider(prompt, model, prefix, meta, tags)
        
        if answer:
            langfuse_context.update_current_trace(
                metadata={"successful_provider": label}
            )
            langfuse_context.flush()
            return answer, label
    
    return "Error: All enabled providers failed.", "NONE"
# test_llama8b_multiple.py
import sys
sys.path.insert(0, '/workspaces/LLM')
import logging


import json
from rag_pipeline.paths import Paths
from production_pipeline.p06_answer_generation.retriever import ContextRetriever
from production_pipeline.p06_answer_generation.config import get_prompt, get_prompt_config
from rag_pipeline.llm_client import call_llm

logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("production_pipeline").setLevel(logging.ERROR)
logging.disable(logging.CRITICAL)
# Load test queries
test_set_path = Paths.test_jsonl()
test_queries = []
with open(test_set_path) as f:
    for line in f:
        if line.strip():
            doc = json.loads(line)
            test_queries.append({
                "id": doc["id"],
                "question": doc["question"],
                "expected_id": doc.get("expected_id") or doc.get("expected_doc_id") or doc["id"],
            })
            if len(test_queries) >= 5:
                break

# Initialize retriever once
retriever = ContextRetriever(model_name="BAAI/bge-base-en-v1.5")

config = get_prompt_config("strict")

print("=" * 80)
print("TESTING 5 QUERIES WITH LLAMA 3.1 8B")
print("=" * 80)

for i, query_data in enumerate(test_queries, 1):
    print(f"\n{'='*80}")
    print(f"QUERY {i}: {query_data['id']}")
    print(f"{'='*80}")
    print(f"\nQUESTION:")
    print(f"{query_data['question']}")
    
    # Get context for this query's expected document
    context = retriever.get_context([query_data['expected_id']])
    
    print(f"\nCONTEXT (from retrieved document):")
    print(f"{context}")
    
    # Build prompt
    prompt = get_prompt("strict", context, query_data['question'])
    
    # Call LLM
    result = call_llm(
        prompt=prompt,
        max_tokens=config.max_tokens,
        model="nvidia_nim/meta/llama-3.1-8b-instruct",
        temperature=config.temperature,
        system=config.system,
    )
    
    print(f"\nMODEL RESPONSE:")
    print(f"{result.content}")
    print(f"\n--- Metrics: Latency={result.latency_ms:.0f}ms, Tokens={result.prompt_tokens} in, {result.completion_tokens} out ---")
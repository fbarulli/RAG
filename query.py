# debug_prompt_template.py
import sys
sys.path.insert(0, '/workspaces/LLM')

from production_pipeline.p06_answer_generation.config import PROMPT_CONFIGS
from production_pipeline.p06_answer_generation.retriever import ContextRetriever

# Get context
retriever = ContextRetriever(model_name='BAAI/bge-base-en-v1.5')
context = retriever.get_context(['e8df9f0d12'])

query = "Docker: When trying to run a streamlit app using docker-compose, I get: Error..."

# Check what the strict prompt template actually is
print("STRICT PROMPT TEMPLATE:")
print("=" * 60)
print(repr(PROMPT_CONFIGS["strict"].template))
print("\n" + "=" * 60)

# Format the full prompt
formatted = PROMPT_CONFIGS["strict"].format(context=context, query=query)
print("\nFULL FORMATTED PROMPT:")
print("=" * 60)
print(formatted)
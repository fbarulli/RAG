import litellm
import os

def fast_evaluate(query: str, response: str, context: str) -> dict:
    """Fast LLM evaluation using direct litellm call."""
    
    prompt = f"""Evaluate if the RESPONSE answers the QUESTION and if it's faithful to the CONTEXT.

QUESTION: {query}
CONTEXT: {context[:500]}
RESPONSE: {response[:500]}

Answer with JSON only:
{{
    "relevant": true/false,
    "faithful": true/false
}}
"""
    
    try:
        litellm_response = litellm.completion(
            model="nvidia_nim/meta/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100
        )
        
        result_text = litellm_response.choices[0].message.content
        
        # Parse JSON
        import json
        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                'faithful': result.get('faithful', False),
                'relevant': result.get('relevant', False)
            }
    except Exception as e:
        print(f"Evaluation error: {e}")
    
    return {'faithful': False, 'relevant': False}

import litellm
import json
import re
from typing import List, Dict, Any

def batch_evaluate_quality(queries_data: List[Dict], batch_size: int = 10) -> List[Dict]:
    """Evaluate answer quality in batches using LLM-as-Judge."""
    
    results = []
    
    for i in range(0, len(queries_data), batch_size):
        batch = queries_data[i:i + batch_size]
        
        prompt = """Rate each query-response pair. Return JSON array with 'relevant' (bool) and 'faithful' (bool).

"""
        for j, q in enumerate(batch):
            prompt += f"""{j+1}. Q:{q['query'][:80]} | A:{q['response'][:150]}\n"""
        
        prompt += "\nReturn: [{\"relevant\": true/false, \"faithful\": true/false}, ...]"
        
        try:
            response = litellm.completion(
                model="nvidia_nim/nvidia/nemotron-mini-4b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500
            )
            result_text = response.choices[0].message.content
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                batch_results = json.loads(json_match.group())
                results.extend(batch_results)
            else:
                results.extend([{'faithful': False, 'relevant': False} for _ in batch])
        except Exception as e:
            print(f"Batch evaluation error: {e}")
            results.extend([{'faithful': False, 'relevant': False} for _ in batch])
    
    return results

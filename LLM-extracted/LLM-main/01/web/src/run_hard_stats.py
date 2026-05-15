import json
import os
from typing import List, Dict

def get_hard_eval_set(filepath: str = "experiments/hard_eval_set.json") -> List[Dict]:
    """Load ALL paraphrased queries for full benchmark."""
    if not os.path.isabs(filepath):
        web_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(web_root, filepath)
    
    with open(filepath, 'r') as f:
        hard_eval = json.load(f)
    
    test_queries = []
    for item in hard_eval:
        for variant in item['paraphrased_queries']:
            test_queries.append({
                "query": variant,
                "course": item['expected_course'],
                "expected_id": item['expected_id']
            })
    
    print(f"📊 Loaded {len(test_queries)} test queries from {len(hard_eval)} original questions")
    return test_queries

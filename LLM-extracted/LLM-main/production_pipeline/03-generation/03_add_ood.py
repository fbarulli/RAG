"""
2-generation/03_add_ood.py
===========================
Adds out-of-distribution queries to the eval set for guardrail testing.
These queries have no correct answer in the FAQ corpus.

Output: experiments/eval_queries_with_ood.json

Run:    uv run python 03_add_ood.py
"""
import json, os
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE))

INPUT = BASE / 'experiments/eval_queries.json'
OUTPUT = BASE / 'experiments/eval_queries_with_ood.json'

OOD_QUERIES = [
    # Pure out-of-domain (should score <0.60)
    {"query": "what is the capital of Burkina Faso", "expected_id": None, "course": None, "type": "ood_pure"},
    {"query": "how do I make sourdough bread from scratch", "expected_id": None, "course": None, "type": "ood_pure"},
    {"query": "what are the best hiking trails in Patagonia", "expected_id": None, "course": None, "type": "ood_pure"},
    {"query": "how do I file taxes as a freelancer in Germany", "expected_id": None, "course": None, "type": "ood_pure"},
    {"query": "what is the airspeed velocity of an unladen swallow", "expected_id": None, "course": None, "type": "ood_pure"},
    {"query": "best way to learn piano as an adult beginner", "expected_id": None, "course": None, "type": "ood_pure"},
    {"query": "how do I train for a marathon in 6 months", "expected_id": None, "course": None, "type": "ood_pure"},
    {"query": "how do I fix a leaking kitchen faucet", "expected_id": None, "course": None, "type": "ood_pure"},
    {"query": "what are the symptoms of vitamin D deficiency", "expected_id": None, "course": None, "type": "ood_pure"},
    {"query": "how do I grow tomatoes in a small apartment", "expected_id": None, "course": None, "type": "ood_pure"},
    {"query": "what is the plot of Don Quixote", "expected_id": None, "course": None, "type": "ood_pure"},
    {"query": "how do I get a visa for visiting Japan", "expected_id": None, "course": None, "type": "ood_pure"},
    {"query": "how do I change a car tire on the highway", "expected_id": None, "course": None, "type": "ood_pure"},
    {"query": "how do I knit a scarf for beginners", "expected_id": None, "course": None, "type": "ood_pure"},
    {"query": "how do I meditate for anxiety relief", "expected_id": None, "course": None, "type": "ood_pure"},
    
    # Adversarial: tech-adjacent but not covered by FAQ (should score 0.60-0.67)
    {"query": "how do I deploy a PyTorch model on AWS Lambda", "expected_id": None, "course": None, "type": "ood_adversarial"},
    {"query": "what is gradient descent", "expected_id": None, "course": None, "type": "ood_adversarial"},
    {"query": "how do I use Apache Airflow instead of Kestra", "expected_id": None, "course": None, "type": "ood_adversarial"},
    {"query": "what is the difference between OLAP and OLTP", "expected_id": None, "course": None, "type": "ood_adversarial"},
    {"query": "how do I fine-tune Llama 3 on my own data", "expected_id": None, "course": None, "type": "ood_adversarial"},
    {"query": "what is a vector database", "expected_id": None, "course": None, "type": "ood_adversarial"},
    {"query": "how do I set up a CI/CD pipeline with GitHub Actions", "expected_id": None, "course": None, "type": "ood_adversarial"},
    {"query": "how do I use Kubernetes for model serving", "expected_id": None, "course": None, "type": "ood_adversarial"},
    {"query": "how do I implement OAuth2 with FastAPI", "expected_id": None, "course": None, "type": "ood_adversarial"},
    {"query": "best way to handle state management in a Flutter app", "expected_id": None, "course": None, "type": "ood_adversarial"},
]

with open(INPUT) as f:
    eval_queries = json.load(f)

# Add OOD queries
ood_entries = []
for q in OOD_QUERIES:
    ood_entries.append({
        'original_question': q['query'],
        'expected_id': None,
        'course': None,
        'prompt_results': {
            'ood': [q['query']]  # Single entry, no variations
        },
        'is_ood': True,
        'ood_type': q['type'],
    })

eval_queries['queries'].extend(ood_entries)
eval_queries['metadata']['ood_queries_added'] = len(OOD_QUERIES)
eval_queries['metadata']['ood_types'] = {
    'ood_pure': sum(1 for q in OOD_QUERIES if q['type'] == 'ood_pure'),
    'ood_adversarial': sum(1 for q in OOD_QUERIES if q['type'] == 'ood_adversarial'),
}
eval_queries['metadata']['updated_at'] = datetime.now().isoformat()

with open(OUTPUT, 'w') as f:
    json.dump(eval_queries, f, indent=2)

print(f"Added {len(OOD_QUERIES)} OOD queries")
print(f"  Pure OOD: {sum(1 for q in OOD_QUERIES if q['type'] == 'ood_pure')}")
print(f"  Adversarial: {sum(1 for q in OOD_QUERIES if q['type'] == 'ood_adversarial')}")
print(f"Total queries: {len(eval_queries['queries'])} (was ~50)")
print(f"Saved: {OUTPUT}")



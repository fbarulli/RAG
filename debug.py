# debug_e5_base.py
"""Debug why e5-base-v2 is failing."""

import json
from pathlib import Path

topic_path = Path("/workspaces/LLM/production_pipeline/p02_eda/experiments/topic_assignments_all.json")

with open(topic_path) as f:
    data = json.load(f)

print("Models found in topic_assignments_all.json:")
for model_name in data.get("results", {}).keys():
    print(f"  - {model_name}")

print(f"\nDoes 'intfloat/e5-base-v2' exist?")
print(f"  {'intfloat/e5-base-v2' in data.get('results', {})}")

# Check if there's a similar name
print("\nLooking for 'e5-base' patterns:")
for model_name in data.get("results", {}).keys():
    if "e5-base" in model_name.lower():
        print(f"  - {model_name}")
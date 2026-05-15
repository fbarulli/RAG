"""
gen/langfuse_score.py
======================
Logs prompt tuning and CAG evaluation results to Langfuse.
Creates a trace for each evaluation run with per-metric spans.

Run:    uv run python gen/langfuse_score.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv('configs/.env')
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")

from src.langfuse_config import init_langfuse, load_api_keys
load_api_keys()

from langfuse import Langfuse

langfuse = Langfuse()

# ── Log prompt tuning results ────────────────────────────────────────────────
with open('experiments/prompt_tuning.json') as f:
    tuning = json.load(f)

trace = langfuse.trace(
    name="prompt_tuning",
    metadata={
        'test_size': tuning['metadata']['test_size'],
        'winner': tuning['metadata']['winner'],
        'timestamp': datetime.now().isoformat(),
    },
)

for prompt_name, scores in tuning.get('scores', {}).items():
    avg = scores['avg']
    trace.span(
        name=f"prompt_{prompt_name}",
        output={
            'completeness': avg['completeness'],
            'accuracy': avg['accuracy'],
            'tone': avg['tone'],
            'total': avg['completeness'] + avg['accuracy'] + avg['tone'],
        },
    )

# ── Log CAG progress ─────────────────────────────────────────────────────────
with open('experiments/cag_answers.json') as f:
    cag = json.load(f)

trace2 = langfuse.trace(
    name="cag_progress",
    metadata={
        'total_answers': len(cag['answers']),
        'target': 1140,
        'completion': f"{len(cag['answers'])/1140:.1%}",
        'timestamp': datetime.now().isoformat(),
    },
)

# Per-course breakdown
from collections import Counter
courses = Counter(a['course'] for a in cag['answers'].values())
for course, count in courses.most_common():
    trace2.span(
        name=f"cag_course_{course}",
        output={'count': count},
    )

langfuse.flush()

print(f"✓ Logged to Langfuse:")
print(f"  Prompt tuning: {trace.id}")
print(f"  CAG progress: {trace2.id}")
print(f"  View at: https://cloud.langfuse.com")

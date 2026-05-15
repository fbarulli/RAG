"""
Score regenerated CAG answers with resume support and batching.
Saves incrementally to experiments/regenerated_scores.json
Run: uv run python /tmp/score_regenerated.py
"""
import json, os, sys, time, numpy as np
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
from dotenv import load_dotenv
load_dotenv('configs/.env')
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import FactualCorrectness
from ragas.run_config import RunConfig
from openai import OpenAI
from ragas.llms import llm_factory

BATCH_SIZE = 8

nvidia_client = OpenAI(api_key=os.getenv("NVIDIA_API_KEY"), base_url="https://integrate.api.nvidia.com/v1")
evaluator_llm = llm_factory(model="meta/llama-3.1-8b-instruct", client=nvidia_client)
evaluator_llm.model_args = {"max_tokens": 4096}
run_config = RunConfig(max_workers=BATCH_SIZE, max_retries=2, timeout=120)

# ── Load data ──────────────────────────────────────────────────────────────
with open('experiments/cag_low_ids.json') as f:
    low_ids = set(json.load(f))
with open('experiments/cag_answers_v2.json') as f:
    v2 = json.load(f)['answers']
with open('experiments/ragas_scores.json') as f:
    raw_scores = json.load(f)['scores']

def extract_score(val):
    if isinstance(val, dict):
        return val.get('factual_correctness')
    if isinstance(val, (int, float)):
        return float(val)
    return None

old_scores = {qid: extract_score(v) for qid, v in raw_scores.items()}

# ── Resume from saved progress ─────────────────────────────────────────────
new_scores = {}
failed = set()
save_path = 'experiments/regenerated_scores.json'
if os.path.exists(save_path):
    with open(save_path) as f:
        saved = json.load(f)
        new_scores = saved.get('scores', {})
        failed = set(saved.get('failed', []))
    print(f"Resumed: {len(new_scores)} scored, {len(failed)} failed")

# ── Build pending list ─────────────────────────────────────────────────────
pending = [q for q in low_ids if q not in new_scores and q not in failed and q in v2]
print(f"Pending: {len(pending)}")

if not pending:
    print("All done!")
    exit()

# ── Batch evaluate ─────────────────────────────────────────────────────────
for i in range(0, len(pending), BATCH_SIZE):
    batch_ids = pending[i:i+BATCH_SIZE]
    samples = []
    for qid in batch_ids:
        a = v2[qid]
        samples.append(SingleTurnSample(
            user_input=a['question'],
            response=a['generated_answer'],
            reference=a.get('original_answer', ''),
        ))
    
    try:
        result = evaluate(
            dataset=EvaluationDataset(samples=samples),
            metrics=[FactualCorrectness()],
            llm=evaluator_llm,
            run_config=run_config,
        )
        
        for qid, sd in zip(batch_ids, result.scores):
            c_key = next((k for k in sd if 'factual_correctness' in k), None)
            new_scores[qid] = float(sd[c_key]) if c_key else None
            old = old_scores.get(qid)
            old_str = f"{old:.3f}" if old is not None else "N/A"
            new_str = f"{new_scores[qid]:.3f}" if new_scores[qid] is not None else "N/A"
            print(f"  {qid}: {old_str} → {new_str}")
        
        # Save incrementally
        common = [q for q in low_ids if old_scores.get(q) is not None and new_scores.get(q) is not None]
        improved = sum(1 for q in common if new_scores[q] > old_scores[q]) if common else 0
        with open(save_path, 'w') as f:
            json.dump({
                'scores': new_scores,
                'old_scores': {q: old_scores[q] for q in common},
                'compared': len(common),
                'improved': improved,
                'failed': list(failed),
                'remaining': len([q for q in low_ids if q not in new_scores and q not in failed]),
            }, f, indent=2)
            
    except Exception as e:
        print(f"  Batch failed: {e}")
        for qid in batch_ids:
            failed.add(qid)
        with open(save_path, 'w') as f:
            json.dump({
                'scores': new_scores,
                'failed': list(failed),
                'remaining': len([q for q in low_ids if q not in new_scores and q not in failed]),
            }, f, indent=2)

    time.sleep(2)

# ── Final summary ──────────────────────────────────────────────────────────
common = [q for q in low_ids if old_scores.get(q) is not None and new_scores.get(q) is not None]
if common:
    old_vals = [old_scores[q] for q in common]
    new_vals = [new_scores[q] for q in common]
    improved = sum(1 for o, n in zip(old_vals, new_vals) if n > o)
    regressed = sum(1 for o, n in zip(old_vals, new_vals) if n < o)
    print(f"\nCompared: {len(common)} samples (failed: {len(failed)})")
    print(f"Old mean: {np.mean(old_vals):.3f}")
    print(f"New mean: {np.mean(new_vals):.3f}")
    print(f"Delta:    {np.mean(new_vals) - np.mean(old_vals):+.3f}")
    print(f"Improved: {improved}  Regressed: {regressed}  Unchanged: {len(common)-improved-regressed}")



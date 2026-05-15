"""
RAGAS evaluation with 70B judge — same 152 samples as 8B run.
Incremental saving, rate limit handling, valid-score means only.

Run:    uv run python gen/ragas_eval_70b.py
"""
import sys, os, json, logging, time, numpy as np
from datetime import datetime
from dotenv import load_dotenv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv('configs/.env')
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import FactualCorrectness
from ragas.run_config import RunConfig
from openai import OpenAI
from ragas.llms import llm_factory

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

CAG_FILE = 'experiments/cag_answers.json'
OUTPUT = 'experiments/ragas_70b_scores.json'
MODEL = "meta/llama-3.1-70b-instruct"

nvidia_client = OpenAI(api_key=os.getenv("NVIDIA_API_KEY"), base_url="https://integrate.api.nvidia.com/v1")
evaluator_llm = llm_factory(model=MODEL, client=nvidia_client)
evaluator_llm.model_args = {"max_tokens": 2048}

# Process one-at-a-time to preserve order and survive failures
with open('experiments/ragas_8b_ids.json') as f:
    target_ids = set(json.load(f))

with open(CAG_FILE) as f:
    cag = json.load(f)['answers']

# Load existing 70B scores if any
scores = {}
if os.path.exists(OUTPUT):
    with open(OUTPUT) as f:
        scores = json.load(f).get('scores', {})

pending = [(qid, cag[qid]) for qid in target_ids if qid in cag and qid not in scores]
logger.info(f"Total: {len(target_ids)} | Done: {len(scores)} | Pending: {len(pending)}")

if not pending:
    logger.info("All done!")
    # Show comparison
    with open('experiments/ragas_scores.json') as f:
        data_8b = json.load(f)
    scored_8b = {qid: data_8b['scores'][qid].get('factual_correctness', 0) 
                 for qid in target_ids if qid in data_8b['scores']}
    common = set(scores.keys()) & set(scored_8b.keys())
    if common:
        a8 = [scored_8b[q] for q in common]
        a70 = [scores[q] for q in common]
        logger.info(f"8B mean: {np.mean(a8):.3f}  70B mean: {np.mean(a70):.3f}")
        logger.info(f"Correlation: {np.corrcoef(a8, a70)[0,1]:.3f}")
    import sys; sys.exit(0)

run_config = RunConfig(max_workers=1, max_retries=2, timeout=90)

# Score one at a time
for i, (qid, answer) in enumerate(pending):
    sample = SingleTurnSample(
        user_input=answer['question'],
        response=answer['generated_answer'],
        reference=answer.get('original_answer', ''),
    )
    
    try:
        t0 = time.time()
        result = evaluate(
            dataset=EvaluationDataset(samples=[sample]),
            metrics=[FactualCorrectness()],
            llm=evaluator_llm,
            run_config=run_config,
        )
        elapsed = time.time() - t0
        
        sd = result.scores[0] if result.scores else {}
        c_key = [k for k in sd if 'factual_correctness' in k]
        if c_key:
            scores[qid] = float(sd[c_key[0]])
            logger.info(f"  [{len(scores)}/{len(target_ids)}] {scores[qid]:.3f} ({elapsed:.0f}s)")
        else:
            logger.warning(f"  [{len(scores)}/{len(target_ids)}] no score key — skipping")
        
        # Save incrementally
        valid = list(scores.values())
        with open(OUTPUT, 'w') as f:
            json.dump({
                'model': MODEL,
                'scores': scores,
                'mean': float(np.mean(valid)) if valid else 0.0,
                'count': len(valid),
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        
        time.sleep(5)  # Light pacing
        
    except Exception as e:
        logger.error(f"  [{i+1}] failed: {e}")
        time.sleep(10)
        continue

# Final comparison
valid_70 = list(scores.values())
logger.info(f"\n70B mean: {np.mean(valid_70):.3f} over {len(valid_70)} samples")
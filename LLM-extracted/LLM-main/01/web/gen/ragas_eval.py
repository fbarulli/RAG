"""
gen/ragas_eval.py
=================
RAGAS evaluation: FactualCorrectness + SemanticSimilarity.
Incremental saving. Resumes from last completed.

Run:    uv run python gen/ragas_eval.py --force
"""
import sys, os, json, logging, time, random, numpy as np
from datetime import datetime
from dotenv import load_dotenv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv('configs/.env')
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")

from src.langfuse_config import init_langfuse, load_api_keys
load_api_keys()

from langfuse import Langfuse
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import FactualCorrectness, SemanticSimilarity
from ragas.run_config import RunConfig
from openai import OpenAI
from ragas.llms import llm_factory

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

CAG_FILE = 'experiments/cag_answers.json'
OUTPUT = 'experiments/ragas_scores.json'
MODEL = "meta/llama-3.1-8b-instruct"
BATCH_SIZE = 10

nvidia_client = OpenAI(api_key=os.getenv("NVIDIA_API_KEY"), base_url="https://integrate.api.nvidia.com/v1")
evaluator_llm = llm_factory(model=MODEL, client=nvidia_client)
evaluator_llm.model_args = {"max_tokens": 2048}
run_config = RunConfig(max_workers=4, max_retries=3, timeout=120)

METRICS = [FactualCorrectness()]


def load_progress():
    if os.path.exists(OUTPUT):
        with open(OUTPUT) as f:
            data = json.load(f)
        return data.get('scores', {}), data.get('completed', [])
    return {}, []


def save_progress(scores, completed):
    with open(OUTPUT, 'w') as f:
        json.dump({
            'metadata': {'count': len(completed), 'timestamp': datetime.now().isoformat()},
            'scores': scores, 'completed': completed,
        }, f, indent=2)


def extract_score(score_dict, key):
    val = score_dict.get(key)
    if val is not None:
        if isinstance(val, list): val = val[0] if val else 0.0
        return float(val)
    matching = [k for k in score_dict if key in k.lower().replace(' ', '_')]
    if matching:
        val = score_dict[matching[0]]
        if isinstance(val, list): val = val[0] if val else 0.0
        return float(val)
    return 0.0


def main():
    with open(CAG_FILE) as f:
        cag = json.load(f)['answers']
    
    scores, completed = load_progress()
    pending = [(qid, cag[qid]) for qid in cag if qid not in completed]
    
    logger.info(f"CAG: {len(cag)} | Done: {len(completed)} | Pending: {len(pending)}")
    
    if not pending:
        logger.info("All done!")
        return
    
    for i in range(0, len(pending), BATCH_SIZE):
        batch = pending[i:i + BATCH_SIZE]
        
        samples = []
        ids = []
        for qid, answer in batch:
            samples.append(SingleTurnSample(
                user_input=answer['question'],
                response=answer['generated_answer'],
                reference=answer.get('original_answer', ''),
            ))
            ids.append(qid)
        
        dataset = EvaluationDataset(samples=samples)
        result = evaluate(dataset=dataset, metrics=METRICS, llm=evaluator_llm, run_config=run_config)
        
        for j, qid in enumerate(ids):
            score_dict = result.scores[j] if j < len(result.scores) else {}
            scores[qid] = {
                'factual_correctness': extract_score(score_dict, 'factual_correctness'),
                'semantic_similarity': extract_score(score_dict, 'semantic_similarity'),
            }
            completed.append(qid)
        
        save_progress(scores, completed)
        logger.info(f"  Batch {i//BATCH_SIZE+1}/{(len(pending)+BATCH_SIZE-1)//BATCH_SIZE} done")
    
    # Summary
    fc = [s['factual_correctness'] for s in scores.values() if s.get('factual_correctness', 0) > 0]
    ss = [s['semantic_similarity'] for s in scores.values() if s.get('semantic_similarity', 0) > 0]
    
    if fc:
        logger.info(f"FactualCorrectness: mean={np.mean(fc):.3f} p50={np.median(fc):.2f}")
    if ss:
        logger.info(f"SemanticSimilarity: mean={np.mean(ss):.3f} p50={np.median(ss):.2f}")
    
    langfuse = Langfuse()
    trace = langfuse.trace(name="RAGAS", metadata={'total': len(scores), 'model': MODEL})
    if fc: trace.span(name="factual_correctness", output={'mean': float(np.mean(fc))})
    if ss: trace.span(name="semantic_similarity", output={'mean': float(np.mean(ss))})
    langfuse.flush()
    logger.info("Done")


if __name__ == '__main__':
    main()
import json, os, sys, time
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
from dotenv import load_dotenv
load_dotenv('configs/.env')
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import FactualCorrectness
from ragas.run_config import RunConfig
from openai import OpenAI
from ragas.llms import llm_factory

nvidia_client = OpenAI(api_key=os.getenv("NVIDIA_API_KEY"), base_url="https://integrate.api.nvidia.com/v1")
evaluator_llm = llm_factory(model="meta/llama-3.1-8b-instruct", client=nvidia_client)

with open('experiments/cag_answers.json') as f:
    cag = json.load(f)['answers']

for qid, answer in cag.items():
    if len(answer.get('original_answer', '')) > 500:
        break

print(f"Q: {answer['question'][:60]}", flush=True)
print(f"Ref: {len(answer['original_answer'])} chars", flush=True)

sample = SingleTurnSample(
    user_input=answer['question'],
    response=answer['generated_answer'],
    reference=answer['original_answer'],
)

t0 = time.time()
result = evaluate(
    dataset=EvaluationDataset(samples=[sample]),
    metrics=[FactualCorrectness()],
    llm=evaluator_llm,
    run_config=RunConfig(max_workers=1, max_retries=3, timeout=60),
)
elapsed = time.time() - t0

print(f"Done in {elapsed:.0f}s", flush=True)
if hasattr(result, 'scores'):
    for score in result.scores:
        for k, v in score.items():
            print(f"  {k}: {v}")
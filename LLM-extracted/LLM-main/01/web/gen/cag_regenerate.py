"""
Regenerate CAG answers for low-scoring FAQs with an improved prompt
that preserves commands, code, and step-by-step instructions.

Run:    uv run python gen/cag_regenerate.py
"""
import sys, os, json, asyncio, logging, time
from dotenv import load_dotenv
from litellm import acompletion

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv('configs/.env')
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

MODEL = "nvidia_nim/meta/llama-3.1-70b-instruct"
GAP = 3.0

TECHNICAL_PROMPT = """You are a course teaching assistant. Answer the question using the FAQ below.

RULES:
- Include ALL commands, code snippets, and file paths exactly as shown
- Keep ALL step-by-step instructions — do not skip steps
- Preserve error messages and their solutions verbatim
- Be thorough: students need every detail to solve their problem

FAQ Question: {question}
FAQ Answer: {answer}

Your answer:"""

with open('experiments/cag_low_ids.json') as f:
    low_ids = set(json.load(f))

with open('experiments/cag_answers_v2.json' if os.path.exists('experiments/cag_answers_v2.json') else 'experiments/cag_answers.json') as f:
    cag = json.load(f)

logger.info(f"Regenerating {len(low_ids)} answers with technical prompt")

async def regenerate(qid, answer):
    prompt = TECHNICAL_PROMPT.format(question=answer['question'], answer=answer['original_answer'])
    for attempt in range(3):
        try:
            response = await acompletion(model=MODEL, messages=[{"role":"user","content":prompt}], temperature=0.3, max_tokens=1024)
            return response.choices[0].message.content.strip()
        except Exception as e:
            if '429' in str(e) or '502' in str(e):
                await asyncio.sleep(60 * (attempt + 1))
            else:
                await asyncio.sleep(5)
    return None

async def main():
    regenerated = 0
    for qid in low_ids:
        if qid not in cag['answers']:
            continue
        answer = cag['answers'][qid]
        new_answer = await regenerate(qid, answer)
        if new_answer:
            cag['answers'][qid]['generated_answer'] = new_answer
            regenerated += 1
            logger.info(f"  [{regenerated}/{len(low_ids)}] {answer['question'][:60]}...")
        
        with open('experiments/cag_answers_v2.json', 'w') as f:
            json.dump(cag, f, indent=2)
        
        await asyncio.sleep(GAP)
    
    logger.info(f"Regenerated {regenerated} answers")

asyncio.run(main())

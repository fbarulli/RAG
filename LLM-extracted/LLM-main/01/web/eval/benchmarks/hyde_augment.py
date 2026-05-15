"""
HyDE Data Augmentation: Generate synthetic FAQ entries for short queries.
Saves to disk for review before indexing.

Run:    uv run python eval/benchmarks/hyde_augment.py
"""
import sys, os, json, time, logging
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from litellm import completion

load_dotenv('configs/.env')
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

MODEL = "nvidia_nim/meta/llama-3.1-8b-instruct"
EMBEDDING = 'BAAI/bge-base-en-v1.5'
COLLECTION = 'faqs_bge_base_en_v1.5'
MAX_SYNTHETIC = 20  # Review limit before full run
MIN_QUERY_LENGTH = 8  # Generate for queries shorter than this

HYDE_PROMPT = """You are writing a FAQ entry for a course Q&A system. 
Given this student question, write a detailed FAQ answer (4-6 sentences).
Include technical terms, specific commands, and troubleshooting steps.

Question: {query}

Here's what the correct answer looks like (use this for factual reference):
{reference}

Write a DIFFERENTLY WORDED version that covers the same information.
Use synonyms and alternative phrasings. FAQ Answer:"""

# ── Build lookup once ─────────────────────────────────────────────────────────
client = QdrantClient('localhost', port=6333)
logger.info("Building payload lookup...")
results = client.scroll(collection_name=COLLECTION, limit=2000, with_payload=True)
id_to_payload = {p.payload.get('es_id'): p.payload for p in results[0]}
logger.info(f"  {len(id_to_payload)} docs indexed")

# ── Load queries ─────────────────────────────────────────────────────────────
with open('experiments/topic0_queries.json') as f:
    t0_data = json.load(f)

test_queries = []
for doc in t0_data['queries']:
    expected_id = doc['expected_id']
    original_q = doc['original_question']
    ref_answer = id_to_payload.get(expected_id, {}).get('answer', '')
    
    for v in doc['variations']:
        test_queries.append({
            'query': v,
            'expected_id': expected_id,
            'original_question': original_q,
            'course': doc['course'],
            'reference': ref_answer,
        })

# Filter to short queries
short_queries = [q for q in test_queries if len(q['query'].split()) < MIN_QUERY_LENGTH]
logger.info(f"Total queries: {len(test_queries)}")
logger.info(f"Short queries (<{MIN_QUERY_LENGTH} words): {len(short_queries)}")

# Sort by query length (shortest first — most vocabulary gap)
short_queries.sort(key=lambda q: len(q['query'].split()))

# ── Generate synthetics ──────────────────────────────────────────────────────
synthetic = {}
skipped = 0

for i, sq in enumerate(short_queries):
    key = sq['expected_id']
    if key in synthetic or len(synthetic) >= MAX_SYNTHETIC:
        continue
    
    prompt = HYDE_PROMPT.format(query=sq['query'], reference=sq['reference'][:500])
    
    for attempt in range(3):
        try:
            response = completion(model=MODEL, messages=[{"role":"user","content":prompt}], temperature=0.5, max_tokens=300)
            answer = response.choices[0].message.content.strip()
            synthetic[key] = {
                'synthetic_answer': answer,
                'source_query': sq['query'],
                'original_question': sq['original_question'],
                'course': sq['course'],
            }
            logger.info(f"  [{len(synthetic)}] '{sq['query'][:50]}' → {len(answer)} chars")
            break
        except Exception as e:
            if '429' in str(e) or '502' in str(e):
                time.sleep(60 * (attempt + 1))
            else:
                logger.warning(f"  Failed: {e}")
                skipped += 1
                break
    
    if len(synthetic) >= MAX_SYNTHETIC:
        break
    
    time.sleep(2)

# ── Save ─────────────────────────────────────────────────────────────────────
output = {
    'description': 'HyDE synthetic FAQ entries for short/abstract queries',
    'count': len(synthetic),
    'min_query_length': MIN_QUERY_LENGTH,
    'synthetic': synthetic,
    'next_step': 'Review answers, then run hyde_index.py to add to Qdrant and re-benchmark',
    'timestamp': datetime.now().isoformat(),
}
with open('experiments/hyde_synthetic.json', 'w') as f:
    json.dump(output, f, indent=2)

logger.info(f"\nGenerated {len(synthetic)} synthetic entries")
logger.info(f"Skipped (errors): {skipped}")
logger.info(f"Saved: experiments/hyde_synthetic.json")
logger.info(f"Next: review + hyde_index.py")
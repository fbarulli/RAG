"""
eval/generation/test_gen.py
============================
Single batch test with FULL transparency: prompt sent, raw response, parsed output.

Run:    uv run python eval/generation/test_gen.py
"""
import sys, os
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

import re, json, asyncio, time
from dotenv import load_dotenv
from litellm import acompletion
from qdrant_client import QdrantClient

load_dotenv('configs/.env')
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")

MODEL = "nvidia_nim/meta/llama-3.1-70b-instruct"

def load_prompts():
    with open('eval/generation/prompts.json') as f:
        return json.load(f)

def build_qa_pairs(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(
            f"FAQ {i}:\n"
            f"QUESTION: {doc['question']}\n"
            f"ANSWER: {doc['answer'][:400]}\n"
        )
    return "\n".join(parts)

async def main():
    client = QdrantClient('localhost', port=6333)
    results = client.scroll(collection_name='faqs', limit=100, with_payload=True)
    
    import random
    random.seed(99)
    docs = random.sample([p.payload for p in results[0]], 5)
    
    qa_pairs = build_qa_pairs(docs)
    prompts = load_prompts()
    
    for prompt_name, info in prompts.items():
        template = info['template']
        prompt = template.format(qa_pairs=qa_pairs)
        
        print(f"\n{'='*70}")
        print(f"STRATEGY: {prompt_name}")
        print(f"{'='*70}")
        print(f"\n=== PROMPT SENT ({len(prompt)} chars) ===")
        print(prompt)
        print(f"\n=== RAW RESPONSE ===")
        
        t0 = time.time()
        try:
            response = await acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
                timeout=60,
            )
            raw = response.choices[0].message.content.strip()
            elapsed = time.time() - t0
            print(raw)
            print(f"\n=== RESPONSE TIME: {elapsed:.1f}s ===")
            
            # Parse
            print(f"\n=== PARSED OUTPUT ===")
            text = re.sub(r'```(?:json)?|```', '', raw).strip()
            try:
                result = json.loads(text)
                for k, v in result.items():
                    print(f"  FAQ {k}: {v}")
                total = sum(len(v) for v in result.values() if isinstance(v, list))
                print(f"  Total queries: {total}")
            except Exception as e:
                print(f"  PARSE ERROR: {e}")
                # Try to extract JSON from text
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    print(f"  Extracted JSON fragment: {match.group()[:300]}")
                
        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR after {elapsed:.1f}s: {type(e).__name__}: {str(e)[:300]}")
        
        await asyncio.sleep(3)

if __name__ == '__main__':
    asyncio.run(main())

import sys, os, json, litellm
from dotenv import load_dotenv

# 1. Setup Environment
load_dotenv('/home/admin/LLM/LLM/01/web/configs/.env')
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")

def run_comparison():
    file_path = 'experiments/results/bm25_default.json'
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    r = data['results'][0]
    query = r['query']
    ctx_list = r.get('contexts', [])
    text = ctx_list[0] if isinstance(ctx_list, list) and len(ctx_list) > 0 else ""
    clean_context = " ".join(str(text).split())[:1500]

    # Model Dictionary with current NIM endpoints
    models = {
        "8B (Llama 3.1 Instruct)": "nvidia_nim/meta/llama-3.1-8b-instruct",
        "4B (Nemotron Nano v1.1)": "nvidia_nim/nvidia/llama-3.1-nemotron-nano-4b-v1.1"
    }

    prompt = f"""[INST] You are a helpful teaching assistant.
Does the CONTEXT provided contain enough information to answer the QUESTION, even if the answer is logically implied?

QUESTION: {query}
CONTEXT: {clean_context}

Step 1: Explain in one sentence if the answer is stated or clearly implied.
Step 2: End your response with exactly one word: 'YES' or 'NO'. [/INST]"""

    print(f"\n{'='*70}\nDEBUGGING QUERY: {query}\n{'='*70}")
    print(f"\n[INPUT CONTEXT]\n{clean_context}\n")

    for label, model_id in models.items():
        try:
            response = litellm.completion(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=150
            )
            
            raw_output = response.choices[0].message.content.strip()
            # Extracts the final word as the verdict
            verdict = "YES" if "YES" in raw_output.split()[-1].upper() else "NO"

            print(f"--- [{label}] ---")
            print(f"RATIONALE: {raw_output}")
            print(f"VERDICT  : {verdict}\n")

        except Exception as e:
            print(f"❌ {label} Error: {e}\n")

if __name__ == "__main__":
    run_comparison()

"""
eval/calibrate_judge.py
=======================
Runs 5 obviously-wrong context/question pairs through the judge model.
If the model can't say NO here, it's too lenient for your data and you
should switch to the 70B model or tighten the prompt.

Usage:
    uv run eval/calibrate_judge.py
    uv run eval/calibrate_judge.py --model 8b
"""
import asyncio
import argparse
from shared import (
    RateLimiter, llm_call, parse_verdicts,
    sanitize_query, JUDGE_MODEL_8B, JUDGE_MODEL_70B,
)

# ── Calibration cases: clearly wrong contexts ─────────────────────────────────
CASES = [
    {
        "query": "How do I install Docker on Ubuntu?",
        "contexts": [
            "The learning rate controls how fast the model updates weights during training.",
            "Use pandas.DataFrame.merge() to join two dataframes on a common column.",
            "Kafka topics can be partitioned to allow parallel consumption.",
            "The cosine similarity between two vectors measures the angle between them.",
            "MLflow tracks experiments, parameters, and metrics for machine learning runs.",
        ],
        "expected_any_yes": False,
    },
    {
        "query": "What is the deadline for the final project?",
        "contexts": [
            "Docker containers share the host OS kernel, making them lightweight.",
            "You can use git rebase to maintain a clean commit history.",
            "Terraform state files track the current state of your infrastructure.",
            "The batch size affects memory usage and training stability.",
            "Redis supports various data structures including strings, hashes, and lists.",
        ],
        "expected_any_yes": False,
    },
    {
        "query": "How do I fix an out of memory error in Spark?",
        "contexts": [
            "To bake sourdough bread, mix flour, water, salt, and starter.",
            "The French Revolution began in 1789 with the storming of the Bastille.",
            "Jupiter is the largest planet in our solar system.",
            "Shakespeare wrote Hamlet around 1600.",
            "The speed of light is approximately 299,792 kilometres per second.",
        ],
        "expected_any_yes": False,
    },
    {
        "query": "How do I connect to a PostgreSQL database in Python?",
        "contexts": [
            "The mitochondria is the powerhouse of the cell.",
            "Photosynthesis converts sunlight into glucose and oxygen.",
            "The Amazon river is the largest river by discharge volume in the world.",
            "In music theory, a tritone is an interval spanning three whole tones.",
            "Sourdough starter contains wild yeast and lactic acid bacteria.",
        ],
        "expected_any_yes": False,
    },
    {
        "query": "What Python version is required for this course?",
        "contexts": [
            "The Eiffel Tower was built in 1889 for the World's Fair.",
            "Octopuses have three hearts and blue blood.",
            "The Great Wall of China stretches over 13,000 miles.",
            "In chess, the knight moves in an L-shape.",
            "Honey never spoils; edible honey has been found in ancient Egyptian tombs.",
        ],
        "expected_any_yes": False,
    },
]


def build_prompt(query: str, contexts: list[str]) -> str:
    clean = sanitize_query(query)
    prompt = f"Question: {clean}\n\nContexts:\n"
    for i, ctx in enumerate(contexts[:5], 1):
        prompt += f"{i}. {ctx}\n\n"
    n = min(len(contexts), 5)
    prompt += (
        f"You have been given exactly {n} contexts above, numbered 1 to {n}.\n"
        f"Answer YES only if a context directly and specifically answers the question.\n"
        f"Answer NO if it is unrelated, tangential, or the question is too vague.\n"
        f"Output a JSON array of exactly {n} strings. No explanation. No extra elements.\n"
        f"Example: {['YES' if False else 'NO'] * n}\n"
        f"Answer:"
    )
    return prompt


async def run_calibration(model: str):
    rate_limiter = RateLimiter(rpm=36)
    semaphore    = asyncio.Semaphore(3)

    print(f"=== CALIBRATION  ({model}) ===\n")
    passed = 0

    for i, case in enumerate(CASES, 1):
        prompt   = build_prompt(case["query"], case["contexts"])
        raw      = await llm_call(prompt, rate_limiter, semaphore, model=model)
        n        = min(len(case["contexts"]), 5)
        verdicts = parse_verdicts(raw, n)
        any_yes  = any(v == "YES" for v in verdicts)
        correct  = any_yes == case["expected_any_yes"]

        status = "✅ PASS" if correct else "❌ FAIL"
        if correct:
            passed += 1
        print(f"Case {i}: {status}")
        print(f"  Query   : {case['query']}")
        print(f"  Verdicts: {verdicts}  (expected all NO)")
        if not correct:
            print(f"  Raw     : {raw[:200]}")
        print()

    print("=" * 50)
    print(f"Result: {passed}/{len(CASES)} passed\n")

    if passed == len(CASES):
        print("✅  Model can say NO — judge is reliable for your data.")
        print("   Your 100% precision result is likely genuine.")
    elif passed >= 3:
        print("⚠️  Model is borderline — some cases slipped through.")
        print("   Consider using the 70B model for final evaluation.")
    else:
        print("❌  Model is too lenient — cannot reliably say NO.")
        print("   Switch to 70B: uv run eval/calibrate_judge.py --model 70b")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["8b", "70b"], default="70b",
                        help="Judge model size (default: 70b)")
    args  = parser.parse_args()
    model = JUDGE_MODEL_8B if args.model == "8b" else JUDGE_MODEL_70B
    asyncio.run(run_calibration(model))
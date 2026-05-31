"""
LLM-based NER for technical FAQ questions.

Extracts primary technical entity per document using the multi-LLM client.
Results saved to experiments/llm_ner_assignments.json for later integration.

Usage:
    uv run python -m rag_pipeline.eda.topics.core.llm_ner
    uv run python -m rag_pipeline.eda.topics.core.llm_ner --limit 50
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from rag_pipeline.core.paths import Paths
from rag_pipeline.core.multi_llm_client import call_with_fallback
from rag_pipeline.logging import get_logger

load_dotenv(Paths.base() / ".env")
logger = get_logger(__name__)

SYSTEM = (
    "You are a technical entity extractor for a course FAQ dataset. "
    "Extract the single most specific technical entity from the question — "
    "a tool, library, framework, command, or error type. "
    "Reply with just the entity in lowercase. If there is no specific technical entity, reply 'none'."
)

PROMPT_TEMPLATE = "Question: {question}\n\nEntity:"

OUTPUT_PATH = Path("experiments/llm_ner_assignments.json")


def extract_entity(question: str) -> str | None:
    prompt = PROMPT_TEMPLATE.format(question=question)
    try:
        result = call_with_fallback(prompt=prompt, max_tokens=15, temperature=0.0, system=SYSTEM)
        entity = result.content.strip().lower().strip("\"'.,")
        return None if entity in ("none", "", "n/a") else entity
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    # load corpus
    docs = []
    with open(Paths.clean_jsonl(), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    if args.limit:
        docs = docs[:args.limit]

    # load existing results for resume
    existing = {}
    if args.resume and OUTPUT_PATH.exists():
        existing = json.load(open(OUTPUT_PATH))
        logger.info(f"Resuming — {len(existing)} already done")

    results = dict(existing)
    pending = [d for d in docs if d["id"] not in results]
    logger.info(f"Processing {len(pending)} docs ({len(results)} already done)")

    for i, doc in enumerate(pending):
        entity = extract_entity(doc["question"])
        results[doc["id"]] = {
            "question": doc["question"],
            "entity": entity,
            "course": doc.get("course", ""),
        }
        if (i + 1) % 50 == 0:
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            json.dump(results, open(OUTPUT_PATH, "w"), indent=2)
            logger.info(f"Progress: {i+1}/{len(pending)} — saved checkpoint")
        time.sleep(0.05)  # gentle rate limiting

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUTPUT_PATH, "w"), indent=2)
    logger.info(f"Done — {len(results)} entities saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

"""
p00_load_llm_queries.py
=======================
Loads LLM-generated eval_queries.json, joins with clean.jsonl to fetch answers,
and flattens to test.jsonl format.

Run: uv run python -m production_pipeline.p01_data_cleaning.p00_load_llm_queries
"""
import argparse
import json
from pathlib import Path

from rag_pipeline.logging import get_logger
from rag_pipeline.paths import Paths

logger = get_logger(__name__)

# Update input path to your moved file location
DEFAULT_INPUT = Paths.processed_dir().parent / "old_gen/eval_queries.json"
DEFAULT_OUTPUT = Paths.processed_dir() / "test.jsonl"
CLEAN_JSONL = Paths.processed_dir() / "clean.jsonl"

def load_faq_answers() -> dict:
    """Loads {id: answer} map from the processed clean data."""
    if not CLEAN_JSONL.exists():
        logger.warning(f"Could not find {CLEAN_JSONL}. Answers will be empty strings.")
        return {}
    
    answers = {}
    with open(CLEAN_JSONL, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            answers[doc.get("id")] = doc.get("answer", "")
    logger.info(f"Loaded {len(answers)} answers from {CLEAN_JSONL.name}")
    return answers

def load_and_flatten(input_path: Path, output_path: Path) -> int:
    """Convert eval_queries.json → test.jsonl with one row per query variant."""
    # 1. Load LLM queries
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    
    # 2. Load source answers to join
    faq_answers = load_faq_answers()
    
    queries = []
    for q in data["queries"]:
        base_id = q.get("expected_id", "unknown")
        course = q.get("course", "unknown")
        original_question = q.get("original_question", "")
        
        # Fetch answer from source data using expected_id
        answer = faq_answers.get(base_id, "")
        
        # 1. Original question
        queries.append({
            "id": f"{base_id}_orig",
            "question": original_question,
            "expected_doc_id": base_id,
            "course": course,
            "answer": answer,  # <--- Added to fix the benchmark warning
            "query_type": "original",
            "source": "llm_generated"
        })
        
        # 2. Prompt variations
        for qtype, variations in q.get("prompt_results", {}).items():
            for i, var in enumerate(variations):
                queries.append({
                    "id": f"{base_id}_{qtype}_{i}",
                    "question": var,
                    "expected_doc_id": base_id,
                    "course": course,
                    "answer": answer,  # <--- Added here too
                    "query_type": qtype,
                    "source": "llm_generated"
                })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    
    logger.info(f"✅ Converted {len(data['queries'])} base queries → {len(queries)} test queries")
    logger.info(f"📁 Saved to {output_path}")
    return len(queries)

def main() -> None:
    parser = argparse.ArgumentParser(description="Load and flatten LLM-generated queries")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return
    
    load_and_flatten(args.input, args.output)

if __name__ == "__main__":
    main()
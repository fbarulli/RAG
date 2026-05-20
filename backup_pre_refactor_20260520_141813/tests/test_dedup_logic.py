"""
tests/test_dedup_logic.py
=========================
Standalone verification of deduplication logic.
Does not modify any pipeline files — just reports what would be removed.

Run: uv run python tests/test_dedup_logic.py [--threshold 0.95] [--sample 50]
"""
import argparse
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_PATH = Path("production_pipeline/p01_data_cleaning/data/processed/parsed.jsonl")
OUTPUT_PATH = Path("tests/dedup_verification.json")
DEFAULT_THRESHOLD = 0.95
DEFAULT_SAMPLE = None  # None = process all


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_similarity(a: str, b: str) -> float:
    """Compute SequenceMatcher ratio between two normalized strings."""
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def find_duplicates(docs: list[dict], threshold: float) -> list[dict]:
    """
    Find duplicate documents by question + answer similarity.
    
    Returns list of removal records with metadata.
    """
    # Group by normalized question
    by_question = defaultdict(list)
    for doc in docs:
        q_key = normalize(doc["question"])
        by_question[q_key].append(doc)
    
    removals = []
    
    for q_key, group in by_question.items():
        if len(group) <= 1:
            continue
        
        # Compare all pairs in group
        n = len(group)
        removed_ids = set()
        
        for i in range(n):
            if i in removed_ids:
                continue
            for j in range(i + 1, n):
                if j in removed_ids:
                    continue
                
                sim = compute_similarity(group[i]["answer"], group[j]["answer"])
                if sim >= threshold:
                    # Keep the one with shorter ID (arbitrary but deterministic)
                    keep, remove = (i, j) if group[i]["id"] < group[j]["id"] else (j, i)
                    removed_ids.add(remove)
                    
                    removals.append({
                        "kept_id": group[keep]["id"],
                        "removed_id": group[remove]["id"],
                        "question": group[remove]["question"],
                        "kept_answer": group[keep]["answer"][:200],
                        "removed_answer": group[remove]["answer"][:200],
                        "similarity": round(sim, 4),
                        "course": group[remove]["course"],
                        "section": group[remove].get("section", ""),
                    })
    
    return removals


def load_documents(path: Path) -> list[dict]:
    """Load documents from JSONL file."""
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify deduplication logic")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Similarity threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                        help="Process only first N documents for quick testing")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help=f"Output JSON path (default: {OUTPUT_PATH})")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading documents from {INPUT_PATH}...")
    docs = load_documents(INPUT_PATH)
    if args.sample:
        docs = docs[:args.sample]
        print(f"Using sample of {len(docs)} documents")
    
    print(f"Total documents: {len(docs)}")
    
    # Find duplicates
    print(f"Finding duplicates with threshold {args.threshold}...")
    removals = find_duplicates(docs, args.threshold)
    
    # Summary
    unique_questions = len(set(normalize(d["question"]) for d in docs))
    print(f"\nResults:")
    print(f"  Unique questions: {unique_questions}")
    print(f"  Potential removals: {len(removals)}")
    print(f"  Estimated output size: {len(docs) - len(removals)}")
    
    # Save detailed report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "input_count": len(docs),
        "threshold": args.threshold,
        "unique_questions": unique_questions,
        "removal_count": len(removals),
        "estimated_output": len(docs) - len(removals),
        "removals": removals,
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed report saved to {args.output}")
    
    # Print sample of removals
    if removals:
        print(f"\nSample removals (first 5):")
        for r in removals[:5]:
            print(f"  Remove: {r['removed_id']} (keep {r['kept_id']})")
            print(f"    Course: {r['course']} | Section: {r['section']}")
            print(f"    Question: {r['question'][:80]}...")
            print(f"    Similarity: {r['similarity']:.2%}")
            print()


if __name__ == "__main__":
    main()
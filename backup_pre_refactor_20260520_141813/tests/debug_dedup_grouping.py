"""
tests/debug_dedup_grouping.py
=============================
Debug script to inspect how questions are being grouped for deduplication.
Shows normalized keys, group sizes, and similarity scores for suspected duplicates.

Run: uv run python tests/debug_dedup_grouping.py [--question "capstone"]
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
SIMILARITY_THRESHOLD = 0.95


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


def load_documents(path: Path) -> list[dict]:
    """Load documents from JSONL file."""
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


# ---------------------------------------------------------------------------
# Debug functions
# ---------------------------------------------------------------------------

def inspect_grouping(docs: list[dict], filter_question: str = None):
    """Inspect how documents are grouped by normalized question."""
    by_question = defaultdict(list)
    for doc in docs:
        q_key = normalize(doc["question"])
        by_question[q_key].append(doc)
    
    print(f"Total documents: {len(docs)}")
    print(f"Unique normalized questions: {len(by_question)}")
    print()
    
    # Show groups with more than 1 document
    multi_groups = {k: v for k, v in by_question.items() if len(v) > 1}
    print(f"Questions with multiple documents: {len(multi_groups)}")
    
    if filter_question:
        print(f"\nFiltering for question containing: '{filter_question}'")
        multi_groups = {
            k: v for k, v in multi_groups.items()
            if filter_question.lower() in k
        }
    
    for q_key, group in sorted(multi_groups.items(), key=lambda x: -len(x[1])):
        print(f"\n{'='*70}")
        print(f"Normalized key: {q_key[:100]}")
        print(f"Group size: {len(group)}")
        print(f"Courses represented: {set(d['course'] for d in group)}")
        print()
        
        for i, doc in enumerate(group):
            print(f"[{i}] ID: {doc['id']} | Course: {doc['course']}")
            print(f"    Question: {doc['question']}")
            print(f"    Answer (first 150 chars): {doc['answer'][:150]}...")
            print()
        
        # Compute pairwise similarities within group
        if len(group) >= 2:
            print("Pairwise answer similarities:")
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    sim = compute_similarity(group[i]["answer"], group[j]["answer"])
                    marker = " [REMOVE CANDIDATE]" if sim >= SIMILARITY_THRESHOLD else ""
                    print(f"  [{i}] vs [{j}]: {sim:.4f}{marker}")
                    if sim >= SIMILARITY_THRESHOLD:
                        print(f"    -> Would remove: {group[j]['id']} (keep {group[i]['id']})")


def search_expected_duplicates(docs: list[dict]):
    """Search for the specific duplicates mentioned in the audit."""
    expected = [
        ("capstone project evaluated", "exact duplicate"),
        ("AUC to evaluate feature importance", "near-duplicate"),
        ("evaluate feature importance", "near-duplicate"),
        ("Prefect questions", "bad answer"),
    ]
    
    print(f"\n{'#'*70}")
    print("SEARCHING FOR EXPECTED DUPLICATES")
    print(f"{'#'*70}\n")
    
    for search_term, desc in expected:
        matches = [d for d in docs if search_term.lower() in d["question"].lower()]
        if matches:
            print(f"Found {len(matches)} match(es) for '{search_term}' ({desc}):")
            for m in matches:
                print(f"  ID: {m['id']} | Course: {m['course']}")
                print(f"  Q: {m['question']}")
                print(f"  A: {m['answer'][:200]}...")
                print()
        else:
            print(f"No matches found for '{search_term}'")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Debug dedup grouping")
    parser.add_argument("--question", type=str, default=None,
                        help="Filter to show only groups containing this text")
    parser.add_argument("--search", action="store_true",
                        help="Search for expected duplicates from audit")
    args = parser.parse_args()
    
    print(f"Loading documents from {INPUT_PATH}...")
    docs = load_documents(INPUT_PATH)
    
    if args.search:
        search_expected_duplicates(docs)
    
    inspect_grouping(docs, filter_question=args.question)


if __name__ == "__main__":
    main()
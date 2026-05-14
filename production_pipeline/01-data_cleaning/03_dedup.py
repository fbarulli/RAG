"""
data_cleaning/dedup.py
======================
Removes duplicate documents. Two documents are considered duplicates
if they have the same question and their answers are substantially
identical (95%+ similarity on the full text).
Keeps the shortest ID as the canonical one.

Input:  data_cleaning/data/processed/parsed.jsonl
Output: data_cleaning/data/processed/clean.jsonl
        Deduplicated, one JSON object per line

Run:    uv run python data_cleaning/dedup.py
"""
import argparse
import json
import os
import re
import tempfile
from collections import defaultdict
from difflib import SequenceMatcher
from typing import TypedDict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_INPUT = "data_cleaning/data/processed/parsed.jsonl"
DEFAULT_OUTPUT = "data_cleaning/data/processed/clean.jsonl"

# Answers with similarity at or above this threshold are considered duplicates.
SIMILARITY_THRESHOLD = 0.95

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

_RE_NON_WORD = re.compile(r"[^\w\s]")
_RE_WHITESPACE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class Document(TypedDict):
    id: str
    question: str
    answer: str
    course: str
    section: str


class RemovedInfo(TypedDict):
    kept: str
    removed: str
    question: str
    similarity: str
    course: str


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for comparison."""
    text = text.lower()
    text = _RE_NON_WORD.sub("", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text


def similarity(a: str, b: str) -> float:
    """Return the SequenceMatcher similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()


def remove_duplicates(
    docs: list[Document],
) -> tuple[list[Document], list[RemovedInfo]]:
    """Remove near-duplicate answers within a group sharing the same question.

    Keeps the document with the shortest ID as the canonical entry (matching
    the docstring contract). Checks all pairs to ensure thorough deduplication.

    Args:
        docs: Documents that all share the same normalised question string.

    Returns:
        (kept, removed_info) where removed_info records what was dropped and why.
    """
    if len(docs) <= 1:
        return docs, []

    # Sort so the shortest ID comes first — that is the one we prefer to keep.
    docs = sorted(docs, key=lambda d: len(d["id"]))

    normalized_answers = [normalize(d["answer"]) for d in docs]

    to_remove: set[int] = set()
    removed_info: list[RemovedInfo] = []

    for i in range(len(docs)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(docs)):
            if j in to_remove:
                continue
            sim = similarity(normalized_answers[i], normalized_answers[j])
            if sim >= SIMILARITY_THRESHOLD:
                to_remove.add(j)
                removed_info.append(
                    RemovedInfo(
                        kept=docs[i]["id"],
                        removed=docs[j]["id"],
                        question=docs[j]["question"][:80],
                        similarity=f"{sim:.2%}",
                        course=docs[j]["course"],
                    )
                )

    kept = [doc for i, doc in enumerate(docs) if i not in to_remove]
    return kept, removed_info


def load_documents(input_path: str) -> list[Document]:
    """Read a JSONL file and return all documents."""
    docs: list[Document] = []
    with open(input_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  Warning: skipping malformed JSON on line {lineno}: {exc}")
    return docs


def group_by_question(docs: list[Document]) -> dict[str, list[Document]]:
    """Group documents by their normalised question string."""
    groups: dict[str, list[Document]] = defaultdict(list)
    for doc in docs:
        groups[normalize(doc["question"])].append(doc)
    return groups


def main(input_path: str = DEFAULT_INPUT, output_path: str = DEFAULT_OUTPUT) -> None:
    docs = load_documents(input_path)
    groups = group_by_question(docs)

    kept: list[Document] = []
    all_removed: list[RemovedInfo] = []

    for group in groups.values():
        kept_docs, removed = remove_duplicates(group)
        kept.extend(kept_docs)
        all_removed.extend(removed)

    # Atomic write: temp file → rename so a crash never corrupts the output.
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(output_path) or ".", suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            for doc in kept:
                f.write(json.dumps(doc) + "\n")
        os.replace(tmp_path, output_path)
    except Exception:
        os.unlink(tmp_path)
        raise

    # Summary
    print(f"Input:              {len(docs)} documents")
    print(f"Output:             {len(kept)} documents")
    print(f"Duplicates removed: {len(all_removed)}")

    if all_removed:
        print("\nRemoved duplicates:")
        for d in all_removed:
            print(f"  {d['removed']} (kept {d['kept']}) [{d['similarity']}]")
            print(f"    {d['course']}: {d['question']}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deduplicate a parsed JSONL document dataset."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Input JSONL file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help=f"Similarity threshold for deduplication (default: {SIMILARITY_THRESHOLD})",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    SIMILARITY_THRESHOLD = args.threshold
    main(input_path=args.input, output_path=args.output)
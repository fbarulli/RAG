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
import os
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher

INPUT = 'data_cleaning/data/processed/parsed.jsonl'
OUTPUT = 'data_cleaning/data/processed/clean.jsonl'

SIMILARITY_THRESHOLD = 0.95


def normalize(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_duplicates(docs: list) -> tuple[list, list]:
    """
    Remove duplicates from a list of docs with the same question.
    Keeps the first one, removes any that are 95%+ similar.
    Checks all pairs to ensure thorough dedup.
    """
    if len(docs) <= 1:
        return docs, []

    # Compute all pairwise similarities
    n = len(docs)
    to_remove = set()
    removed_info = []

    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            a = normalize(docs[i]['answer'])
            b = normalize(docs[j]['answer'])
            similarity = SequenceMatcher(None, a, b).ratio()
            if similarity >= SIMILARITY_THRESHOLD:
                to_remove.add(j)
                removed_info.append({
                    'kept': docs[i]['id'],
                    'removed': docs[j]['id'],
                    'question': docs[j]['question'][:80],
                    'similarity': f'{similarity:.2%}',
                    'course': docs[j]['course'],
                })

    kept = [doc for i, doc in enumerate(docs) if i not in to_remove]
    return kept, removed_info


def main():
    by_question = defaultdict(list)
    total = 0

    with open(INPUT) as f:
        for line in f:
            doc = json.loads(line)
            q_key = normalize(doc['question'])
            by_question[q_key].append(doc)
            total += 1

    kept = []
    all_removed = []

    for q_key, docs in by_question.items():
        kept_docs, removed = remove_duplicates(docs)
        kept.extend(kept_docs)
        all_removed.extend(removed)

    with open(OUTPUT, 'w') as f:
        for doc in kept:
            f.write(json.dumps(doc) + '\n')

    print(f'Input:  {total} documents')
    print(f'Output: {len(kept)} documents')
    print(f'Duplicates removed: {len(all_removed)}')

    if all_removed:
        print('\nRemoved duplicates:')
        for d in all_removed:
            print(f"  {d['removed']} (kept {d['kept']}) [{d['similarity']}]")
            print(f"    {d['course']}: {d['question']}")


if __name__ == '__main__':
    main()

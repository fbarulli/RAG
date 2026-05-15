"""
data_cleaning/train_test_split.py
==================================
Splits clean documents into train and holdout test sets.
Train goes to ES/Qdrant for indexing. Test becomes evaluation queries.

Output: data_cleaning/data/processed/train.jsonl
        data_cleaning/data/processed/test.jsonl

Run:    uv run python data_cleaning/train_test_split.py
"""
import json
import random
from collections import defaultdict

INPUT = 'data_cleaning/data/processed/clean.jsonl'
TRAIN_OUTPUT = 'data_cleaning/data/processed/train.jsonl'
TEST_OUTPUT = 'data_cleaning/data/processed/test.jsonl'
TEST_FRACTION = 0.20


def main():
    random.seed(42)

    # Load all docs
    docs = []
    with open(INPUT) as f:
        for line in f:
            docs.append(json.loads(line))

    # Group by course to ensure proportional sampling
    by_course = defaultdict(list)
    for doc in docs:
        by_course[doc['course']].append(doc)

    train = []
    test = []

    for course, course_docs in sorted(by_course.items()):
        random.shuffle(course_docs)
        n_test = max(1, int(len(course_docs) * TEST_FRACTION))
        
        train.extend(course_docs[:-n_test])
        test.extend(course_docs[-n_test:])

    # Save
    for path, data in [(TRAIN_OUTPUT, train), (TEST_OUTPUT, test)]:
        with open(path, 'w') as f:
            for doc in data:
                f.write(json.dumps(doc) + '\n')

    print(f"Total documents: {len(docs)}")
    print(f"Train: {len(train)} → {TRAIN_OUTPUT}")
    print(f"Test:  {len(test)} → {TEST_OUTPUT}")
    print(f"Test fraction: {len(test)/len(docs):.1%}")

    # Show distribution
    print("\nPer course:")
    for course in sorted(by_course.keys()):
        c_docs = by_course[course]
        c_test = [d for d in test if d['course'] == course]
        print(f"  {course}: {len(c_docs)} total → {len(c_test)} test ({len(c_test)/len(c_docs):.0%})")


if __name__ == '__main__':
    main()

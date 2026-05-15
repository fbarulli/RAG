"""
p04_split.py
============
Stratified train/test split of cleaned FAQ documents.
Ensures proportional representation per course in the holdout set.

Input:  clean.jsonl
Output: train.jsonl, test.jsonl

Run:    just run split
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from rag_pipeline.paths import Paths
from rag_pipeline.schemas import FAQDocument
from rag_pipeline.logging import get_logger

logger = get_logger(__name__)

DEFAULT_INPUT = Paths.processed_dir() / "clean.jsonl"
TRAIN_OUTPUT = Paths.processed_dir() / "train.jsonl"
TEST_OUTPUT = Paths.processed_dir() / "test.jsonl"
TEST_FRACTION = 0.20
RANDOM_SEED = 42

def main(input_path: Path = DEFAULT_INPUT, test_frac: float = TEST_FRACTION):
    random.seed(RANDOM_SEED)
    docs = [FAQDocument.from_dict(json.loads(line)) for line in open(input_path) if line.strip()]
    logger.info(f"Loaded {len(docs)} documents from {input_path}")

    by_course = defaultdict(list)
    for doc in docs:
        by_course[doc.course].append(doc)

    train, test = [], []
    for course, course_docs in sorted(by_course.items()):
        random.shuffle(course_docs)
        n_test = max(1, int(len(course_docs) * test_frac))
        train.extend(course_docs[:-n_test])
        test.extend(course_docs[-n_test:])

    for path, data in [(TRAIN_OUTPUT, train), (TEST_OUTPUT, test)]:
        with open(path, 'w', encoding='utf-8') as f:
            for doc in data:
                f.write(doc.to_json() + '\n')

    logger.info(f"Train: {len(train)} → {TRAIN_OUTPUT}")
    logger.info(f"Test:  {len(test)} → {TEST_OUTPUT}")
    logger.info(f"Test fraction: {len(test)/len(docs):.1%}")
    logger.info("Per course distribution:")
    for course in sorted(by_course.keys()):
        c_docs = by_course[course]
        c_test = sum(1 for d in test if d.course == course)
        logger.info(f"  {course}: {len(c_docs)} total → {c_test} test ({c_test/len(c_docs):.0%})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--test-fraction', type=float, default=TEST_FRACTION)
    args = parser.parse_args()
    main(args.input, args.test_fraction)

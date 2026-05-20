"""
p03_dedup.py
============
Removes duplicate documents. Two documents are considered duplicates
if they have the same question and their answers are substantially
identical (95%+ similarity on the full text).

By default, deduplication is course-aware: only documents from the same
course are compared. Use --global-dedup to deduplicate across all courses.

Keeps the document with the shorter ID as canonical (deterministic tiebreaker).

Input:  parsed.jsonl
Output: clean.jsonl (deduplicated)

Run:    just run dedup
        just run dedup "--global-dedup"  # dedupe across courses
"""
import argparse
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from rag_pipeline.core.paths import Paths
from rag_pipeline.core.schemas import FAQDocument
from rag_pipeline.core.logging import get_logger
logger = get_logger(__name__)
DEFAULT_INPUT = Paths.processed_dir() / 'parsed.jsonl'
DEFAULT_OUTPUT = Paths.processed_dir() / 'clean.jsonl'
DEFAULT_THRESHOLD = 0.95
DEFAULT_COURSE_AWARE = True

def normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub('[^\\w\\s]', '', text)
    text = re.sub('\\s+', ' ', text).strip()
    return text

def compute_similarity(a: str, b: str) -> float:
    """Compute SequenceMatcher ratio between two normalized strings."""
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()

def find_duplicates_in_group(docs: list[FAQDocument], threshold: float, course_aware: bool=True) -> tuple[list[FAQDocument], list[dict]]:
    """
    Remove duplicates from a list of docs with the same question.
    
    Args:
        docs: List of FAQDocument objects with identical normalized questions
        threshold: Similarity threshold for considering answers duplicates
        course_aware: If True, only dedupe within same course; if False, dedupe globally
    
    Returns:
        Tuple of (kept_docs, removed_records)
    """
    if len(docs) <= 1:
        return (docs, [])
    if course_aware:
        by_course = defaultdict(list)
        for doc in docs:
            by_course[doc.course].append(doc)
        groups = list(by_course.values())
    else:
        groups = [docs]
    kept = []
    removed_info = []
    for group in groups:
        if len(group) <= 1:
            kept.extend(group)
            continue
        n = len(group)
        removed_ids = set()
        for i in range(n):
            if i in removed_ids:
                continue
            for j in range(i + 1, n):
                if j in removed_ids:
                    continue
                sim = compute_similarity(group[i].answer, group[j].answer)
                if sim >= threshold:
                    keep_idx, remove_idx = (i, j) if group[i].id < group[j].id else (j, i)
                    removed_ids.add(remove_idx)
                    removed_info.append({'kept_id': group[keep_idx].id, 'removed_id': group[remove_idx].id, 'question': group[remove_idx].question, 'similarity': f'{sim:.4f}', 'course': group[remove_idx].course, 'section': group[remove_idx].section or '', 'kept_answer': group[keep_idx].answer, 'removed_answer': group[remove_idx].answer})
        for idx, doc in enumerate(group):
            if idx not in removed_ids:
                kept.append(doc)
    return (kept, removed_info)

def main(input_path: Path=DEFAULT_INPUT, output_path: Path=DEFAULT_OUTPUT, threshold: float=DEFAULT_THRESHOLD, course_aware: bool=DEFAULT_COURSE_AWARE):
    docs = []
    with open(input_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            docs.append(FAQDocument.from_dict(json.loads(line)))
    logger.info(f'Loaded {len(docs)} documents from {input_path}')
    by_question = defaultdict(list)
    for doc in docs:
        q_key = normalize(doc.question)
        by_question[q_key].append(doc)
    logger.info(f'Found {len(by_question)} unique questions')
    kept = []
    all_removed = []
    for q_key, group in by_question.items():
        kept_docs, removed = find_duplicates_in_group(group, threshold, course_aware)
        kept.extend(kept_docs)
        all_removed.extend(removed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in kept:
            f.write(doc.to_json() + '\n')
    logger.info(f'Input:  {len(docs)} documents')
    logger.info(f'Output: {len(kept)} documents')
    logger.info(f'Duplicates removed: {len(all_removed)}')
    logger.info(f"Mode: {('course-aware' if course_aware else 'global')}")
    if all_removed:
        logger.info('Removed duplicates:')
        for r in all_removed:
            logger.info(f"  Kept ID: {r['kept_id']}")
            logger.info(f"  Removed ID: {r['removed_id']}")
            logger.info(f"  Question: {r['question']}")
            logger.info(f"  Similarity: {r['similarity']}")
            logger.info(f"  Course: {r['course']}")
            logger.info(f"  Section: {r['section']}")
            logger.info(f"  Kept Answer: {r['kept_answer']}")
            logger.info(f"  Removed Answer: {r['removed_answer']}")
            logger.info('---')
    report_path = Paths.experiments_dir() / 'dedup_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({'input_count': len(docs), 'output_count': len(kept), 'removed_count': len(all_removed), 'threshold': threshold, 'course_aware': course_aware, 'removed': all_removed}, f, indent=2, ensure_ascii=False)
    logger.info(f'Saved removal report: {report_path}')

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Deduplicate FAQ documents')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT, help=f'Input JSONL path (default: {DEFAULT_INPUT})')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help=f'Output JSONL path (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD, help=f'Similarity threshold for dedup (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--global-dedup', action='store_true', help='Deduplicate across all courses (default: course-aware only)')
    return parser
if __name__ == '__main__':
    args = _build_parser().parse_args()
    main(input_path=args.input, output_path=args.output, threshold=args.threshold, course_aware=not args.global_dedup)
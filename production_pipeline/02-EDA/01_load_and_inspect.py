"""
02-EDA/01_load_and_inspect.py
==============================
Load the cleaned FAQ dataset and print comprehensive statistics.
Validates required fields, logs issues, saves summary JSON.

Output: prints to terminal, saves experiments/eda_summary.json

Run:    uv run python 01_load_and_inspect.py [--dry-run]
"""
import sys
import os
import json
import re
import logging
import hashlib
import argparse
from pathlib import Path
from collections import Counter

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE))

INPUT = BASE / '01-data-cleaning/data/processed/clean.jsonl'
OUTPUT = BASE / 'experiments/eda_summary.json'

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

STOPWORDS = {
    'the','a','an','how','do','i','my','to','is','in','of','and','it','me',
    'for','with','can','what','why','when','using','use','get','that','this',
    'not','on','be','so','but','or','we','you','are','does','have','has','been',
    'will','would','should','could','just','all','if','no','am','up','out','some',
    'any','very','really','need','go','going','way','also','as','at','its','from',
    'like','make','more','than','too','one','about','which','there','their','them',
}


def load_docs(path: Path) -> list[dict]:
    """Load and validate JSONL dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    docs = []
    with open(path, encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                # Validate required fields
                missing = {'id', 'question', 'answer', 'course'} - doc.keys()
                if missing:
                    logger.warning(f"Line {line_num}: missing fields {missing} — skipping")
                    continue
                if len(doc['question']) < 5:
                    logger.warning(f"Line {line_num}: question too short — skipping")
                    continue
                if len(doc['answer']) < 10:
                    logger.warning(f"Line {line_num}: answer too short — skipping")
                    continue
                docs.append(doc)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: malformed JSON — {e}")
                continue
    return docs


def compute_length_stats(values: list[int]) -> dict:
    """Min, max, mean, percentiles for a list of lengths."""
    if not values:
        # Return safe defaults to prevent KeyError if called on empty list
        return {'min': 0, 'max': 0, 'mean': 0.0, 'p10': 0, 'p50': 0, 'p90': 0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return {
        'min': min(values),
        'max': max(values),
        'mean': round(sum(values) / n, 1),
        'p10': sorted_vals[n // 10],
        'p50': sorted_vals[n // 2],
        'p90': sorted_vals[n * 9 // 10],
    }


def extract_words(texts: list[str]) -> Counter:
    """Extract and count meaningful words from texts."""
    pattern = re.compile(r'\b[a-z][a-z-]*[a-z]\b')
    words = []
    for text in texts:
        words.extend(
            w for w in pattern.findall(text.lower())
            if w not in STOPWORDS and len(w) > 2
        )
    return Counter(words)


def compute_answer_signals(answers: list[str]) -> dict:
    """Quality signals for answers."""
    return {
        'has_code_block': sum(1 for a in answers if '```' in a),
        'has_url': sum(1 for a in answers if 'http' in a),
        'has_list': sum(1 for a in answers if re.search(r'[\*\-]\s|\d+\.', a)),
    }


def file_hash(path: Path) -> str:
    """SHA256 hash of file for versioning."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def print_bar(value: float, max_value: float, width: int = 25) -> str:
    """Simple text bar."""
    if max_value == 0:
        return ''
    return '█' * int((value / max_value) * width)


def main():
    parser = argparse.ArgumentParser(description='EDA on cleaned FAQ dataset')
    parser.add_argument('--dry-run', action='store_true', help='Skip writing output file')
    args = parser.parse_args()

    docs = load_docs(INPUT)
    if not docs:
        logger.error("No valid documents loaded.")
        return
    
    questions = [d['question'] for d in docs]
    answers = [d['answer'] for d in docs]
    courses = [d['course'] for d in docs]
    sections = [d.get('section', 'unknown') for d in docs]
    ids = [d['id'] for d in docs]

    # ── Basic counts ──────────────────────────────────────────────────────
    logger.info(f"{'='*60}")
    logger.info("DATASET OVERVIEW")
    logger.info(f"{'='*60}")
    logger.info(f"Total documents: {len(docs)}")
    logger.info(f"Unique IDs: {len(set(ids))}")
    dup_count = len(ids) - len(set(ids))
    if dup_count > 0:
        logger.warning(f"Duplicate IDs: {dup_count}")

    # ── By course ─────────────────────────────────────────────────────────
    course_counts = Counter(courses)
    logger.info(f"\n{'='*60}")
    logger.info("BY COURSE")
    logger.info(f"{'='*60}")
    for course, count in course_counts.most_common():
        pct = count / len(docs) * 100
        bar = print_bar(pct, 100)
        logger.info(f"  {course:<30}: {count:>4} ({pct:>5.1f}%) {bar}")

    # ── By section ────────────────────────────────────────────────────────
    section_counts = Counter(sections)
    logger.info(f"\n{'='*60}")
    logger.info("BY SECTION (top 15)")
    logger.info(f"{'='*60}")
    for section, count in section_counts.most_common(15):
        pct = count / len(docs) * 100
        logger.info(f"  {section:<35}: {count:>4} ({pct:>5.1f}%)")

    # ── Question lengths ──────────────────────────────────────────────────
    q_stats = compute_length_stats([len(q) for q in questions])
    logger.info(f"\n{'='*60}")
    logger.info("QUESTION LENGTHS (chars)")
    logger.info(f"{'='*60}")
    logger.info(f"  Min: {q_stats['min']:>6}  Max: {q_stats['max']:>6}  Mean: {q_stats['mean']:.0f}")
    logger.info(f"  P10: {q_stats['p10']:>6}  P50: {q_stats['p50']:>6}  P90: {q_stats['p90']:>6}")

    # ── Answer lengths ────────────────────────────────────────────────────
    a_stats = compute_length_stats([len(a) for a in answers])
    logger.info(f"\n{'='*60}")
    logger.info("ANSWER LENGTHS (chars)")
    logger.info(f"{'='*60}")
    logger.info(f"  Min: {a_stats['min']:>6}  Max: {a_stats['max']:>6}  Mean: {a_stats['mean']:.0f}")
    logger.info(f"  P10: {a_stats['p10']:>6}  P50: {a_stats['p50']:>6}  P90: {a_stats['p90']:>6}")

    # ── Answer quality signals ────────────────────────────────────────────
    signals = compute_answer_signals(answers)
    logger.info(f"\n{'='*60}")
    logger.info("ANSWER QUALITY SIGNALS")
    logger.info(f"{'='*60}")
    logger.info(f"  Has code blocks: {signals['has_code_block']} ({signals['has_code_block']/len(docs):.1%})")
    logger.info(f"  Has URLs:        {signals['has_url']} ({signals['has_url']/len(docs):.1%})")
    logger.info(f"  Has lists:       {signals['has_list']} ({signals['has_list']/len(docs):.1%})")

    # ── Most common words ─────────────────────────────────────────────────
    word_counts = extract_words(questions)
    logger.info(f"\n{'='*60}")
    logger.info("MOST COMMON WORDS (questions)")
    logger.info(f"{'='*60}")
    for word, count in word_counts.most_common(30):
        logger.info(f"  {word:<25}: {count}")

    # ── Per-course section breakdown ──────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("PER-COURSE SECTIONS")
    logger.info(f"{'='*60}")
    for course in sorted(course_counts.keys()):
        course_docs = [d for d in docs if d['course'] == course]
        course_sections = Counter(d.get('section', '?') for d in course_docs)
        tech_count = sum(1 for d in course_docs if 'module' in d.get('section', '').lower())
        logger.info(f"\n  [{course}] {len(course_docs)} questions")
        logger.info(f"    Technical (module sections): {tech_count} ({tech_count/len(course_docs):.0%})")
        logger.info(f"    Top sections:")
        for section, count in course_sections.most_common(5):
            logger.info(f"      {section:<30}: {count}")

    # ── Save summary ──────────────────────────────────────────────────────
    summary = {
        'total_docs': len(docs),
        'unique_ids': len(set(ids)),
        'duplicate_ids': dup_count,
        'input_file_hash': file_hash(INPUT),
        'courses': dict(course_counts),
        'sections': dict(section_counts),
        'question_length': q_stats,
        'answer_length': a_stats,
        'answer_signals': signals,
        'top_words': dict(word_counts.most_common(50)),
    }

    if not args.dry_run:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"\nSaved summary: {OUTPUT}")
    else:
        logger.info("\nDry-run: output file not written")


if __name__ == '__main__':
    main()
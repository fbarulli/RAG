"""
02_eda/01_load_and_inspect.py
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
from collections import Counter, defaultdict
from rag_pipeline.paths import Paths
from rag_pipeline.schemas import FAQDocument

INPUT = Paths.input_file("eda")
OUTPUT = Paths.output_file("eda")

from rag_pipeline.logging import get_logger
logger = get_logger(__name__)

STOPWORDS = {
    'the','a','an','how','do','i','my','to','is','in','of','and','it','me',
    'for','with','can','what','why','when','using','use','get','that','this',
    'not','on','be','so','but','or','we','you','are','does','have','has','been',
    'will','would','should','could','just','all','if','no','am','up','out','some',
    'any','very','really','need','go','going','way','also','as','at','its','from',
    'like','make','more','than','too','one','about','which','there','their','them',
}


def load_docs(path: Path) -> list[FAQDocument]:
    """Load and validate JSONL dataset into FAQDocument objects."""
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
                docs.append(FAQDocument.from_dict(doc))
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: malformed JSON — {e}")
            except Exception as e:
                logger.warning(f"Line {line_num}: schema validation failed — {e}")
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
    
    questions = [d.question for d in docs]
    answers = [d.answer for d in docs]
    courses = [d.course for d in docs]
    sections = [d.section or 'unknown' for d in docs]
    ids = [d.id for d in docs]

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
        course_docs = [d for d in docs if d.course == course]
        course_sections = Counter(d.section or '?' for d in course_docs)
        tech_count = sum(1 for d in course_docs if 'module' in (d.section or '').lower())
        logger.info(f"\n  [{course}] {len(course_docs)} questions")
        logger.info(f"    Technical (module sections): {tech_count} ({tech_count/len(course_docs):.0%})")
        logger.info(f"    Top sections:")
        for section, count in course_sections.most_common(5):
            logger.info(f"      {section:<30}: {count}")

    # ── High-Signal Relationship Analyses ─────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("RUNNING DEEP RELATIONSHIP ANALYSIS")
    logger.info(f"{'='*60}")
    relationships = compute_relationships(docs)
    recommendations = generate_recommendations(relationships)
    
    logger.info("\n📊 KEY FINDINGS & RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"  {i}. {rec}")

    # ── Save summary ──────────────────────────────────────────────────────
    summary = {
        'total_docs': len(docs),
        'unique_ids': len(set(ids)),
        'duplicate_ids': dup_count,
        'input_file_hash': file_hash(INPUT),
        'courses': dict(course_counts),
        'sections': dict(section_counts),
        'question_length': {k: v for k, v in q_stats.items() if v is not None},
        'answer_length': {k: v for k, v in a_stats.items() if v is not None},
        'answer_signals': signals,
        'top_words': dict(word_counts.most_common(50)),
        'relationships': relationships,
        'recommendations': recommendations,
    }

    if not args.dry_run:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"\nSaved summary: {OUTPUT}")
    else:
        logger.info("\nDry-run: output file not written")


# ── Relationship Analysis Functions ───────────────────────────────────────
def analyze_short_answers(docs: list[FAQDocument], threshold: int = 50) -> list[dict]:
    return [
        {'id': d.id, 'question': d.question, 'answer': d.answer, 'course': d.course, 'answer_length': len(d.answer)}
        for d in docs if len(d.answer.strip()) < threshold
    ]

def analyze_code_signals(docs: list[FAQDocument]) -> dict:
    by_course = {}
    for course in set(d.course for d in docs):
        course_docs = [d for d in docs if d.course == course]
        code_count = sum(1 for d in course_docs if '```' in d.answer or '<code>' in d.answer.lower())
        by_course[course] = {'total': len(course_docs), 'with_code': code_count, 'code_pct': round(code_count / len(course_docs) * 100, 2) if course_docs else 0}
    
    code_keywords = {'docker', 'python', 'pip', 'git', 'aws', 'gcp', 'spark', 'mlflow', 'terraform', 'kubernetes'}
    potential_misses = []
    for d in docs:
        if '```' not in d.answer and '<code>' not in d.answer.lower():
            q_words = set(re.findall(r'\b[a-z]+\b', d.question.lower()))
            if q_words & code_keywords:
                potential_misses.append({'id': d.id, 'question': d.question[:100], 'course': d.course, 'matched_keywords': list(q_words & code_keywords)})
    return {'by_course': by_course, 'potential_code_misses': potential_misses[:50]}

def analyze_course_imbalance(docs: list[FAQDocument]) -> dict:
    counts = Counter(d.course for d in docs)
    total = len(docs)
    bias = {c: {'count': n, 'pct': round(n/total*100, 1), 'weight_vs_smallest': round(n/min(counts.values()), 2)} for c, n in counts.items()}
    ratio = max(counts.values()) / min(counts.values())
    return {'distribution': bias, 'max_min_ratio': ratio, 'recommendation': 'Consider course-aware filtering' if ratio > 3 else 'Balanced enough'}

def analyze_query_intent(docs: list[FAQDocument]) -> dict:
    intents = {'troubleshooting': {'error','failed','cannot','unable','denied','exception','bug'}, 'setup': {'install','setup','docker','environment'}, 'conceptual': {'what','why','explain','difference','vs'}, 'homework': {'homework','assignment','deadline','leaderboard'}}
    overall = Counter()
    by_course = defaultdict(Counter)
    for d in docs:
        q = d.question.lower()
        matched = [i for i, kws in intents.items() if any(k in q for k in kws)]
        for i in matched or ['other']:
            overall[i] += 1
            by_course[d.course][i] += 1
    return {'overall': dict(overall), 'by_course': {c: dict(v) for c, v in by_course.items()}}

def compute_qa_correlation_safe(q_lens, a_lens):
    try:
        from scipy import stats
        r, p = stats.pearsonr(q_lens, a_lens)
        return {'pearson_r': round(r, 3), 'p_value': round(p, 4)}
    except ImportError:
        return None

def compute_relationships(docs: list[FAQDocument]) -> dict:
    return {
        'short_answers': analyze_short_answers(docs),
        'code_signals': analyze_code_signals(docs),
        'course_imbalance': analyze_course_imbalance(docs),
        'query_intent': analyze_query_intent(docs),
        'qa_correlation': compute_qa_correlation_safe([len(d.question) for d in docs], [len(d.answer) for d in docs]),
        'answer_length_distribution': {
            'very_short (<50)': sum(1 for d in docs if len(d.answer) < 50),
            'short (50-200)': sum(1 for d in docs if 50 <= len(d.answer) < 200),
            'medium (200-800)': sum(1 for d in docs if 200 <= len(d.answer) < 800),
            'long (800+)': sum(1 for d in docs if len(d.answer) >= 800)
        }
    }

def generate_recommendations(relationships: dict) -> list[str]:
    recs = []
    short = relationships['short_answers']
    if len(short) > 10: recs.append(f"🔴 Audit {len(short)} answers <50 chars — likely broken/placeholder content")
    code = relationships['code_signals']
    low_code = [c for c, s in code['by_course'].items() if s['code_pct'] < 1.0 and s['total'] > 20]
    if low_code: recs.append(f"🔴 Investigate code parsing for {', '.join(low_code)} — expected more code blocks")
    imb = relationships['course_imbalance']
    if imb['max_min_ratio'] > 5: recs.append(f"🟡 Retrieval bias risk (max/min ratio: {imb['max_min_ratio']:.1f}) — consider course-aware filtering")
    intent = relationships['query_intent']
    if intent['overall'].get('troubleshooting', 0) > intent['overall'].get('conceptual', 0) * 2: recs.append("🟢 Corpus is troubleshooting-heavy — prioritize error-message retrieval")
    dist = relationships['answer_length_distribution']
    if dist['long (800+)'] / sum(dist.values()) > 0.2: recs.append("🟡 20%+ answers are long — implement hierarchical chunking")
    return recs if recs else ["✅ No critical data quality issues detected"]


if __name__ == '__main__':
    main()

"""
04_data_audit.py
================
Deep data quality audit for parsed FAQ documents.
Checks code block integrity, encoding, metadata, and content patterns.

Run: uv run python production_pipeline/p02_eda/04_data_audit.py [--sample N]
"""
import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from rag_pipeline.core.paths import Paths
from rag_pipeline.core.schemas import FAQDocument
from rag_pipeline.logging import get_logger
logger = get_logger(__name__)

def load_docs(path: Path) -> list[FAQDocument]:
    docs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(FAQDocument.from_dict(json.loads(line)))
            except Exception as e:
                logger.warning(f'Skipping malformed line: {e}')
    return docs

def audit_code_blocks(docs: list[FAQDocument]) -> dict:
    """Check if fenced code blocks are complete and well-formed."""
    results = {'total_with_code': 0, 'complete_blocks': 0, 'truncated_blocks': 0, 'malformed_blocks': 0, 'by_course': defaultdict(lambda: {'total': 0, 'complete': 0, 'truncated': 0})}
    complete_pattern = re.compile('```(?:\\w+)?\\n(?:[^\\n]*\\n)+```', re.MULTILINE)
    truncated_pattern = re.compile('```(?:\\w+)?\\n[^\\n]*(?:\\n[^\\n]*){0,2}(?!\\n```)', re.MULTILINE)
    for doc in docs:
        blocks = re.findall('```.*?```', doc.answer, re.DOTALL)
        if not blocks:
            continue
        results['total_with_code'] += 1
        results['by_course'][doc.course]['total'] += 1
        for block in blocks:
            if complete_pattern.search(block):
                results['complete_blocks'] += 1
                results['by_course'][doc.course]['complete'] += 1
            elif truncated_pattern.search(block):
                results['truncated_blocks'] += 1
                results['by_course'][doc.course]['truncated'] += 1
            else:
                results['malformed_blocks'] += 1
    return results

def audit_content_quality(docs: list[FAQDocument]) -> dict:
    """Check for encoding issues, unusual patterns, and content anomalies."""
    results = {'encoding_issues': 0, 'empty_answers': 0, 'placeholder_answers': 0, 'excessive_whitespace': 0, 'unusual_chars': Counter(), 'answer_length_outliers': [], 'question_answer_ratio_outliers': []}
    placeholders = {'todo', 'coming soon', 'n/a', 'tbd', 'fixme', 'placeholder', 'see above', 'see below'}
    for doc in docs:
        try:
            doc.answer.encode('utf-8')
        except UnicodeEncodeError:
            results['encoding_issues'] += 1
            continue
        ans_lower = doc.answer.strip().lower()
        if not ans_lower:
            results['empty_answers'] += 1
        elif ans_lower in placeholders:
            results['placeholder_answers'] += 1
        if len(doc.answer) - len(re.sub('\\s', '', doc.answer)) > len(doc.answer) * 0.5:
            results['excessive_whitespace'] += 1
        for char in doc.answer:
            if unicodedata.category(char).startswith('C'):
                results['unusual_chars'][repr(char)] += 1
        if len(doc.answer) < 10 or len(doc.answer) > 2000:
            results['answer_length_outliers'].append({'id': doc.id, 'course': doc.course, 'length': len(doc.answer), 'question': doc.question[:50], 'answer_snippet': doc.answer[:100]})
        q_len, a_len = (len(doc.question), len(doc.answer))
        if q_len < 20 and a_len > 1000 or (q_len > 200 and a_len < 50):
            results['question_answer_ratio_outliers'].append({'id': doc.id, 'q_len': q_len, 'a_len': a_len, 'question': doc.question[:50], 'answer_snippet': doc.answer[:100]})
    return results

def audit_metadata(docs: list[FAQDocument]) -> dict:
    """Check metadata completeness and consistency."""
    results = {'missing_ids': 0, 'duplicate_ids': 0, 'missing_courses': 0, 'unknown_courses': set(), 'missing_sections': 0, 'section_distribution': Counter()}
    ids = [d.id for d in docs]
    results['duplicate_ids'] = len(ids) - len(set(ids))
    for doc in docs:
        if not doc.id:
            results['missing_ids'] += 1
        if not doc.course:
            results['missing_courses'] += 1
        else:
            results['unknown_courses'].add(doc.course)
        if not doc.section:
            results['missing_sections'] += 1
        else:
            results['section_distribution'][doc.section] += 1
    results['unknown_courses'] = list(results['unknown_courses'])
    return results

def audit_code_patterns(docs: list[FAQDocument]) -> dict:
    """Analyze what types of code appear (bash, python, yaml, etc.)."""
    lang_counts = Counter()
    inline_code_counts = Counter()
    for doc in docs:
        fenced = re.findall('```(\\w+)?', doc.answer)
        for lang in fenced:
            lang_counts[lang or 'unknown'] += 1
        inline = re.findall('`([^`]+)`', doc.answer)
        for code in inline:
            if any((kw in code.lower() for kw in ['import', 'def ', 'class ', 'print(', 'pip ', 'python'])):
                inline_code_counts['python'] += 1
            elif any((kw in code.lower() for kw in ['docker', 'kubectl', 'helm', 'kubectl'])):
                inline_code_counts['k8s'] += 1
            elif any((kw in code.lower() for kw in ['SELECT', 'INSERT', 'FROM', 'WHERE'])):
                inline_code_counts['sql'] += 1
            else:
                inline_code_counts['other'] += 1
    return {'fenced_languages': dict(lang_counts.most_common(10)), 'inline_code_types': dict(inline_code_counts.most_common(10))}

def run_audit(input_path: Path=None, sample_size: int=None) -> dict:
    """Run all audits and return consolidated results."""
    if input_path is None:
        input_path = Paths.processed_dir() / 'clean.jsonl'
    logger.info(f'Loading documents from {input_path}...')
    docs = load_docs(input_path)
    if sample_size and sample_size < len(docs):
        logger.info(f'Sampling {sample_size} documents for audit...')
        import random
        docs = random.sample(docs, sample_size)
    logger.info(f'Auditing {len(docs)} documents...')
    return {'metadata': audit_metadata(docs), 'content_quality': audit_content_quality(docs), 'code_blocks': audit_code_blocks(docs), 'code_patterns': audit_code_patterns(docs), 'summary': {'total_docs': len(docs), 'docs_with_code': audit_code_blocks(docs)['total_with_code'], 'code_preservation_rate': round(audit_code_blocks(docs)['complete_blocks'] / max(1, audit_code_blocks(docs)['total_with_code']) * 100, 1) if audit_code_blocks(docs)['total_with_code'] > 0 else 0}}

def print_report(results: dict):
    """Print a human-readable audit report."""
    print('\n' + '=' * 70)
    print('🔍 DATA QUALITY AUDIT REPORT')
    print('=' * 70)
    s = results['summary']
    print(f'\n📊 Summary:')
    print(f"  Total documents: {s['total_docs']}")
    print(f"  Documents with code: {s['docs_with_code']} ({s['docs_with_code'] / s['total_docs'] * 100:.1f}%)")
    print(f"  Code block preservation rate: {s['code_preservation_rate']}%")
    cb = results['code_blocks']
    print(f'\n🧩 Code Block Integrity:')
    print(f"  Total blocks found: {cb['total_with_code']}")
    print(f"  Complete blocks: {cb['complete_blocks']} ({cb['complete_blocks'] / max(1, cb['total_with_code']) * 100:.1f}%)")
    print(f"  Truncated blocks: {cb['truncated_blocks']}")
    print(f"  Malformed blocks: {cb['malformed_blocks']}")
    if cb['by_course']:
        print(f'\n  By course:')
        for course, stats in sorted(cb['by_course'].items()):
            if stats['total'] > 0:
                rate = stats['complete'] / stats['total'] * 100
                print(f"    {course}: {stats['complete']}/{stats['total']} complete ({rate:.1f}%)")
    cq = results['content_quality']
    print(f'\n⚠️  Content Quality Issues:')
    print(f"  Empty answers: {cq['empty_answers']}")
    print(f"  Placeholder answers: {cq['placeholder_answers']}")
    print(f"  Encoding issues: {cq['encoding_issues']}")
    print(f"  Excessive whitespace: {cq['excessive_whitespace']}")
    if cq['unusual_chars']:
        print(f"  Unusual characters: {dict(cq['unusual_chars'])}")
    if cq['answer_length_outliers']:
        print(f"  Length outliers: {len(cq['answer_length_outliers'])} (review manually)")
    md = results['metadata']
    print(f'\n🏷️  Metadata Integrity:')
    print(f"  Missing IDs: {md['missing_ids']}")
    print(f"  Duplicate IDs: {md['duplicate_ids']}")
    print(f"  Missing courses: {md['missing_courses']}")
    print(f"  Missing sections: {md['missing_sections']}")
    if md['unknown_courses']:
        print(f"  Unknown courses: {md['unknown_courses']}")
    cp = results['code_patterns']
    if cp['fenced_languages']:
        print(f'\n💻 Code Languages (fenced blocks):')
        for lang, count in cp['fenced_languages'].items():
            print(f"  {lang or '(no lang)'}: {count}")
    print('\n' + '=' * 70)
    if s['code_preservation_rate'] >= 90 and cq['empty_answers'] == 0:
        print('✅ Data quality looks GOOD — ready for experiments')
    else:
        print('⚠️  Some issues detected — review before proceeding')
    print('=' * 70 + '\n')

def main():
    parser = argparse.ArgumentParser(description='Deep data quality audit')
    parser.add_argument('--input', type=Path, default=None, help='Input JSONL path')
    parser.add_argument('--sample', type=int, default=None, help='Sample size for faster audit')
    parser.add_argument('--json', action='store_true', help='Output JSON instead of printing')
    args = parser.parse_args()
    results = run_audit(args.input, args.sample)
    if args.json:
        import json
        from rag_pipeline.core.paths import Paths
        output = Paths.experiments_dir() / 'data_audit.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f'Audit results saved to {output}')
    else:
        print_report(results)
if __name__ == '__main__':
    main()
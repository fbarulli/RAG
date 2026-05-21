"""



Public Functions for Topic Assignment Merging and Reclassification:

def reclassify_other(category: str, question: str) -> str:
    Reclassify OTHER using signal matching. Priority: ERROR > ADMIN > CONCEPT.
    I/O: category (str), question (str) -> str

def merge(exp_dir: Path = EXP_DIR, output_path: Path = OUTPUT_PATH) -> None:
    Merge per-model topic assignment files and apply NER reclassification to OTHER questions using rule-based signals.
    I/O: exp_dir (Path), output_path (Path) -> None







_topic_merge.py
===============
Merge per-model topic assignment files and apply NER reclassification
to OTHER questions using rule-based signals.

Usage
-----
    cd /workspaces/LLM && uv run python -m rag_pipeline.p02_eda._topic_merge
"""
import json
from pathlib import Path
from rag_pipeline.logging import get_logger
logger = get_logger(__name__)
EXP_DIR = Path('rag_pipeline/p02_eda/experiments')
OUTPUT_PATH = EXP_DIR / 'topic_assignments_all.json'
_ERROR_SIGNALS = {'error', 'exception', 'failed', 'failure', 'warning', 'cannot', "can't", 'unable', 'not found', 'permission denied', 'attributeerror', 'valueerror', 'typeerror', 'importerror', 'modulenotfounderror', 'filenotfounderror', 'keyerror', 'oserror', 'runtimeerror', 'nameerror', 'traceback', 'convergencewarning', 'futurewarning', 'userwarning', 'deprecationwarning', 'timeout', 'refused', 'denied', 'crash', 'invalid', 'unrecognized', 'could not', 'no module', 'no such'}
_ADMIN_SIGNALS = {'certificate', 'homework', 'deadline', 'cohort', 'office hours', 'self-paced', 'graduate', 'leaderboard', 'peer review', 'capstone', 'lecture', 'video', 'live', 'recorded', 'session', 'form', 'confirmation', 'email', 'registration', 'enroll', 'books', 'resources', 'additional resources'}
_CONCEPT_SIGNALS = {'what is', 'why do', 'why does', 'what does', 'difference between', 'how does', 'explain', 'understanding', 'when to use', 'what are', 'how to choose', "what's the", 'why use', 'should i use'}

def reclassify_other(category: str, question: str) -> str:
    """Reclassify OTHER using signal matching. Priority: ERROR > ADMIN > CONCEPT."""
    if category != 'OTHER':
        return category
    q = question.lower()
    if any((sig in q for sig in _ERROR_SIGNALS)):
        return 'ERROR'
    if any((sig in q for sig in _ADMIN_SIGNALS)):
        return 'ADMIN'
    if any((sig in q for sig in _CONCEPT_SIGNALS)):
        return 'CONCEPT'
    return 'OTHER'

def merge(exp_dir: Path=EXP_DIR, output_path: Path=OUTPUT_PATH) -> None:
    merged = {'models': [], 'results': {}}
    files = sorted((f for f in exp_dir.glob('topic_assignments_*.json') if f.name != 'topic_assignments_all.json' and '_validated' not in f.name))
    if not files:
        raise FileNotFoundError(f'No topic_assignments_*.json files found in {exp_dir}')
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        model = data['metadata']['model']
        reclassified = 0
        for a in data['assignments']:
            original = a.get('ner_category', 'OTHER')
            updated = reclassify_other(original, a['question'])
            if updated != original:
                a['ner_category'] = updated
                reclassified += 1
        logger.info(f"[merge] {model}: {len(data['assignments'])} assignments, {reclassified} OTHER reclassified")
        merged['models'].append(model)
        merged['results'][model] = data
    with open(output_path, 'w') as fh:
        json.dump(merged, fh, indent=2)
    logger.info(f"[merge] {len(merged['models'])} models -> {output_path}")
    print(f"\n{'=' * 60}")
    print(f"Merged {len(merged['models'])} models into {output_path.name}")
    print(f"{'=' * 60}")
    for model in merged['models']:
        assignments = merged['results'][model]['assignments']
        from collections import Counter
        cats = Counter((a.get('ner_category', 'MISSING') for a in assignments))
        outliers = sum((1 for a in assignments if a['topic'] == -1))
        print(f'\n  {model}')
        print(f'    outliers: {outliers} ({outliers / len(assignments):.1%})')
        for cat, count in cats.most_common():
            print(f'    {cat:10s} {count:4d} ({count / len(assignments):.1%})')
if __name__ == '__main__':
    merge()
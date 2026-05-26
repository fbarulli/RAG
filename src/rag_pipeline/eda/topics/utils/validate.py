"""validate.py
Schema-aware dataset validation using FAQDocument.
"""
from collections import Counter
from rag_pipeline.core.schemas import FAQDocument
from typing import Union

def validate_dataset(docs: list[Union[dict, FAQDocument]], valid_courses: set[str] | None=None, min_answer_len: int=10) -> list[str]:
    """Run sanity checks on FAQ dataset. Returns list of issue strings."""
    issues = []
    if valid_courses is None:
        valid_courses = {'llm-zoomcamp', 'mlops-zoomcamp', 'de-zoomcamp', 'ml-zoomcamp', 'data-engineering-zoomcamp', 'machine-learning-zoomcamp'}
    if not docs:
        return ['Dataset is empty']
    for i, d in enumerate(docs):
        try:
            if isinstance(d, dict):
                doc = FAQDocument.from_dict(d)
            elif isinstance(d, FAQDocument):
                doc = d
            else:
                issues.append(f'Doc {i}: invalid type {type(d)}')
                continue
            if not doc.id or not doc.question or (not doc.answer) or (not doc.course):
                issues.append(f'Doc {i}: empty required field')
            if len(doc.answer.strip()) < min_answer_len:
                issues.append(f'Doc {i}: answer too short ({len(doc.answer)} chars)')
        except Exception as e:
            issues.append(f'Doc {i}: schema error — {e}')
    courses = {d.course if isinstance(d, FAQDocument) else d.get('course') for d in docs}
    unknown = courses - valid_courses
    if unknown:
        issues.append(f'Unknown courses: {sorted(unknown)[:10]}')
    ids = [d.id if isinstance(d, FAQDocument) else d['id'] for d in docs]
    dupes = [id_ for id_, cnt in Counter(ids).items() if cnt > 1]
    if dupes:
        issues.append(f'Duplicate IDs ({len(dupes)} total): {dupes[:5]}')
    artifacts = sum((1 for d in docs if '<{' in (d.answer if isinstance(d, FAQDocument) else d['answer']) or '<IMAGE' in (d.answer if isinstance(d, FAQDocument) else d['answer'])))
    if artifacts:
        issues.append(f'Uncleaned artifacts (<{{, IMAGE): {artifacts}')
    return issues

def assert_valid(docs: list[Union[dict, FAQDocument]], **kwargs) -> None:
    """Raise ValueError if validation fails."""
    issues = validate_dataset(docs, **kwargs)
    if issues:
        raise ValueError('Dataset validation failed:\n  • ' + '\n  • '.join(issues))
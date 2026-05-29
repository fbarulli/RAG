"""
data_cleaning/parse.py
======================
Reads raw markdown files from data/raw/ and extracts structured documents.
Cleans markdown formatting from answer text.

Input:  data_cleaning/data/raw/<course>/<section>/*.md
Output: data_cleaning/data/processed/parsed.jsonl
        One JSON object per line: {id, question, answer, course, section}

Run:    uv run python data_cleaning/parse.py
"""
import argparse
import json
import os
import re
import tempfile
from typing import Dict, Generator, Optional, Tuple
import yaml
from rag_pipeline.core.schemas import FAQDocument
from rag_pipeline.core.paths import Paths
DEFAULT_RAW_DIR = str(Paths.raw_dir())
DEFAULT_OUTPUT = str(Paths.output_file('parse'))
_NEWLINE_COLLAPSE_THRESHOLD = 3
SKIP_IDS: Dict[str, str] = {'841966c903': 'Answer was only a URL (link to Prefect FAQ); no useful content.'}
_RE_IMAGE_PLACEHOLDER = re.compile('<\\{\\s*IMAGE:[^}]+\\s*\\}>')
_RE_HEADERS = re.compile('^#{1,6}\\s+', re.MULTILINE)
_RE_BOLD = re.compile('\\*\\*(.+?)\\*\\*', re.DOTALL)
_RE_ITALIC = re.compile('\\*(.+?)\\*', re.DOTALL)
_RE_INLINE_CODE = re.compile('`([^`]+)`')
_RE_MD_LINK = re.compile('\\[([^\\]]+)\\]\\([^)]+\\)')
_RE_MD_IMAGE = re.compile('!\\[[^\\]]*\\]\\([^)]+\\)')
_RE_HTML_TAG = re.compile('<[^>]+>')
_RE_JINJA_BLOCK = re.compile('\\{[%{#][^}]*[%}#]\\}')
_RE_JINJA_VAR = re.compile('\\{\\{[^}]+\\}\\}')
_RE_EXCESS_NEWLINES = re.compile('\\n{' + str(_NEWLINE_COLLAPSE_THRESHOLD) + ',}')
_RE_FENCED_CODE = re.compile('```(\\w+)\\n(.*?)```|```\\n?(.*?)```', re.DOTALL)

def clean_answer(text: str) -> str:
    """Remove markdown formatting while preserving fenced code blocks with language tags."""
    code_blocks = []

    def protect_fenced_code(match):
        if match.group(1):
            lang = match.group(1)
            body = match.group(2).strip()
        else:
            lang = ''
            body = (match.group(3) or '').strip()
        code_blocks.append((lang, body))
        return f'__FENCED_CODE_{len(code_blocks) - 1}__'
    text = _RE_FENCED_CODE.sub(protect_fenced_code, text)
    text = _RE_IMAGE_PLACEHOLDER.sub('', text)
    text = _RE_HEADERS.sub('', text)
    text = _RE_MD_IMAGE.sub('', text)
    text = _RE_BOLD.sub('\\1', text)
    text = _RE_ITALIC.sub('\\1', text)
    text = _RE_INLINE_CODE.sub('\\1', text)
    text = _RE_MD_LINK.sub('\\1', text)
    text = _RE_HTML_TAG.sub('', text)
    text = _RE_JINJA_BLOCK.sub('', text)
    text = _RE_JINJA_VAR.sub('', text)
    text = _RE_EXCESS_NEWLINES.sub('\n\n', text)
    for i, (lang, body) in enumerate(code_blocks):
        lang_str = f'{lang}\n' if lang else ''
        text = text.replace(f'__FENCED_CODE_{i}__', f'```{lang_str}{body}\n```')
    return text.strip()

def parse_file(filepath: str, course: str, section: str) -> Tuple[Optional[FAQDocument], Optional[str]]:
    """Parse a single markdown file into a structured document.

    Returns:
        (document, None)        on success.
        (None, reason_string)   on any failure, with a human-readable reason.
    """
    with open(filepath, encoding='utf-8') as f:
        content = f.read()
    if not content.startswith('---'):
        return (None, 'No YAML frontmatter fence')
    parts = content.split('---', 2)
    if len(parts) < 3:
        return (None, 'Malformed frontmatter (missing closing fence)')
    frontmatter_str, body = (parts[1], parts[2])
    try:
        frontmatter = yaml.safe_load(frontmatter_str) or {}
    except yaml.YAMLError as exc:
        return (None, f'YAML parse error: {exc}')
    doc_id = frontmatter.get('id', '')
    question = frontmatter.get('question', '')
    if not doc_id:
        return (None, "Missing 'id' field in frontmatter")
    if not isinstance(doc_id, str):
        return (None, f"'id' must be a string, got {type(doc_id).__name__}: {doc_id!r} — quote it in frontmatter")
    if not question:
        return (None, "Missing 'question' field in frontmatter")
    if not isinstance(question, str):
        return (None, f"'question' must be a string, got {type(question).__name__}: {question!r} — quote it in frontmatter")
    return (FAQDocument(id=doc_id, question=question.strip(), answer=clean_answer(body), course=course, section=section), None)

def walk_raw_dir(raw_dir: str) -> Generator[Tuple[str, str, str], None, None]:
    """Yield (filepath, course, section) for every .md file under raw_dir."""
    for course in sorted(os.listdir(raw_dir)):
        course_path = os.path.join(raw_dir, course)
        if not os.path.isdir(course_path):
            continue
        for section in sorted(os.listdir(course_path)):
            section_path = os.path.join(course_path, section)
            if not os.path.isdir(section_path):
                continue
            for filename in sorted(os.listdir(section_path)):
                if filename.endswith('.md'):
                    yield (os.path.join(section_path, filename), course, section)

def main(raw_dir: str=DEFAULT_RAW_DIR, output: str=DEFAULT_OUTPUT) -> None:
    os.makedirs(os.path.dirname(output), exist_ok=True)
    total = skipped = failed = 0
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(output), suffix='.tmp')
    try:
        with os.fdopen(tmp_fd, 'w') as out:
            for filepath, course, section in walk_raw_dir(raw_dir):
                doc, reason = parse_file(filepath, course, section)
                if doc is None:
                    print(f'  Skip (parse failure) {filepath}: {reason}')
                    failed += 1
                    continue
                if doc.id in SKIP_IDS:
                    print(f'  Skip (known) {doc.id}: {SKIP_IDS[doc.id]}')
                    skipped += 1
                    continue
                out.write(doc.to_json() + '\n')
                total += 1
        os.replace(tmp_path, output)
    except Exception:
        os.unlink(tmp_path)
        raise
    print(f'\nDone: {total} written, {skipped} intentionally skipped, {failed} parse failures → {output}')

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Parse raw markdown Q&A files into a JSONL dataset.')
    parser.add_argument('--raw-dir', default=DEFAULT_RAW_DIR, help=f'Root directory of raw markdown files (default: {DEFAULT_RAW_DIR})')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help=f'Output JSONL file path (default: {DEFAULT_OUTPUT})')
    return parser
if __name__ == '__main__':
    args = _build_parser().parse_args()
    main(raw_dir=args.raw_dir, output=args.output)
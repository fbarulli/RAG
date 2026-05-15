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
import os
import re
import json
import yaml
from typing import Dict, Optional

RAW_DIR = 'data_cleaning/data/raw'
OUTPUT = 'data_cleaning/data/processed/parsed.jsonl'

# Documents to skip (lost their content during cleaning)
SKIP_IDS = {
    '841966c903',  # "Where is the FAQ for Prefect questions?" - answer was just a link
}


def parse_frontmatter(content: str) -> Dict:
    """Extract YAML frontmatter from markdown content."""
    if not content.startswith('---'):
        return {}
    end = content.find('---', 3)
    if end == -1:
        return {}
    frontmatter_str = content[3:end].strip()
    try:
        return yaml.safe_load(frontmatter_str) or {}
    except yaml.YAMLError:
        return {}


def clean_answer(text: str) -> str:
    """Remove markdown formatting and special characters from answer text."""
    # Remove <{IMAGE:...}> placeholders
    text = re.sub(r'<\{\s*IMAGE:[^}]+\s*\}>', '', text)
    
    # Remove markdown headers (##, ###, etc.) - keep the text
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove bold/italic markers
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    
    # Remove inline code markers
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove markdown links, keep text: [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove markdown images: ![alt](url) → empty
    text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove Jinja2/macro syntax
    text = re.sub(r'\{[%{#][^}]*[%}#]\}', '', text)
    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def parse_file(filepath: str, course: str, section: str) -> Optional[Dict]:
    """Parse a single markdown file into a structured document."""
    with open(filepath) as f:
        content = f.read()

    if not content.startswith('---'):
        return None

    parts = content.split('---', 2)
    if len(parts) < 3:
        return None

    frontmatter = parse_frontmatter(content)
    answer = clean_answer(parts[2])

    doc_id = frontmatter.get('id', '')
    question = str(frontmatter.get('question', '')).strip()

    if not doc_id or not question:
        return None

    return {
        'id': doc_id,
        'question': question,
        'answer': answer,
        'course': course,
        'section': section,
    }


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    total = 0
    skipped = 0
    with open(OUTPUT, 'w') as out:
        for course in sorted(os.listdir(RAW_DIR)):
            course_path = os.path.join(RAW_DIR, course)
            if not os.path.isdir(course_path):
                continue

            for section in sorted(os.listdir(course_path)):
                section_path = os.path.join(course_path, section)
                if not os.path.isdir(section_path):
                    continue

                for filename in sorted(os.listdir(section_path)):
                    if not filename.endswith('.md'):
                        continue

                    filepath = os.path.join(section_path, filename)
                    doc = parse_file(filepath, course, section)

                    if doc:
                        if doc['id'] in SKIP_IDS:
                            skipped += 1
                            continue
                        out.write(json.dumps(doc) + '\n')
                        total += 1

    print(f'Parsed {total} documents ({skipped} skipped) → {OUTPUT}')


if __name__ == '__main__':
    # Install pyyaml if needed
    try:
        import yaml
    except ImportError:
        import subprocess
        subprocess.check_call(['uv', 'pip', 'install', 'pyyaml'])
        import yaml
    
    main()

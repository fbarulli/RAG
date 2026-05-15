"""
tests/inspect_comparisons.py
============================
Samples documents from raw/parsed data and compares them to verify
that code blocks and formatting are preserved correctly.

Usage: uv run python tests/inspect_comparisons.py [--count 10] [--clear]
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

def get_frontmatter_id(filepath):
    try:
        with open(filepath, encoding='utf-8') as f:
            content = f.read()
        match = re.search(r'---\n.*?id:\s*([a-f0-9]+)', content, re.DOTALL)
        return match.group(1) if match else None
    except Exception:
        return None

def get_body(filepath):
    try:
        with open(filepath, encoding='utf-8') as f:
            content = f.read()
        parts = content.split('---', 2)
        return parts[2].strip() if len(parts) > 2 else ""
    except Exception:
        return ""

def main():
    parser = argparse.ArgumentParser(description='Compare raw vs parsed FAQ data')
    parser.add_argument('--count', type=int, default=10, help='Number of documents to sample')
    parser.add_argument('--clear', action='store_true', help='Clear existing JSON before running')
    args = parser.parse_args()

    base_dir = Path('production_pipeline/p01_data_cleaning')
    raw_dir = base_dir / 'data' / 'raw'
    processed_dir = base_dir / 'data' / 'processed'
    output_file = Path('tests/sample_comparisons.json')

    if not raw_dir.exists():
        print(f"Error: Raw directory not found at {raw_dir}")
        sys.exit(1)

    # Load parsed data
    parsed_data = {}
    parsed_file = processed_dir / 'clean.jsonl'
    if not parsed_file.exists():
        parsed_file = processed_dir / 'parsed.jsonl'
    
    if parsed_file.exists():
        with open(parsed_file, encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    doc = json.loads(line)
                    parsed_data[doc['id']] = doc
                except json.JSONDecodeError:
                    continue
    else:
        print(f"Error: No parsed data found (clean.jsonl or parsed.jsonl)")
        sys.exit(1)

    print(f"Loaded {len(parsed_data)} parsed documents")

    # Load raw data
    raw_data = {}
    for dirpath, _, filenames in os.walk(raw_dir):
        for filename in filenames:
            if filename.endswith('.md'):
                filepath = os.path.join(dirpath, filename)
                doc_id = get_frontmatter_id(filepath)
                if doc_id:
                    raw_data[doc_id] = get_body(filepath)

    print(f"Loaded {len(raw_data)} raw documents")

    # Find common IDs
    common_ids = list(set(parsed_data.keys()) & set(raw_data.keys()))
    if not common_ids:
        print("Error: No matching documents found between raw and parsed data")
        sys.exit(1)

    # Sample
    count = min(args.count, len(common_ids))
    samples = random.sample(common_ids, count)
    print(f"Sampling {count} documents for comparison...")

    results = []
    stats = {"total_sampled": count, "ok": 0, "code_preserved": 0, "code_missing": 0}

    for qid in samples:
        r_body = raw_data[qid]
        p_doc = parsed_data[qid]
        p_ans = p_doc['answer']

        raw_has_code = '```' in r_body
        parsed_has_code = '```' in p_ans

        if raw_has_code and not parsed_has_code:
            status = "CODE MISSING"
            stats["code_missing"] += 1
        elif raw_has_code and parsed_has_code:
            status = "CODE PRESERVED"
            stats["code_preserved"] += 1
        else:
            status = "OK"
            stats["ok"] += 1

        # Only keep entries that show code behavior (modifications)
        if status != "OK":
            results.append({
                'id': qid,
                'course': p_doc['course'],
                'question': p_doc['question'],
                'status': status,
                'raw_body': r_body,
                'parsed_answer': p_ans,
                'raw_has_code': raw_has_code,
                'parsed_has_code': parsed_has_code
            })

    # Append to JSON
    if args.clear and output_file.exists():
        output_file.unlink()

    existing = []
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = []

    existing.extend(results)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"Results: {len(results)} entries with code activity written to {output_file}")
    print(f"Stats: {stats}")
    print(f"Total entries in file: {len(existing)}")

if __name__ == '__main__':
    main()
"""
data_cleaning/analyze_markdown.py
=================================
Analyzes all raw markdown files to:
- Identify markdown features (bold, links, code blocks, etc.)
- Track special characters
- Generate test cases for the cleaner
- Output: data_cleaning/data/analysis/ (summary.json + test_cases.json)

Run: uv run python data_cleaning/analyze_markdown.py
"""

import os
import json
import yaml
import hashlib
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

# --- Markdown Features to Detect ---
FEATURES = {
    # Formatting
    "bold": r"\*\*.+?\*\*|__.+?__",
    "italic": r"\*.+?\*|_[^_]+_",
    "strikethrough": r"~~.+?~~",
    # Code
    "inline_code": r"`[^`]+`",
    "code_block_fenced": r"```([^\n]*)\n([\s\S]*?)```|```\n?([\s\S]*?)```",
    "code_block_indented": r"^\s{4}.+$",
    # Links & Images
    "link": r"\[([^\]]+)\]\([^)]+\)",
    "image": r"!\[([^\]]*)\]\([^)]+\)",
    "image_placeholder": r"<\{ *IMAGE:[^}]+ *\}>",
    # HTML
    "html_tag": r"<[^>]+>",
    "html_comment": r"<!--[\s\S]*?-->",
    # Jinja2
    "jinja_block": r"\{[%{#][^}]*[%}#]\}",
    "jinja_var": r"\{\{[^}]+\}\}",
    # Structure
    "header": r"^#{1,6}\s+.+$",
    "blockquote": r"^\s*>.*$",
    "list_ul": r"^\s*[-*+]\s+.+$",
    "list_ol": r"^\s*\d+\.\s+.+$",
    "task_list": r"^\s*[-*+]\s+\[[xX ]]\s+.+$",
    "horizontal_rule": r"^-{3,}$|^\*{3,}$|^_{3,}$",
    "table": r"^\s*\|.+\|$",
    # Other
    "escape": r"\\\S",
    "footnote": r"\[\^[^\]]+\]",
}

# --- Special Characters to Track ---
SPECIAL_CHARS = set("*_`[]()!<>{}#+-=|~@$%^&\\")

def analyze_file(filepath: str) -> Dict[str, Any]:
    """Analyze a single markdown file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse frontmatter
    has_frontmatter = content.lstrip().startswith("---")
    frontmatter = {}
    body = content
    if has_frontmatter:
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1]) or {}
                body = parts[2]
            except yaml.YAMLError:
                pass

    # Detect features
    features = defaultdict(list)
    for name, pattern in FEATURES.items():
        import re
        matches = re.finditer(pattern, body, re.MULTILINE | re.DOTALL)
        for match in matches:
            features[name].append(match.group(0))

    # Count special characters
    special_chars = defaultdict(int)
    for char in SPECIAL_CHARS:
        special_chars[char] = body.count(char)

    # Count non-ASCII
    non_ascii = defaultdict(int)
    for char in body:
        if ord(char) > 127:
            non_ascii[char] += 1

    return {
        "filepath": filepath,
        "course": Path(filepath).parent.parent.name,
        "section": Path(filepath).parent.name,
        "id": frontmatter.get("id", Path(filepath).stem),
        "question": frontmatter.get("question", ""),
        "has_frontmatter": has_frontmatter,
        "body_length": len(body),
        "features": dict(features),
        "special_chars": dict(special_chars),
        "non_ascii": dict(non_ascii),
    }

def analyze_directory(raw_dir: str) -> Dict:
    """Analyze all markdown files in a directory."""
    all_files = []
    feature_stats = defaultdict(lambda: {"count": 0, "files": []})
    char_stats = defaultdict(int)
    non_ascii_stats = defaultdict(int)

    # Collect all .md files
    files_to_analyze = []
    for root, _, files in os.walk(raw_dir):
        for filename in files:
            if filename.endswith(".md"):
                files_to_analyze.append(os.path.join(root, filename))

    # Process in parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(analyze_file, files_to_analyze))

    for result in results:
        all_files.append(result)
        # Update feature stats
        for feature, matches in result["features"].items():
            feature_stats[feature]["count"] += len(matches)
            if len(feature_stats[feature]["files"]) < 5:
                feature_stats[feature]["files"].append({
                    "id": result["id"],
                    "filepath": result["filepath"],
                    "question": result["question"][:80],
                    "example": matches[0][:100] if matches else "",
                })
        # Update char stats
        for char, count in result["special_chars"].items():
            char_stats[char] += count
        # Update non-ASCII stats
        for char, count in result["non_ascii"].items():
            non_ascii_stats[char] += count

    return {
        "files": all_files,
        "feature_stats": dict(feature_stats),
        "char_stats": dict(char_stats),
        "non_ascii_stats": dict(non_ascii_stats),
        "total_files": len(all_files),
    }

def save_analysis(analysis: Dict, output_dir: str = "data_cleaning/data/analysis"):
    """Save analysis results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save summary
    summary = {
        "total_files": analysis["total_files"],
        "feature_stats": {
            f: {
                "count": d["count"],
                "file_count": len(d["files"]),
                "examples": d["files"],
            }
            for f, d in analysis["feature_stats"].items()
        },
        "char_stats": analysis["char_stats"],
        "non_ascii_stats": {
            c: n for c, n in sorted(
                analysis["non_ascii_stats"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:20]  # Top 20 non-ASCII
        },
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save test cases (for pytest)
    test_cases = []
    for feature, data in analysis["feature_stats"].items():
        if data["count"] > 0:
            for ex in data["files"]:
                test_cases.append({
                    "feature": feature,
                    "id": ex["id"],
                    "filepath": ex["filepath"],
                    "input": ex["example"],
                    "expected": None,  # Fill this in manually
                })

    with open(os.path.join(output_dir, "test_cases.json"), "w") as f:
        json.dump(test_cases, f, indent=2)

    print(f"✅ Analysis saved to {output_dir}")

if __name__ == "__main__":
    from rag_pipeline.cleaning.core.paths import Paths
    raw_dir = str(Paths.raw_dir())
    print(f"🔍 Analyzing files in {raw_dir}...")
    analysis = analyze_directory(raw_dir)
    save_analysis(analysis)

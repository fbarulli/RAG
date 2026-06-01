"""
data_cleaning/parse.py
======================
Reads raw markdown files from data/raw/ and extracts structured documents.
Uses markdown-it-py for robust markdown parsing + data-driven analysis for validation.

Input:  data_cleaning/data/raw/<course>/<section>/*.md
Output: data_cleaning/data/processed/parsed.jsonl
Format: One JSON object per line: {id, question, answer, course, section}

Run: uv run python data_cleaning/parse.py
Dependencies: uv add markdown-it-py
"""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple

import yaml
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode

from rag_pipeline.cleaning.core.schemas import FAQDocument
from rag_pipeline.cleaning.core.paths import Paths

# --- Constants ---
DEFAULT_RAW_DIR = str(Paths.raw_dir())
DEFAULT_OUTPUT = str(Paths.output_file("parse"))
_NEWLINE_COLLAPSE_THRESHOLD = 3

# IDs to skip with reasons
SKIP_IDS: Dict[str, str] = {
    "841966c903": "Answer was only a URL (link to Prefect FAQ); no useful content."
}

# --- Markdown Parser Setup ---
def _get_markdown_parser() -> MarkdownIt:
    """Configure markdown-it with plugins for all features we need to handle."""
    md = MarkdownIt(
        "commonmark",
        {
            "html": False,  # Disable HTML parsing
            "linkify": False,  # Don't auto-convert URLs to links
            "typographer": False,  # Don't replace quotes, etc.
        },
    )
    # Enable additional plugins for features in our data
    md.enable(["table", "strikethrough", "task_lists"])
    return md

# --- Cleaning Function (Hybrid: Parser + Regex Fallbacks) ---
def clean_answer(text: str) -> str:
    """
    Remove markdown formatting while preserving:
    - Fenced code blocks (with language tags)
    - Inline code (as plain text, no backticks)
    - All other text content (without formatting)
    """
    if not text or not text.strip():
        return ""

    md = _get_markdown_parser()
    tokens = md.parse(text)
    root = SyntaxTreeNode(tokens)

    def _clean_node(node: SyntaxTreeNode) -> str:
        """Recursively clean a node and its children."""
        if node.type == "text":
            return node.content

        # === PRESERVE: Code Blocks ===
        elif node.type == "code_block":
            return f"```\n{node.content}\n```"
        elif node.type == "fence":
            lang = node.info.strip() if node.info else ""
            return f"```{lang}\n{node.content}\n```"

        # === STRIP FORMATTING: Keep text, remove markup ===
        elif node.type == "code_inline":
            return node.content  # Remove backticks, keep content
        elif node.type in ["em", "strong", "strikethrough"]:
            return "".join(_clean_node(child) for child in node.children)
        elif node.type == "link":
            return "".join(_clean_node(child) for child in node.children)
        elif node.type == "image":
            return ""  # Remove images entirely

        # === STRUCTURAL ELEMENTS: Keep text content, remove structure ===
        elif node.type in [
            "paragraph",
            "heading",
            "blockquote",
            "list_item",
            "bullet_list",
            "ordered_list",
            "table",
            "table_row",
            "table_cell",
        ]:
            return "".join(_clean_node(child) for child in node.children)

        # === REMOVE ENTIRELY ===
        elif node.type in ["hr", "html_block", "html_inline"]:
            return ""

        # === FALLBACK: Try to extract text from children ===
        else:
            if node.children:
                return "".join(_clean_node(child) for child in node.children)
            return ""

    cleaned = _clean_node(root)

    # Clean up excessive whitespace (like original)
    cleaned = re.sub(rf"\n{{{_NEWLINE_COLLAPSE_THRESHOLD},}}", "\n\n", cleaned)
    return cleaned.strip()

# --- Frontmatter Parsing (Unchanged) ---
def parse_file(
    filepath: str, course: str, section: str
) -> Tuple[Optional[FAQDocument], Optional[str]]:
    """Parse a single markdown file into a structured document."""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # Robust frontmatter detection
    stripped_content = content.lstrip()
    if not stripped_content.startswith("---"):
        return (None, "No YAML frontmatter fence (must start with '---')")

    parts = content.split("---", 2)
    if len(parts) < 3:
        return (None, "Malformed frontmatter (missing closing '---')")
    if parts[0].strip() != "":
        return (None, "Malformed frontmatter (content before opening fence)")

    frontmatter_str = parts[1].strip()
    body = parts[2]

    try:
        frontmatter = yaml.safe_load(frontmatter_str) or {}
    except yaml.YAMLError as exc:
        return (None, f"YAML parse error: {exc}")

    doc_id = frontmatter.get("id")
    question = frontmatter.get("question")

    if not doc_id:
        return (None, "Missing 'id' field in frontmatter")
    if not isinstance(doc_id, str):
        return (
            None,
            f"'id' must be a string, got {type(doc_id).__name__}: {doc_id!r} — quote it in frontmatter",
        )
    if not question:
        return (None, "Missing 'question' field in frontmatter")
    if not isinstance(question, str):
        return (
            None,
            f"'question' must be a string, got {type(question).__name__}: {question!r} — quote it in frontmatter",
        )

    answer = clean_answer(body)
    if not answer:
        return (None, "Answer is empty after cleaning")

    return (
        FAQDocument(
            id=doc_id,
            question=question.strip(),
            answer=answer,
            course=course,
            section=section,
        ),
        None,
    )

# --- File Walking (Optimized) ---
def walk_raw_dir(raw_dir: str) -> Generator[Tuple[str, str, str], None, None]:
    """Yield (filepath, course, section) for every .md file under raw_dir."""
    for root, _, files in os.walk(raw_dir):
        for filename in sorted(files):
            if filename.endswith(".md"):
                rel_path = os.path.relpath(root, raw_dir)
                path_parts = rel_path.split(os.sep)
                if len(path_parts) == 2:  # Only course/section paths
                    course, section = path_parts
                    yield (os.path.join(root, filename), course, section)

# --- Main Function (Unchanged) ---
def main(raw_dir: str = DEFAULT_RAW_DIR, output: str = DEFAULT_OUTPUT) -> None:
    output_dir = Path(output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    total = skipped = failed = 0
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(output_dir), suffix=".tmp")

    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as out:
            for filepath, course, section in walk_raw_dir(raw_dir):
                doc, reason = parse_file(filepath, course, section)
                if doc is None:
                    print(f"❌ SKIP (parse failure) {filepath}: {reason}")
                    failed += 1
                    continue
                if doc.id in SKIP_IDS:
                    print(f"⚠️ SKIP (known) {doc.id}: {SKIP_IDS[doc.id]}")
                    skipped += 1
                    continue
                out.write(doc.to_json() + "\n")
                total += 1
        os.replace(tmp_path, output)
    except Exception:
        os.unlink(tmp_path)
        raise

    print(
        f"\n✅ Done: {total} written, {skipped} intentionally skipped, {failed} parse failures → {output}"
    )

# --- CLI (Unchanged) ---
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse raw markdown Q&A files into a JSONL dataset."
    )
    parser.add_argument(
        "--raw-dir", default=DEFAULT_RAW_DIR, help=f"Root directory of raw markdown files (default: {DEFAULT_RAW_DIR})"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help=f"Output JSONL file path (default: {DEFAULT_OUTPUT})"
    )
    return parser

if __name__ == "__main__":
    args = _build_parser().parse_args()
    main(raw_dir=args.raw_dir, output=args.output)

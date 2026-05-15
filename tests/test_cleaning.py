"""tests/test_cleaning.py"""
import pytest
import sys
from pathlib import Path

# Add project root to path to import local module
sys.path.insert(0, str(Path(__file__).parent.parent))
from production_pipeline.p01_data_cleaning.p02_parse import clean_answer

def test_inline_code_stripped():
    assert clean_answer("Use `pip install` for setup.") == "Use pip install for setup."

def test_fenced_block_preserved():
    inp = "Run this:\n```python\nprint('hi')\n```"
    out = clean_answer(inp)
    assert "```python\nprint('hi')```" in out

def test_no_sentinel_leakage():
    """Ensure placeholders are replaced even in multi-block docs."""
    inp = "Block 1:\n```bash\necho 1\n```\nBlock 2:\n```bash\necho 2\n```"
    out = clean_answer(inp)
    assert "__FENCED_CODE_" not in out
    assert "```bash\necho 1```" in out
    assert "```bash\necho 2```" in out

def test_edge_case_no_newline_after_tag():
    """Handles ```python code... where there's no newline after 'python'."""
    inp = "```python print('hi')```"
    out = clean_answer(inp)
    # Should preserve the block content, even if formatting is weird
    assert "```python\nprint('hi')```" in out or "```python print('hi')```" in out

def test_round_trip_metadata():
    """Ensure headers and links are stripped but code remains."""
    inp = "### Header\n**Bold**\n`code`\nLink: [Click](url)\n```bash\nrun\n```"
    out = clean_answer(inp)
    assert "###" not in out
    assert "**" not in out
    assert "[Click]" not in out
    assert "```bash\nrun```" in out
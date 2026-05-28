"""tests/test_cleaning.py"""
import pytest
from pathlib import Path
from rag_pipeline.cleaning.parse import clean_answer

def test_inline_code_stripped():
    assert clean_answer('Use `pip install` for setup.') == 'Use pip install for setup.'

def test_fenced_block_preserved():
    inp = "Run this:\n```python\nprint('hi')\n```"
    out = clean_answer(inp)
    assert "```python\nprint('hi')\n```" in out

def test_no_sentinel_leakage():
    inp = 'Block 1:\n```bash\necho 1\n```\nBlock 2:\n```bash\necho 2\n```'
    out = clean_answer(inp)
    assert '__FENCED_CODE_' not in out
    assert '```bash\necho 1\n```' in out
    assert '```bash\necho 2\n```' in out

def test_edge_case_no_newline_after_tag():
    inp = "```python print('hi')```"
    out = clean_answer(inp)
    assert "```python print('hi')\n```" in out

def test_round_trip_metadata():
    inp = '### Header\n**Bold**\n`code`\nLink: [Click](url)\n```bash\nrun\n```'
    out = clean_answer(inp)
    assert '###' not in out
    assert '**' not in out
    assert '[Click]' not in out
    assert '```bash\nrun\n```' in out
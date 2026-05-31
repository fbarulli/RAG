# rag_pipeline/core/io.py
"""Centralised I/O utilities — use these instead of raw open/json.dump."""
import json
from pathlib import Path
from typing import Any


def atomic_json_write(path: Path, data: Any, **kwargs) -> None:
    """Write JSON atomically: write to .tmp, validate, then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    text = json.dumps(data, **kwargs)
    json.loads(text)  # validate before touching the real file
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

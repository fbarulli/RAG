"""
_benchmark_loader.py
====================
Load test data, topic assignments, and retrieval configs for benchmarking.
"""
import json
from pathlib import Path

def load_test_set(path: Path) -> list[dict]:
    """Load holdout test queries with expected document IDs."""
    if not path.exists():
        raise FileNotFoundError(f"Test set not found: {path}")
    tests = []
    required = {"id", "question", "answer", "course"}
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip(): continue
            try: doc = json.loads(line)
            except json.JSONDecodeError: continue
            if not required.issubset(doc.keys()): continue
            tests.append({
                "query_id": doc["id"],
                "query": doc["question"],
                "expected_id": doc.get("expected_id", doc["id"]),
                "course": doc["course"],
                "section": doc.get("section", ""),
                "answer": doc["answer"],
            })
    return tests

def load_topic_assignments(path: Path) -> dict[str, dict]:
    """Load topic assignments indexed by document ID."""
    if not path.exists():
        raise FileNotFoundError(f"Topic assignments not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {a["id"]: a for a in data.get("assignments", [])}

def load_configs(path: Path) -> dict:
    """Load retrieval configuration presets."""
    if not path.exists():
        raise FileNotFoundError(f"Retrieval configs not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

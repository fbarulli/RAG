
"""
_benchmark_loader.py
====================
Load test data, topic assignments, and retrieval configs for benchmarking.

Single responsibility: I/O for benchmark inputs.
No metric computation, no retrieval logic, no reporting.

Functions:
    load_test_set(path: Path) -> list[dict]
    load_topic_assignments(path: Path) -> dict[str, dict]
    load_configs(path: Path) -> dict
"""
import json
from pathlib import Path

from rag_pipeline.logging import get_logger

logger = get_logger(__name__)


def load_test_set(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Test set not found: {path}")

    tests = []
    required = {"id", "question", "answer", "course"}

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Test set line {line_num}: JSON error: {e}")
                continue

            missing = required - doc.keys()
            if missing:
                logger.warning(f"Test set line {line_num}: Missing fields {missing}")
                continue

            tests.append({
                "query_id": doc["id"],
                "query": doc["question"],
                "expected_id": doc.get("expected_id", doc["id"]),
                "course": doc["course"],
                "section": doc.get("section", ""),
                "answer": doc["answer"],
            })

    logger.info(f"Loaded {len(tests)} test queries from {path}")
    return tests


def load_topic_assignments(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Topic assignments not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    assignments = {a["id"]: a for a in data.get("assignments", [])}
    logger.info(f"Loaded {len(assignments)} topic assignments from {path}")
    return assignments


def load_configs(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Retrieval configs not found: {path}")

    with open(path, encoding="utf-8") as f:
        configs = json.load(f)

    logger.info(f"Loaded {len(configs)} retrieval configs from {path}")
    return configs

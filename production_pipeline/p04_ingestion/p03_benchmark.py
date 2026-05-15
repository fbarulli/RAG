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


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_test_set(path: Path) -> list[dict]:
    """
    Load holdout test queries with expected document IDs.

    Args:
        path: Path to test.jsonl file

    Returns:
        List of dicts with:
            - query_id: unique identifier for the query
            - query: the question text
            - expected_id: the document ID that should be retrieved (falls back to query_id if not explicit)
            - course: course name for filtering
            - answer: reference answer for quality checks

    Note: If a test document has an explicit "expected_id" field, it is used.
          Otherwise, "id" is used as both query_id and expected_id (common when
          test queries are the original FAQ questions themselves).
    """
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
    """
    Load topic assignments indexed by document ID.

    Args:
        path: Path to topic_assignments.json

    Returns:
        Dict mapping doc_id -> {topic, subtopic, subtopic_keywords, ...}
    """
    if not path.exists():
        raise FileNotFoundError(f"Topic assignments not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    assignments = {a["id"]: a for a in data.get("assignments", [])}
    logger.info(f"Loaded {len(assignments)} topic assignments from {path}")
    return assignments


def load_configs(path: Path) -> dict:
    """
    Load retrieval configuration presets.

    Args:
        path: Path to retrieval_configs.json

    Returns:
        Dict mapping config_name -> config dict
    """
    if not path.exists():
        raise FileNotFoundError(f"Retrieval configs not found: {path}")

    with open(path, encoding="utf-8") as f:
        configs = json.load(f)

    logger.info(f"Loaded {len(configs)} retrieval configs from {path}")
    return configs

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





COURSE_NAME_MAP = {
    "ml-zoomcamp": "machine-learning-zoomcamp",
    "de-zoomcamp": "data-engineering-zoomcamp",
    "mlops-zoomcamp": "mlops-zoomcamp",
    "llm-zoomcamp": "llm-zoomcamp",
}

def load_valid_ids(clean_path: Path) -> set[str]:
    """Load all valid document IDs from clean.jsonl."""
    ids = set()
    if clean_path.exists():
        for line in open(clean_path, encoding="utf-8"):
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids



def load_test_set(path: Path, clean_path: Path | None = None) -> list[dict]:
    valid_ids = load_valid_ids(clean_path) if clean_path else None
    if not path.exists():
        raise FileNotFoundError(f"Test set not found: {path}")

    tests = []
    skipped = 0
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

            expected_id = doc.get("expected_id", doc.get("expected_doc_id", doc["id"]))
            if valid_ids and expected_id not in valid_ids:
                logger.warning(f"Skipping query {doc['id']}: expected_id {expected_id} not in corpus")
                skipped += 1
                continue

            tests.append({
                "query_id": doc["id"],
                "query": doc["question"],
                "query_type": doc.get("query_type", "unknown"),
                "expected_id": expected_id,
                "course": COURSE_NAME_MAP.get(doc["course"], doc["course"]),
                "section": doc.get("section", ""),
                "answer": doc["answer"],
            })

    logger.info(f"Loaded {len(tests)} test queries from {path} ({skipped} skipped — invalid expected_id)")
    return tests


def load_topic_assignments(path: Path, model: str | None = None) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Topic assignments not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle both single-model and all-models format
    if "results" in data and model:
        assignments_list = data["results"][model]["assignments"]
    elif "results" in data:
        # Default to first model
        first_model = list(data["results"].keys())[0]
        logger.warning(f"No model specified, using {first_model}")
        assignments_list = data["results"][first_model]["assignments"]
    else:
        assignments_list = data.get("assignments", [])
    
    assignments = {a["id"]: a for a in assignments_list}
    logger.info(f"Loaded {len(assignments)} topic assignments from {path}")
    return assignments


def load_configs(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Retrieval configs not found: {path}")

    with open(path, encoding="utf-8") as f:
        configs = json.load(f)

    logger.info(f"Loaded {len(configs)} retrieval configs from {path}")
    return configs

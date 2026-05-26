"""


Public Functions for Topic Document Loading:

def load_documents(path: Path) -> list[dict]:
    Load and validate documents from a JSONL file.
    I/O: path (Path) -> list[dict]

def group_by_course(docs: list[dict]) -> dict[str, list[dict]]:
    Group documents by course for stratified analysis.
    I/O: docs (list[dict]) -> dict[str, list[dict]]





_topic_loader.py
================
Load and validate FAQ documents from JSONL for topic modeling.

Single responsibility: read input, validate schema, return clean list.
No topic logic, no embedding logic, no output logic.

Functions:
    load_documents(path: Path) -> list[dict]
"""
import json
from collections import defaultdict
from pathlib import Path
from rag_pipeline.logging import get_logger
logger = get_logger(__name__)
REQUIRED_FIELDS = {'id', 'question', 'course'}

def load_documents(path: Path) -> list[dict]:
    """
    Load and validate documents from a JSONL file.
    
    Args:
        path: Path to input JSONL file
        
    Returns:
        List of validated document dicts with required fields
        
    Raises:
        FileNotFoundError: If input path does not exist
        ValueError: If no valid documents could be loaded
    """
    if not path.exists():
        raise FileNotFoundError(f'Input file not found: {path}')
    docs = []
    skipped = 0
    with open(path, encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f'Line {line_num}: JSON parse error: {e}')
                skipped += 1
                continue
            missing = REQUIRED_FIELDS - doc.keys()
            if missing:
                doc_id = doc.get('id', f'line_{line_num}')
                logger.warning(f'Line {line_num} [{doc_id}]: Missing required fields {missing}')
                skipped += 1
                continue
            if not doc['question'].strip():
                logger.warning(f"Line {line_num} [{doc['id']}]: Empty question field")
                skipped += 1
                continue
            docs.append(doc)
    if not docs:
        raise ValueError(f'No valid documents loaded from {path}')
    logger.info(f'Loaded {len(docs)} valid documents ({skipped} skipped) from {path}')
    return docs

def group_by_course(docs: list[dict]) -> dict[str, list[dict]]:
    """
    Group documents by course for stratified analysis.
    
    Args:
        docs: List of validated document dicts
        
    Returns:
        Dict mapping course name -> list of docs in that course
    """
    by_course = defaultdict(list)
    for doc in docs:
        by_course[doc['course']].append(doc)
    return dict(by_course)
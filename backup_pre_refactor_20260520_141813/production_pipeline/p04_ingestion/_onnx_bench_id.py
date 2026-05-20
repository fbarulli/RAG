"""
production_pipeline/p04_ingestion/_onnx_bench_id.py
RESPONSIBILITY: Handles data type formatting and deterministic UUID text alignments.
"""
import uuid
from typing import Any, Dict, List, Tuple

def normalize_single_id(hid: Any) -> str:
    """RESPONSIBILITY: Enforces an individual raw scalar variant into a clean lowercase UUID string."""
    hid_str = str(hid).strip()
    
    try:
        uuid_obj = uuid.UUID(hid_str)
        return str(uuid_obj).lower()
    except ValueError:
        pass
        
    # FIX: Swapped from uuid3 to uuid5 to align with the ingestion hashing strategy
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, hid_str)).lower()


def normalize_target_ids(hit_ids: List[Any], limit: int) -> Tuple[List[str], Dict[str, str]]:
    """RESPONSIBILITY: Translates lookup array batches and generates backward reference dictionaries."""
    cleaned_ids: List[str] = []
    id_mapping: Dict[str, str] = {}
    
    for hid in hit_ids[:limit]:
        cleaned_id = normalize_single_id(hid)
        cleaned_ids.append(cleaned_id)
        id_mapping[cleaned_id] = str(hid)
        
    return cleaned_ids, id_mapping

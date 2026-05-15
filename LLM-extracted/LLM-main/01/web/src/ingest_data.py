import sys
import json
import requests
import traceback 
from typing import List, Dict
from elasticsearch import Elasticsearch
from src.config_manager import load_config
from src.core import generate_document_id 
from src.logger_config import logger



def log_and_print(message: str, level: str = "info"):
    # This was originally in ingest_data, keeping it local to its responsibility
    if level == "error":
        logger.error(message)
    print(f"[{level.upper()}] {message}")


def fetch_raw_data(data_url: str) -> list:
    try:
        response = requests.get(url=data_url, timeout=30)
        if response.status_code == 200:
            log_and_print(f"Successfully downloaded data from {data_url}", "info")
            return response.json()
        else:
            raise requests.ConnectionError(f"HTTP code {response.status_code}")
    except Exception as e:
        log_and_print(f"Failed to fetch raw data: {e}", "error")
        sys.exit(1)

def transform_documents(raw_data: List[Dict]) -> List[Dict]:
    flattened_records: List[Dict] = []
    for course in raw_data:
        course_name = course["course"]
        for doc in course["documents"]:
            # Remove any prefix like "Course - ", "Homework - ", "Office Hours - ", etc.
            clean_question = doc["question"]
            if " - " in clean_question:
                clean_question = clean_question.split(" - ", 1)[1].strip()
            
            new_doc = {
                "text": doc["text"],
                "section": doc.get("section", "General"),
                "question": clean_question,
                "course": course_name
            }
            new_doc["id"] = generate_document_id(new_doc)
            flattened_records.append(new_doc)
    return flattened_records


def setup_index_and_ingest(es_client: Elasticsearch, index_name: str, records: List[Dict]):
    """Standardizes index creation and sequential ingestion."""
    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)

    index_settings = {"number_of_shards": 1, "number_of_replicas": 0}
    mappings = {
        "properties": {
            "id": {"type": "keyword"},
            "text": {"type": "text"},
            "section": {"type": "keyword"},
            "question": {"type": "text"},
            "course": {"type": "keyword"}
        }
    }

    # Fixed: Passing settings and mappings as kwargs
    es_client.indices.create(index=index_name, settings=index_settings, mappings=mappings)
    
    for doc in records:
        es_client.index(index=index_name, id=doc["id"], document=doc)
    log_and_print(f"Ingested {len(records)} documents with stable IDs.")



if __name__ == "__main__":
    log_and_print("Starting automated data ingestion pipeline...", "info")
    
    # 1. Load settings (Responsibility: config_manager)
    settings = load_config("settings.json")
    
    # 2. Connect to ES
    # Note: Use settings.get('es_host') if you want to be fully dynamic
    es_client = Elasticsearch("http://localhost:9200")
    if not es_client.ping():
        log_and_print("Elasticsearch is not responding at http://localhost:9200.", "error")
        sys.exit(1)

    # 3. Load Local Data (The Ground Truth)
    # This ensures hashes in ES match hashes in run_stats.py perfectly
    try:
        with open("documents.json", "r") as f:
            raw_data = json.load(f)
        log_and_print("Loaded local documents.json for ingestion.", "info")
    except FileNotFoundError:
        log_and_print("documents.json not found locally!", "error")
        sys.exit(1)

    # 4. Transform and Ingest
    # transform_documents now generates the 32-char stable hashes
    flattened_docs = transform_documents(raw_data=raw_data)
    
    # setup_index_and_ingest uses the modern ES API (no 'body' param)
    setup_index_and_ingest(
        es_client=es_client, 
        index_name="course-questions", 
        records=flattened_docs
    )
    
    log_and_print("Ingestion script completed without failures.", "info")

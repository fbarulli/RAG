#!/bin/bash
# data_cleaning/run_all.sh
# Full pipeline: download → parse → extract metadata → dedup → ingest
# Run from project root: bash data_cleaning/run_all.sh

set -e

echo "=== Step 1: Download raw data ==="
uv run python data_cleaning/download.py

echo ""
echo "=== Step 2: Parse markdown files ==="
uv run python data_cleaning/parse.py

echo ""
echo "=== Step 3: Extract image metadata ==="
uv run python data_cleaning/extract_metadata.py

echo ""
echo "=== Step 4: Deduplicate ==="
uv run python data_cleaning/dedup.py

echo ""
echo "=== Step 5: Ingest to Elasticsearch ==="
uv run python data_cleaning/ingest.py

echo ""
echo "=== Step 6: Ingest to Qdrant ==="
uv run python data_cleaning/ingest_qdrant.py

echo ""
echo "=== Verify ==="
python3 -c "
from elasticsearch import Elasticsearch
es = Elasticsearch('http://localhost:9200')
count = es.count(index='faqs_complete')['count']
print(f'Documents in faqs_complete: {count}')

from qdrant_client import QdrantClient
qdrant = QdrantClient(host='localhost', port=6333)
count = qdrant.count(collection_name='faqs').count
print(f'Documents in Qdrant faqs: {count}')
"

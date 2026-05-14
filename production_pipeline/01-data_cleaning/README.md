# Data Cleaning Pipeline

Downloads FAQ markdown from DataTalksClub/faq, extracts structured data, deduplicates, and ingests into Elasticsearch and Qdrant.

## Files

| Step | File | Description |
|------|------|-------------|
| 1 | `01_download.py` | Downloads FAQ zip from GitHub, extracts markdown files for all 4 courses |
| 2 | `02_parse.py` | Parses markdown → JSONL. Extracts `id`, `question`, `answer`, `course`, `section`. Removes `<{IMAGE}>` tags, markdown headers, HTML, Jinja2 macros |
| 3 | `03_dedup.py` | Removes duplicate questions (95% similarity threshold on full text) |
| 4 | `04_ingest_es.py` | Ingests to Elasticsearch `faqs_complete` index with `question_vector` (bge-base 768d) |
| 5 | `05_ingest_qdrant.py` | Ingests to Qdrant `faqs_bge_base_en_v1.5` collection with bge-base embeddings |

## Output

- `data/processed/clean.jsonl` — 1140 cleaned, deduplicated documents
- Elasticsearch: `faqs_complete` index (1140 docs)
- Qdrant: `faqs_bge_base_en_v1.5` collection (1140 vectors)

## Run

```bash
bash run.sh
```

Or step by step:

```bash
uv run python 01_download.py
uv run python 02_parse.py
uv run python 03_dedup.py
uv run python 04_ingest_es.py
uv run python 05_ingest_qdrant.py
```
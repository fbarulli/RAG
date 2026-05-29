## Pipeline Steps

| Step | Command | File | Description |
|------|---------|------|-------------|
| 1 | `just run download` | `download.py` | Downloads FAQ zip from GitHub, extracts markdown files for all courses |
| 2 | `just run parse` | `parse.py` | Parses markdown → JSONL. Validates `id` (must be a quoted string), extracts `question`, `answer`, `course`, `section`. Removes image placeholders, markdown headers, HTML, Jinja2 macros |
| 3 | `just run dedup` | `dedup.py` | Removes duplicate questions (95% similarity threshold, course-aware) |
| 4 | `just run split` | `stratified_test_split.py` | Stratified train/test split by course |
| 5 | `just run all` | — | Runs steps 1–4 in sequence |

## Output
- `data/processed/parsed.jsonl` — cleaned documents (1207 docs)
- `data/processed/clean.jsonl` — deduplicated documents (1207 docs)

## Schemas

### `FAQDocument` (Pydantic, frozen)
Defined in `src/rag_pipeline/core/schemas.py`. All fields validated on construction.

| Field | Type | Notes |
|-------|------|-------|
| `id` | `str` | Must be a non-empty string. Raw markdown frontmatter must quote integer-looking IDs: `id: '6739977244'` |
| `question` | `str` | Non-empty, whitespace-stripped |
| `answer` | `str` | Non-empty, markdown-cleaned |
| `course` | `str` | Non-empty |
| `section` | `str \| None` | Optional |

### Public functions

#### `parse.py`
- `parse_file(filepath, course, section) -> (FAQDocument | None, str | None)` — parses a single markdown file; returns `(None, reason)` on failure
- `walk_raw_dir(raw_dir) -> Generator[(filepath, course, section)]` — yields all `.md` files under `raw_dir`
- `clean_answer(text) -> str` — strips markdown formatting, preserves fenced code blocks

#### `dedup.py`
- `main(input_path, output_path, threshold, course_aware)` — deduplicates parsed JSONL; default threshold 0.95

## Notes
- Raw markdown `id` fields must be **quoted strings** in YAML frontmatter. Unquoted integers will be rejected by `parse.py` with a clear error message.
- `just run all` is the canonical way to rebuild `clean.jsonl` from scratch.

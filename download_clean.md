data/raw/<course>/<section>/*.md
        ↓
  [walk_raw_dir()]          ← yields (filepath, course, section)
        ↓
  [parse_file()]            ← extracts YAML frontmatter → id, question
        ↓
  [clean_answer()]          ← strips markdown, preserves code blocks
        ↓
  FAQDocument(id, question, answer, course, section)
        ↓
  data/processed/parsed.jsonl   ← one JSON object per line
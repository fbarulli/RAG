#!/usr/bin/env python3
from pathlib import Path
import ast

p = Path("src/rag_pipeline/ablation/corpus_sampler.py")
text = p.read_text(encoding="utf-8")

fixes = [
    # _run( ingest ) before "# 2. Benchmark"
    (
        "        f'--port {port}'\n\n      # 2. Benchmark",
        "        f'--port {port}'\n      )\n\n      # 2. Benchmark",
    ),
    # _run( benchmark ) before "# 3. Parse results"
    (
        "        f'--qdrant-port {port}'\n\n      # 3. Parse",
        "        f'--qdrant-port {port}'\n      )\n\n      # 3. Parse",
    ),
    # print( ... ) in _print_report before print(=)
    (
        "          f\"{h1_str:>8}  {mrr_str:>8}  {delta_str:>8}\"\n      print(f\"{'=' * 62}",
        "          f\"{h1_str:>8}  {mrr_str:>8}  {delta_str:>8}\"\n          )\n      print(f\"{'=' * 62}",
    ),
    # ArgumentParser( ) before parser.add_argument --fractions
    (
        "        description=\"Benchmark H@1 vs corpus size on original queries only\",\n      parser.add_argument(\n          \"--fractions\"",
        "        description=\"Benchmark H@1 vs corpus size on original queries only\",\n      )\n      parser.add_argument(\n          \"--fractions\"",
    ),
    # add_argument --fractions ) before parser.add_argument --model
    (
        "        help=\"Corpus fractions to test, largest first (default: 1.0 0.8 0.6 0.4 0.2)\",\n      parser.add_argument(\"--model\"",
        "        help=\"Corpus fractions to test, largest first (default: 1.0 0.8 0.6 0.4 0.2)\",\n      )\n      parser.add_argument(\"--model\"",
    ),
    # add_argument --input ) before parser.add_argument --query-type
    (
        "        help=\"Corpus JSONL (default: data/processed/clean.jsonl)\",\n      parser.add_argument(\"--query-type\"",
        "        help=\"Corpus JSONL (default: data/processed/clean.jsonl)\",\n      )\n      parser.add_argument(\"--query-type\"",
    ),
    # logger.info( Counter ) before rows
    (
        "        dict(Counter(d.get(\"course\") for d in docs)),\n\n      rows",
        "        dict(Counter(d.get(\"course\") for d in docs)),\n      )\n\n      rows",
    ),
    # run_fraction( ) before rows.append
    (
        "              port=args.port,\n              rows.append",
        "              port=args.port,\n          )\n          rows.append",
    ),
    # logger.info fraction ) before except
    (
        "              f\"{row['mrr']:.4f}\" if isinstance(row[\"mrr\"], float) else \"?\",\n          except",
        "              f\"{row['mrr']:.4f}\" if isinstance(row[\"mrr\"], float) else \"?\",\n          )\n          except",
    ),
    # _run( restore ) before logger.info restored
    (
        "          f'--port {args.port}'\n          logger.info(\"Full corpus restored",
        "          f'--port {args.port}'\n          )\n          logger.info(\"Full corpus restored",
    ),
]

applied = 0
for old, new in fixes:
    # Normalise indentation differences by working on the actual text
    if old in text:
        text = text.replace(old, new)
        applied += 1
    else:
        print(f"  WARNING not matched: {old[:60]!r}")

p.write_text(text, encoding="utf-8")
print(f"  applied {applied}/{len(fixes)} fixes")

try:
    ast.parse(text)
    print("  ok — corpus_sampler.py parses cleanly")
except SyntaxError as e:
    print(f"  FAIL line {e.lineno}: {e.msg}")
    # Show context
    lines = text.splitlines()
    for i in range(max(0, e.lineno-4), min(len(lines), e.lineno+2)):
        print(f"    {i+1:4d}  {lines[i]}")
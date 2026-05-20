#!/usr/bin/env python3
"""
Migration script: Clean up RAG-a-muffin project structure
Run with: uv run python migrate_to_clean_structure.py [--execute]
"""

import shutil
import sys
from pathlib import Path
from datetime import datetime
import re
import ast
import argparse


def create_backup():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"backup_pre_refactor_{timestamp}")
    shutil.copytree(".", backup_dir, ignore=shutil.ignore_patterns('backup_*', '__pycache__', '.git'))
    print(f"✅ Full backup created: {backup_dir}")
    return backup_dir


class ImportRewriter(ast.NodeTransformer):
    def __init__(self, old_to_new):
        self.old_to_new = old_to_new

    def visit_ImportFrom(self, node):
        if node.module:
            for old, new in self.old_to_new.items():
                if node.module.startswith(old):
                    node.module = new + node.module[len(old):]
                    break
        return self.generic_visit(node)

    def visit_Import(self, node):
        for name in node.names:
            for old, new in self.old_to_new.items():
                if name.name.startswith(old):
                    name.name = new + name.name[len(old):]
                    break
        return self.generic_visit(node)


def rewrite_imports_in_file(file_path: Path, old_to_new: dict):
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)
        rewriter = ImportRewriter(old_to_new)
        new_tree = rewriter.visit(tree)
        new_content = ast.unparse(new_tree)

        # Also do a safety regex pass for strings / dynamic imports
        for old, new in old_to_new.items():
            new_content = re.sub(rf'\b{re.escape(old)}\b', new, new_content)

        file_path.write_text(new_content, encoding="utf-8")
        print(f"   Rewrote imports: {file_path}")
    except Exception as e:
        print(f"   ⚠️  Skipped import rewrite for {file_path}: {e}")


def main(execute=False):
    if not execute:
        print("🚀 DRY RUN MODE — No changes will be made. Use --execute to apply.")
    
    backup = create_backup() if execute else None

    root = Path(".")
    src_pkg = root / "src" / "rag_pipeline"
    new_src = root / "src" / "rag_pipeline"

    # 1. Create new structure
    dirs_to_create = [
        "data/raw",
        "data/processed",
        "src/rag_pipeline/cleaning",
        "src/rag_pipeline/ingestion",
        "src/rag_pipeline/core",
        "src/rag_pipeline/evaluation",
    ]
    for d in dirs_to_create:
        (root / d).mkdir(parents=True, exist_ok=True)

    # 2. Mapping: old → new
    moves = {
        # Cleaning
        "production_pipeline/p01_data_cleaning/p00_load_llm_queries.py": "src/rag_pipeline/cleaning/load_llm_queries.py",
        "production_pipeline/p01_data_cleaning/p01_download.py": "src/rag_pipeline/cleaning/download.py",
        "production_pipeline/p01_data_cleaning/p02_parse.py": "src/rag_pipeline/cleaning/parse.py",
        "production_pipeline/p01_data_cleaning/p03_dedup.py": "src/rag_pipeline/cleaning/dedup.py",
        "production_pipeline/p01_data_cleaning/p04_stratified_test_split.py": "src/rag_pipeline/cleaning/stratified_test_split.py",
        
        # Ingestion + ONNX stuff
        "production_pipeline/p04_ingestion/p00_ingest_es.py": "src/rag_pipeline/ingestion/ingest_es.py",
        "production_pipeline/p04_ingestion/p00_ingest_qdrant.py": "src/rag_pipeline/ingestion/ingest_qdrant.py",
        "production_pipeline/p04_ingestion/p02_ingest_models.py": "src/rag_pipeline/ingestion/ingest_models.py",
        # ... (most _onnx_* and _benchmark_* stay under ingestion/)
    }

    # Move shared core files
    core_files = ["paths.py", "schemas.py", "llm_client.py", "logging.py", "gem_client.py"]
    for f in core_files:
        src = src_pkg / f
        if src.exists():
            moves[str(src.relative_to(root))] = f"src/rag_pipeline/core/{f}"

    # Run the moves
    for old_rel, new_rel in moves.items():
        old_path = root / old_rel
        new_path = root / new_rel
        if old_path.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            if execute:
                shutil.move(str(old_path), str(new_path))
                print(f"✅ Moved: {old_rel} → {new_rel}")
            else:
                print(f"   Would move: {old_rel} → {new_rel}")
        else:
            print(f"   ⚠️  Not found: {old_rel}")

    # 3. Move remaining p0X folders if needed (you can expand)

    # 4. Import mapping
    import_map = {
        "production_pipeline.p01_data_cleaning": "rag_pipeline.cleaning",
        "production_pipeline.p04_ingestion": "rag_pipeline.ingestion",
        "src.rag_pipeline": "rag_pipeline",           # important
        "rag_pipeline.paths": "rag_pipeline.core.paths",
        "rag_pipeline.schemas": "rag_pipeline.core.schemas",
        "rag_pipeline.llm_client": "rag_pipeline.core.llm_client",
    }

    if execute:
        print("\n🔄 Rewriting imports across project...")
        for pyfile in root.rglob("*.py"):
            if any(p in str(pyfile) for p in ["backup_", "migrate_to_clean_structure.py"]):
                continue
            rewrite_imports_in_file(pyfile, import_map)

    # 5. Update justfile + run_clean_pipeline.py (manual touch recommended after)

    print("\n🎉 Migration finished!")
    if not execute:
        print("Run with --execute to apply changes.")
    else:
        print("Don't forget to:")
        print("   - Review and update justfile")
        print("   - Update run_clean_pipeline.py if needed")
        print("   - Run `uv sync` and test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    main(execute=args.execute)
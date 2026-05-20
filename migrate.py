#!/usr/bin/env python3
"""
Improved Migration Script for RAG-a-muffin
"""

import shutil
import sys
from pathlib import Path
from datetime import datetime
import re
import ast
import argparse


def create_light_backup():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    backup_dir = Path(f"backup_pre_refactor_{timestamp}")
    ignore = shutil.ignore_patterns(
        'backup_*', '__pycache__', '.git', 'data', '*.onnx', '*.bin', 
        '*.pt', 'node_modules', '*.pdf', 'wandb'
    )
    shutil.copytree(".", backup_dir, ignore=ignore, dirs_exist_ok=True)
    print(f"✅ Light backup created: {backup_dir} (skipped heavy files)")
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

        for old, new in old_to_new.items():
            new_content = re.sub(rf'\b{re.escape(old)}\b', new, new_content)

        file_path.write_text(new_content, encoding="utf-8")
        print(f"   ✓ Rewrote {file_path.name}")
    except Exception as e:
        print(f"   ⚠️  Skipped rewrite {file_path.name}: {e}")


def main(execute=False, backup=True):
    if not execute:
        print("🚀 DRY-RUN MODE (use --execute to apply changes)\n")

    if execute and backup:
        create_light_backup()

    root = Path(".")

    # Create target structure
    for d in [
        "data/raw", "data/processed",
        "src/rag_pipeline/cleaning",
        "src/rag_pipeline/ingestion",
        "src/rag_pipeline/core",
        "src/rag_pipeline/evaluation"
    ]:
        (root / d).mkdir(parents=True, exist_ok=True)

    # ================== FILE MOVES ==================
    moves = {
        # Cleaning
        "production_pipeline/p01_data_cleaning/p00_load_llm_queries.py": "src/rag_pipeline/cleaning/load_llm_queries.py",
        "production_pipeline/p01_data_cleaning/p01_download.py": "src/rag_pipeline/cleaning/download.py",
        "production_pipeline/p01_data_cleaning/p02_parse.py": "src/rag_pipeline/cleaning/parse.py",
        "production_pipeline/p01_data_cleaning/p03_dedup.py": "src/rag_pipeline/cleaning/dedup.py",
        "production_pipeline/p01_data_cleaning/p04_stratified_test_split.py": "src/rag_pipeline/cleaning/stratified_test_split.py",

        # Ingestion
        "production_pipeline/p04_ingestion/": "src/rag_pipeline/ingestion/",   # whole folder if you want

        # Core files currently in src/rag_pipeline/
        "src/rag_pipeline/paths.py": "src/rag_pipeline/core/paths.py",
        "src/rag_pipeline/schemas.py": "src/rag_pipeline/core/schemas.py",
        "src/rag_pipeline/llm_client.py": "src/rag_pipeline/core/llm_client.py",
        "src/rag_pipeline/gem_client.py": "src/rag_pipeline/core/gem_client.py",
        "src/rag_pipeline/logging.py": "src/rag_pipeline/core/logging.py",
    }
    # Add to the moves dict in migrate.py
    moves.update({
        # EDA
        "production_pipeline/p02_eda/": "src/rag_pipeline/eda/",
        
        # Generation
        "production_pipeline/p03_generation/": "src/rag_pipeline/generation/",
        
        # Evaluation
        "production_pipeline/p05_evaluation/": "src/rag_pipeline/evaluation/",
        
        # Answer generation
        "production_pipeline/p06_answer_generation/": "src/rag_pipeline/answer_generation/",
        
        # Keep orchestrator at top level or move it
        "production_pipeline/run_clean_pipeline.py": "src/rag_pipeline/run_clean_pipeline.py",
        
        # Any leftover ingestion files
        "production_pipeline/p04_ingestion/": "src/rag_pipeline/ingestion/",
    })
    for old_rel, new_rel in moves.items():
        old_path = root / old_rel
        new_path = root / new_rel

        if old_path.is_dir():
            if execute:
                shutil.copytree(old_path, new_path, dirs_exist_ok=True)
                shutil.rmtree(old_path)
                print(f"✅ Moved directory: {old_rel} → {new_rel}")
            else:
                print(f"   Would move dir: {old_rel} → {new_rel}")
        elif old_path.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            if execute:
                shutil.move(str(old_path), str(new_path))
                print(f"✅ Moved: {old_rel} → {new_rel}")
            else:
                print(f"   Would move: {old_rel} → {new_rel}")
        else:
            print(f"   ⚠️  Not found: {old_rel}")

    # Import rewrites
    import_map = {
        "production_pipeline.p01_data_cleaning": "rag_pipeline.cleaning",
        "production_pipeline.p04_ingestion": "rag_pipeline.ingestion",
        "src.rag_pipeline": "rag_pipeline",
        "rag_pipeline.paths": "rag_pipeline.core.paths",
        "rag_pipeline.schemas": "rag_pipeline.core.schemas",
        "rag_pipeline.llm_client": "rag_pipeline.core.llm_client",
        "rag_pipeline.gem_client": "rag_pipeline.core.gem_client",
    }

    if execute:
        print("\n🔄 Rewriting imports...")
        for pyfile in sorted(root.rglob("*.py")):
            if any(skip in str(pyfile) for skip in ["backup_", "migrate", "venv", ".git"]):
                continue
            rewrite_imports_in_file(pyfile, import_map)

    print("\n✅ Migration step completed!")
    if not execute:
        print("Run again with:  uv run migrate.py --execute")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    main(execute=args.execute, backup=not args.no_backup)
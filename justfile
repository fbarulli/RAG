# justfile - Task runner for rag-pipeline
set dotenv-load := true
set shell := ["bash", "-cu"]

default:
  just lint && just test

# ── Quality Gates ─────────────────────────────────────────────────────
lint:
  @echo "▶ Running linter..."
  uv run ruff check . --fix
  uv run ruff format .

typecheck:
  @echo "▶ Running type checker..."
  uv run mypy src/ production_pipeline/ --ignore-missing-imports

test extra_args='':
  @echo "▶ Running tests..."
  uv run pytest tests/ -v --cov=src/rag_pipeline {{extra_args}}

# ── Pipeline Execution ────────────────────────────────────────────────
# Usage: just run <stage> [args...]
# Example: just run eda "--dry-run"
run stage args='':
  @echo "▶ Running stage: {{stage}}"
  @case "{{stage}}" in \
    download) uv run python production_pipeline/p01_data_cleaning/p01_download.py ;; \
    parse)    uv run python production_pipeline/p01_data_cleaning/p02_parse.py {{args}} ;; \
    dedup)    uv run python production_pipeline/p01_data_cleaning/p03_dedup.py {{args}} ;; \
    eda)      uv run python production_pipeline/p02_eda/p01_load_and_inspect.py {{args}} ;; \
    all)      just run download && just run parse && just run dedup && just run eda ;; \
    *) echo "Unknown stage: {{stage}}. Use: download | parse | dedup | eda | all"; exit 1 ;; \
  esac

# ── Data Management ───────────────────────────────────────────────────
clean-data:
  @echo "⚠️  Removing processed data (keep raw)..."
  rm -rf production_pipeline/p01_data_cleaning/data/processed/*
  rm -rf production_pipeline/experiments/*
  @echo "✓ Cleaned"

clean-all:
  @echo "⚠️  Removing ALL data..."
  rm -rf production_pipeline/p01_data_cleaning/data/raw/*
  rm -rf production_pipeline/p01_data_cleaning/data/processed/*
  rm -rf production_pipeline/experiments/*
  @echo "✓ Fully cleaned"

# ── Dev Utilities ─────────────────────────────────────────────────────
setup:
  @echo "▶ Setting up with uv..."
  uv sync --extra dev
  @echo "✓ Ready"

shell:
  uv run python

help:
  @just --list

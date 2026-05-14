# justfile - Task runner for rag-pipeline
# Install: https://github.com/casey/just
# Usage: just <task>  (e.g., just test, just run stage=parse)

set dotenv-load := true
set shell := ["bash", "-cu"]

# Default target
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

test:
  @echo "▶ Running tests..."
  uv run pytest tests/ -v --cov=src/rag_pipeline {{extra_args}}

test-fast:
  uv run pytest tests/ -v -m "not slow and not integration" {{extra_args}}

# ── Pipeline Execution ────────────────────────────────────────────────
# Run a specific stage: just run stage=parse
run:
  @echo "▶ Running stage: {{stage}}"
  @case "{{stage}}" in \
    download) uv run python production_pipeline/01-data_cleaning/01_download.py ;; \
    parse) uv run python production_pipeline/01-data_cleaning/02_parse.py {{args|default('')}} ;; \
    dedup) uv run python production_pipeline/01-data_cleaning/03_dedup.py {{args|default('')}} ;; \
    eda) uv run python production_pipeline/02-EDA/01_load_and_inspect.py {{args|default('')}} ;; \
    all) just run stage=download && just run stage=parse && just run stage=dedup && just run stage=eda ;; \
    *) echo "Unknown stage: {{stage}}. Use: download | parse | dedup | eda | all"; exit 1 ;; \
  esac

# Dry-run any stage: just run stage=eda dry_run=true
run-dry:
  just run stage={{stage}} args="--dry-run {{args}}"

# ── Data Management ───────────────────────────────────────────────────
clean-data:
  @echo "⚠️  Removing processed data (keep raw)..."
  rm -rf production_pipeline/01-data_cleaning/data/processed/*
  rm -rf production_pipeline/experiments/*
  @echo "✓ Cleaned"

clean-all:
  @echo "⚠️  Removing ALL data (raw + processed)..."
  rm -rf production_pipeline/01-data_cleaning/data/raw/*
  rm -rf production_pipeline/01-data_cleaning/data/processed/*
  rm -rf production_pipeline/experiments/*
  @echo "✓ Fully cleaned"

# ── Dev Utilities ─────────────────────────────────────────────────────
setup:
  @echo "▶ Setting up with uv..."
  uv sync --extra dev
  @echo "✓ Ready"

shell:
  uv run python

# ── Help ──────────────────────────────────────────────────────────────
help:
  @just --list
  @echo "\nExamples:"
  @echo "  just run stage=parse"
  @echo "  just run stage=eda args='--dry-run'"
  @echo "  just test extra_args='-k validate'"

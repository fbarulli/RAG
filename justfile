# justfile - Task runner for rag-pipeline
# All recipe lines MUST use tabs (not spaces) for indentation
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
	uv run mypy src/rag_pipeline src/rag_pipeline/ --ignore-missing-imports

test extra_args='':
	@echo "▶ Running tests..."
	uv run pytest tests/ -v --cov=src/rag_pipeline --cov=src/rag_pipeline/ {{extra_args}}

# ── Pipeline Execution ────────────────────────────────────────────────
# Usage: just run <stage> [args...]
# Example: just run eda "--dry-run"
run stage args='':
	@echo "▶ Running stage: {{stage}}"
	@case "{{stage}}" in \
		download) uv run python src/rag_pipeline/cleaning/download.py ;; \
		parse)    uv run python src/rag_pipeline/cleaning/parse.py {{args}} ;; \
		dedup)    uv run python src/rag_pipeline/cleaning/dedup.py {{args}} ;; \
		split)    uv run python src/rag_pipeline/cleaning/stratified_test_split.py {{args}} ;; \
		eda)      uv run python src/rag_pipeline/eda/p02_eda/p01_load_and_inspect.py {{args}} ;; \
		ingest-models) uv run python src/rag_pipeline/ingestion/p02_ingest_models.py {{args}} ;; \
		all)      just run download && just run parse && just run dedup && just run split && just run eda ;; \
		*) echo "Unknown stage: {{stage}}. Use: download | parse | dedup | split | eda | ingest-models | all"; exit 1 ;; \
	esac

# Dry-run variant (passes --dry-run to stage scripts)
run-dry stage args='':
	just run {{stage}} "--dry-run {{args}}"

# ── Data Management ───────────────────────────────────────────────────
clean-data:
	@echo "  Removing processed data (keeping raw)..."
	rm -rf src/rag_pipeline/cleaning/data/processed/*.jsonl
	rm -rf src/rag_pipeline/experiments/*.json
	@echo "✓ Cleaned processed data"

clean-all:
	@echo " Removing ALL data (raw + processed)..."
	rm -rf src/rag_pipeline/cleaning/data/raw/*
	rm -rf src/rag_pipeline/cleaning/data/processed/*
	rm -rf src/rag_pipeline/experiments/*
	@echo "✓ Fully cleaned"

# ── Infrastructure ────────────────────────────────────────────────────
up:
	@echo "▶ Starting services (elasticsearch, qdrant)..."
	docker compose up -d
	@echo "✓ Services starting. Check with: just ps"

down:
	@echo "▶ Stopping services..."
	docker compose down
	@echo "✓ Services stopped"

down-v:
	@echo "  Stopping services + removing volumes (data will be lost)..."
	docker compose down -v
	@echo "✓ Services stopped, volumes removed"

ps:
	@docker compose ps

logs service='':
	@if [ -z "{{service}}" ]; then \
		docker compose logs -f; \
	else \
		docker compose logs -f {{service}}; \
	fi

# ── Dev Utilities ─────────────────────────────────────────────────────
setup:
	@echo "▶ Setting up with uv..."
	uv sync --extra dev
	@echo "✓ Ready"

shell:
	uv run python

help:
	@just --list
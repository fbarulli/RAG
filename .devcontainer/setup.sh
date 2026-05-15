#!/bin/bash
# Sync and create .venv from lock file
uv sync

# Git permissions
git config --global --add safe.directory "${containerWorkspaceFolder:-$(pwd)}"

# Env setup
[ -f .env.example ] && [ ! -f .env ] && cp .env.example .env

# Register venv python to Jupyter
.venv/bin/python -m ipykernel install --user --name python3 --display-name "Python 3 (uv venv)"

# Auto-activate venv for all future shells (only adds line once)
grep -qxF 'source /workspaces/LLM/.venv/bin/activate' ~/.bashrc || \
    echo 'source /workspaces/LLM/.venv/bin/activate' >> ~/.bashrc
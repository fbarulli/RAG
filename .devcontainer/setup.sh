#!/bin/bash

# REMOVE or COMMENT OUT the rm -rf .venv line!
# Instead, tell uv to build or restore the environment from your lock file:
uv sync

# Git permissions
git config --global --add safe.directory "${containerWorkspaceFolder:-$(pwd)}"

# Env setup
[ -f .env.example ] && [ ! -f .env ] && cp .env.example .env

# Register the newly synced environment's python to Jupyter
.venv/bin/python -m ipykernel install --user --name python3 --display-name "Python 3 (uv venv)"

ARG PYTHON_VER=3.11
FROM python:${PYTHON_VER}-bookworm

# Use official uv image for high-speed package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# Context is parent directory; copy and install requirements to system
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

"""
secret_agent_man/tools.py
==========================
smolagents Tool subclasses.
Tools:
    SecondOpinionTool — get a second LLM opinion from a different provider
    ReadFileTool      — read a file from disk and return its contents
    SearchCodeTool    — grep/search for a pattern across source files
"""
from __future__ import annotations
import fnmatch
import os
import re
from typing import TYPE_CHECKING

from smolagents import Tool
from secret_agent_man.llm.logging import get_logger

if TYPE_CHECKING:
    from secret_agent_man.cascading_model import CascadingModel

logger = get_logger(__name__)

import subprocess

class RunCommandTool(Tool):
    name = "run_command"
    description = (
        "Runs a shell command on the local Linux terminal and returns stdout + stderr. "
        "Use for filesystem operations, git, or any CLI task. "
        "Input: command (str) — the shell command to execute. "
        "Returns: combined stdout and stderr as a string."
    )
    inputs = {
        "command": {
            "type": "string",
            "description": "The shell command to run.",
        }
    }
    output_type = "string"

    def forward(self, command: str) -> str:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True
        )
        output = result.stdout + result.stderr
        logger.info(f"RunCommandTool: cmd='{command}' rc={result.returncode}")
        return output.strip() or f"(exit code {result.returncode})"
    
    
class SecondOpinionTool(Tool):
    """
    Ask a different LLM provider for a second opinion on the same prompt.
    Use when the agent wants to verify or cross-check an answer.
    Always calls a provider different from the one that answered first.
    """

    name = "second_opinion"
    description = (
        "Gets a second opinion on a prompt from a different LLM provider. "
        "Use when you want to verify or cross-check an answer. "
        "Inputs: prompt (str) — the question to verify; "
        "exclude_provider (str) — provider to skip: "
        "'nvidia', 'groq', 'openrouter', 'huggingface'. "
        "Returns: second opinion string from a different provider."
    )
    inputs = {
        "prompt": {
            "type": "string",
            "description": "The prompt or question to get a second opinion on.",
        },
        "exclude_provider": {
            "type": "string",
            "description": "Provider to exclude: 'nvidia', 'groq', 'openrouter', 'huggingface'.",
        },
    }
    output_type = "string"

    def __init__(self, model: CascadingModel) -> None:
        super().__init__()
        self._model = model

    def forward(self, prompt: str, exclude_provider: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        result = self._model.second_opinion(
            messages=messages,
            exclude_provider=exclude_provider.strip().lower(),
        )
        logger.info(
            f"SecondOpinionTool: provider='{result.raw.provider}' "
            f"latency={result.raw.latency_ms:.0f}ms"
        )
        return result.content


class ReadFileTool(Tool):
    """
    Read a file from disk and return its full contents as a string.
    Use when the agent needs to inspect source code or other text files.
    """

    name = "read_file"
    description = (
        "Reads a file from disk and returns its contents as a string. "
        "Use when you need to inspect source code or any text file. "
        "Input: path (str) — absolute or relative path to the file. "
        "Returns: file contents as a string."
    )
    inputs = {
        "path": {
            "type": "string",
            "description": "Absolute or relative path to the file to read.",
        },
    }
    output_type = "string"

    def forward(self, path: str) -> str:
        path = os.path.expanduser(path.strip())
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ReadFileTool: no file at '{path}'")
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            contents = fh.read()
        logger.info(f"ReadFileTool: read {len(contents)} chars from '{path}'")
        return contents


class SearchCodeTool(Tool):
    """
    Search for a regex pattern across source files under a directory.
    Returns matching lines with file path and line number context.
    """

    name = "search_code"
    description = (
        "Searches for a regex pattern across source files in a directory. "
        "Returns matching lines with file path and line number. "
        "Inputs: pattern (str) — regex to search for; "
        "directory (str) — root directory to search (default '.'). "
        "glob (str) — file glob filter (default '*.py'). "
        "Returns: formatted match results as a string."
    )
    inputs = {
    "pattern": {
        "type": "string",
        "description": "Regex pattern to search for.",
    },
    "directory": {
        "type": "string",
        "description": "Root directory to walk. Defaults to '.'.",
        "nullable": True,
    },
    "glob": {
        "type": "string",
        "description": "Filename glob to filter files. Defaults to '*.py'.",
        "nullable": True,
    },
}
    output_type = "string"

    def forward(self, pattern: str, directory: str = ".", glob: str = "*.py") -> str:
        directory = os.path.expanduser(directory.strip())
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"SearchCodeTool: no directory at '{directory}'")

        try:
            rx = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"SearchCodeTool: invalid regex '{pattern}': {exc}") from exc

        matches: list[str] = []
        for root, _, files in os.walk(directory):
            for fname in files:
                if not fnmatch.fnmatch(fname, glob):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                        for lineno, line in enumerate(fh, 1):
                            if rx.search(line):
                                matches.append(f"{fpath}:{lineno}: {line.rstrip()}")
                except OSError:
                    continue

        logger.info(
            f"SearchCodeTool: pattern='{pattern}' dir='{directory}' "
            f"glob='{glob}' matches={len(matches)}"
        )
        if not matches:
            return f"No matches for '{pattern}' in '{directory}' ({glob})"
        return "\n".join(matches)
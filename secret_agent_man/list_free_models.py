"""
secret_agent_man/list_free_models.py
======================================
Fetches all free/free-tier models from Groq, OpenRouter, NVIDIA NIM,
and HuggingFace and prints them grouped by parameter size.

Run:
    uv run python3 secret_agent_man/list_free_models.py
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from dotenv import load_dotenv
import requests

load_dotenv()


@dataclass
class ModelEntry:
    provider: str
    model_id: str
    size_b: float  # billions of parameters, 0 if unknown


def _extract_size(model_id: str) -> float:
    """Extract parameter size in billions from model id string."""
    # matches patterns like 70b, 8b, 1.5b, 405b, 32b, 3.1-8b, 3.3-70b
    match = re.search(r'[\-_]?(\d+\.?\d*)b(?:[\-_]|$)', model_id.lower())
    if match:
        return float(match.group(1))
    # also catch NxB MoE patterns like 17b-16e
    match = re.search(r'(\d+)b', model_id.lower())
    if match:
        return float(match.group(1))
    return 0.0


def fetch_groq(api_key: str) -> list[ModelEntry]:
    resp = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    resp.raise_for_status()
    entries = []
    for m in resp.json().get("data", []):
        mid = m["id"]
        # skip audio/guard models
        if any(x in mid.lower() for x in ["whisper", "guard", "tts", "vision"]):
            continue
        entries.append(ModelEntry("groq", f"groq/{mid}", _extract_size(mid)))
    return entries


def fetch_openrouter(api_key: str) -> list[ModelEntry]:
    resp = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    resp.raise_for_status()
    entries = []
    for m in resp.json().get("data", []):
        pricing = m.get("pricing", {})
        prompt_cost = pricing.get("prompt", "1")
        if str(prompt_cost) not in ("0", "0.0", 0):
            continue
        mid = m["id"]
        # skip image/audio/embedding
        arch = m.get("architecture", {})
        if arch.get("modality", "") not in ("text->text", "text+image->text", ""):
            continue
        entries.append(ModelEntry("openrouter", f"openrouter/{mid}", _extract_size(mid)))
    return entries


def fetch_nvidia(api_key: str) -> list[ModelEntry]:
    resp = requests.get(
        "https://integrate.api.nvidia.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    resp.raise_for_status()
    entries = []
    for m in resp.json().get("data", []):
        mid = m["id"]
        # skip embeddings, vision-only, audio
        if any(x in mid.lower() for x in ["embed", "whisper", "rerank", "bge", "clip", "fuyu", "vision"]):
            continue
        entries.append(ModelEntry("nvidia", f"nvidia_nim/{mid}", _extract_size(mid)))
    return entries


def fetch_huggingface(api_key: str) -> list[ModelEntry]:
    resp = requests.get(
        "https://huggingface.co/api/models",
        params={"pipeline_tag": "text-generation", "inference": "warm", "limit": 30},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    resp.raise_for_status()
    entries = []
    for m in resp.json():
        mid = m["id"]
        entries.append(ModelEntry("huggingface", f"huggingface/{mid}", _extract_size(mid)))
    return entries


SIZE_BUCKETS: list[tuple[str, float, float]] = [
    ("tiny  (<3B)",      0,    3),
    ("small (3B–10B)",   3,   10),
    ("medium(10B–40B)", 10,   40),
    ("large (40B–100B)",40,  100),
    ("xl    (100B+)",  100, 9999),
    ("unknown",          0,    0),  # special bucket
]


def bucket(size: float) -> str:
    if size == 0:
        return "unknown"
    for label, lo, hi in SIZE_BUCKETS[:-1]:
        if lo <= size < hi:
            return label
    return SIZE_BUCKETS[-2][0]  # xl


def main() -> None:
    groq_key = os.getenv("GROQ_API_KEY", "")
    or_key   = os.getenv("OPENROUTER_API_KEY", "")
    nv_key   = os.getenv("NVIDIA_API_KEY", "")
    hf_key   = os.getenv("HUGGINGFACE_API_KEY", "")

    all_models: list[ModelEntry] = []

    fetchers = [
        ("Groq",        fetch_groq,        groq_key),
        ("OpenRouter",  fetch_openrouter,  or_key),
        ("NVIDIA NIM",  fetch_nvidia,      nv_key),
        ("HuggingFace", fetch_huggingface, hf_key),
    ]

    for name, fn, key in fetchers:
        if not key:
            print(f"⚠  {name}: key not set, skipping.")
            continue
        try:
            models = fn(key)
            all_models.extend(models)
            print(f"✓  {name}: {len(models)} models fetched.")
        except Exception as e:
            print(f"✗  {name}: failed — {e}")

    # deduplicate by model_id
    seen: set[str] = set()
    unique: list[ModelEntry] = []
    for m in all_models:
        if m.model_id not in seen:
            seen.add(m.model_id)
            unique.append(m)

    # group by size bucket
    grouped: dict[str, list[ModelEntry]] = {b[0]: [] for b in SIZE_BUCKETS}
    for m in unique:
        grouped[bucket(m.size_b)].append(m)

    print(f"\n{'='*70}")
    print(f"FREE MODELS — {len(unique)} total across {len(fetchers)} providers")
    print(f"{'='*70}")

    for label, _, _ in SIZE_BUCKETS:
        models = grouped.get(label, [])
        if not models:
            continue
        print(f"\n  {label.upper()}  ({len(models)} models)")
        print(f"  {'-'*60}")
        for m in sorted(models, key=lambda x: (x.provider, x.model_id)):
            print(f"  [{m.provider:<12}] {m.model_id}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
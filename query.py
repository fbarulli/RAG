#!/usr/bin/env python3
"""
Quick CLI script to validate reranker entries in retrieval_configs.json
and cross-reference against rerankers.json.

Usage:
    uv run python check_reranker_configs.py
    uv run python check_reranker_configs.py --configs-path /path/to/retrieval_configs.json
"""
import json
import argparse
from pathlib import Path

EXPECTED_CONFIGS = [
    "entity_boosted_tinybert",
    "entity_boosted_minilm_l6",
    "entity_boosted_mxbai_xsmall",
    "entity_boosted_answerdotai",
    "entity_boosted_bge_reranker",
]

REQUIRED_FIELDS = ["name", "search_type", "reranker", "reranker_name", "top_k"]


def main():
    parser = argparse.ArgumentParser(description="Validate reranker retrieval configs.")
    parser.add_argument(
        "--configs-path", type=Path,
        default=Path("configs/retrieval_configs.json"),
        help="Path to retrieval_configs.json",
    )
    parser.add_argument(
        "--rerankers-path", type=Path,
        default=Path("configs/rerankers.json"),
        help="Path to rerankers.json",
    )
    args = parser.parse_args()

    ok = True

    # --- Load retrieval_configs.json ---
    if not args.configs_path.exists():
        print(f"[ERROR] retrieval_configs.json not found at: {args.configs_path}")
        raise SystemExit(1)

    retrieval_configs = json.loads(args.configs_path.read_text())
    print(f"[OK] Loaded retrieval_configs.json ({len(retrieval_configs)} entries)\n")

    # --- Load rerankers.json ---
    if not args.rerankers_path.exists():
        print(f"[ERROR] rerankers.json not found at: {args.rerankers_path}")
        raise SystemExit(1)

    rerankers_data = json.loads(args.rerankers_path.read_text())
    known_reranker_names = {m["name"] for m in rerankers_data.get("models", [])}
    print(f"[OK] Loaded rerankers.json — known reranker names: {sorted(known_reranker_names)}\n")

    # --- Check each expected config ---
    print("=== Checking expected reranker configs ===\n")
    for key in EXPECTED_CONFIGS:
        if key not in retrieval_configs:
            print(f"  [MISSING] '{key}' not found in retrieval_configs.json")
            ok = False
            continue

        cfg = retrieval_configs[key]
        print(f"  [{key}]")

        # Required fields
        for field in REQUIRED_FIELDS:
            if field not in cfg:
                print(f"    [ERROR] Missing required field: '{field}'")
                ok = False
            else:
                print(f"    {field}: {cfg[field]}")

        # reranker_name must exist in rerankers.json
        reranker_name = cfg.get("reranker_name")
        if reranker_name and reranker_name not in known_reranker_names:
            print(f"    [ERROR] reranker_name '{reranker_name}' not found in rerankers.json")
            ok = False
        elif reranker_name:
            print(f"    [OK] reranker_name '{reranker_name}' found in rerankers.json")

        # reranker flag must be true
        if not cfg.get("reranker", False):
            print(f"    [ERROR] 'reranker' field is not true")
            ok = False

        # search_type must be entity_boosted
        if cfg.get("search_type") != "entity_boosted":
            print(f"    [WARN] search_type is '{cfg.get('search_type')}', expected 'entity_boosted'")

        print()

    # --- Check for stale reranker_name in rerankers.json nested blocks ---
    print("=== Checking rerankers.json for stale nested configs ===\n")
    for m in rerankers_data.get("models", []):
        nested = m.get("vector_with_reranking", {})
        if nested:
            hardcoded = nested.get("reranker_name")
            if hardcoded and hardcoded != m["name"]:
                print(
                    f"  [WARN] '{m['name']}' has nested reranker_name='{hardcoded}' "
                    f"— should be '{m['name']}' or removed entirely"
                )
                ok = False
        if "vector_with_reranking" in m:
            print(f"  [WARN] '{m['name']}' still has deprecated 'vector_with_reranking' block — safe to remove")

    print("\n" + ("=" * 50))
    if ok:
        print("All checks passed.")
    else:
        print("Some checks failed — see above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
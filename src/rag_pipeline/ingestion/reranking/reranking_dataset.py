"""
reranking_dataset.py
====================
Dataset and collation for listwise reranker fine-tuning.

Each item in the triples JSON becomes a QueryGroup: one query paired with
its positive answer and N hard negatives. Groups with fewer than
max_negatives real negatives are padded with the positive text; padding
positions are masked out in the loss so they contribute no gradient.

Public API
----------
    QueryGroup          dataclass
    RerankerDataset     torch.utils.data.Dataset
    make_collate_fn()   -> Callable
"""
from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class QueryGroup:
    """
    One training example: a query with its positive and hard negatives.

    Attributes
    ----------
    query            : the question text
    positive         : the correct answer
    negatives        : hard negatives padded to max_negatives with the positive
    n_real_negatives : count of real (non-padding) negatives
    """
    query:            str
    positive:         str
    negatives:        list[str]
    n_real_negatives: int


class RerankerDataset(Dataset):
    """
    Loads triples from a JSON file and exposes them as QueryGroups.

    Parameters
    ----------
    triples_path  : absolute path to triples_sample_*.json
    max_negatives : maximum hard negatives per group; shorter lists are padded
    """

    def __init__(self, triples_path: str, max_negatives: int) -> None:
        path = Path(triples_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Triples file not found: {path}. "
                "Run create_training_triples.py first."
            )

        logger.info("Loading triples from %s", path)
        try:
            raw: list[dict] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.error("Failed to parse triples JSON\n%s", traceback.format_exc())
            raise

        self.groups: list[QueryGroup] = []
        skipped = 0

        for item in raw:
            try:
                negs   = item.get("hard_negatives") or item.get("negatives") or []
                negs   = list(negs[:max_negatives])
                n_real = len(negs)

                # Pad to fixed width so batches are rectangular
                while len(negs) < max_negatives:
                    negs.append(item["positive"])

                self.groups.append(QueryGroup(
                    query            = item["query"],
                    positive         = item["positive"],
                    negatives        = negs,
                    n_real_negatives = n_real,
                ))
            except KeyError as exc:
                logger.warning("Skipping malformed triple (missing key %s): %s", exc, item)
                skipped += 1

        logger.info(
            "RerankerDataset ready — %d groups loaded, %d skipped, "
            "max_negatives=%d",
            len(self.groups), skipped, max_negatives,
        )

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> QueryGroup:
        return self.groups[idx]


def make_collate_fn(tokenizer: AutoTokenizer, max_length: int) -> Callable:
    """
    Build a collate function for DataLoader.

    Each group is tokenised as (1 + max_negatives) query-passage pairs:
        [query, positive], [query, neg_0], ..., [query, neg_{N-1}]

    Returns a batch dict with shapes:
        input_ids / attention_mask / token_type_ids : (B, G, seq_len)
        mask                                        : (B, G)  bool
            True  = real candidate (positive or real negative)
            False = padding position (zeroed out in loss)
    """

    def _collate(batch: list[QueryGroup]) -> dict[str, torch.Tensor]:
        all_pairs: list[tuple[str, str]] = []
        masks:     list[list[int]]       = []

        for g in batch:
            all_pairs.append((g.query, g.positive))
            for neg in g.negatives:
                all_pairs.append((g.query, neg))

            masks.append(
                [1]
                + [1] * g.n_real_negatives
                + [0] * (len(g.negatives) - g.n_real_negatives)
            )

        try:
            encoded = tokenizer(
                [p[0] for p in all_pairs],
                [p[1] for p in all_pairs],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        except Exception:
            logger.error("Tokenisation failed\n%s", traceback.format_exc())
            raise

        B = len(batch)
        G = 1 + len(batch[0].negatives)   # 1 positive + max_negatives
        L = encoded["input_ids"].shape[1]

        result: dict[str, torch.Tensor] = {
            "input_ids":      encoded["input_ids"].view(B, G, L),
            "attention_mask": encoded["attention_mask"].view(B, G, L),
            "mask":           torch.tensor(masks, dtype=torch.bool),
        }
        if "token_type_ids" in encoded:
            result["token_type_ids"] = encoded["token_type_ids"].view(B, G, L)

        return result

    return _collate

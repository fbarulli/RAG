# src/rag_pipeline/eda/topics/classification/topic_rules.py
"""
Hybrid classification rules.
Loads signals from entity_patterns.json — no hardcoded terms.
Reclassifies OTHER using rules, but defers to cluster signal when confidence is high.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_pipeline.eda.core.logging import get_logger
from rag_pipeline.eda.core.paths import Paths

logger = get_logger(__name__)

_PRIORITY = ["ERROR", "ADMIN", "LANGUAGE", "TOOL", "CONCEPT"]


def _load_entity_patterns() -> dict[str, list[str]]:
    with open(Paths.entity_patterns(), encoding="utf-8") as f:
        return json.load(f)


def _word_match(signal: str, text: str) -> bool:
    """Whole-word match for short signals (<=3 chars), substring for longer."""
    if len(signal) <= 3:
        return bool(re.search(rf"\b{re.escape(signal)}\b", text))
    return signal in text


def _cluster_majority_category(
    topic_id: int,
    all_assignments: list[dict[str, Any]],
    min_confidence: float = 0.8,
) -> str | None:
    if topic_id == -1:
        return None
    peers = [a for a in all_assignments if a.get("topic") == topic_id]
    if not peers:
        return None
    counts: dict[str, int] = {}
    for a in peers:
        cat = a.get("ner_category", "OTHER")
        if cat != "OTHER":
            counts[cat] = counts.get(cat, 0) + 1
    if not counts:
        return None
    top_cat = max(counts, key=lambda c: counts[c])
    confidence = counts[top_cat] / len(peers)
    if confidence >= min_confidence:
        logger.debug("Cluster %d majority=%s confidence=%.2f", topic_id, top_cat, confidence)
        return top_cat
    return None


@dataclass
class ClassificationRules:
    signals: dict[str, set[str]]

    @classmethod
    def load(cls) -> "ClassificationRules":
        patterns = _load_entity_patterns()
        signals = {cat: set(terms) for cat, terms in patterns.items()}
        for cat, terms in signals.items():
            logger.info("Loaded %d signals for %s", len(terms), cat)
        return cls(signals=signals)

    def reclassify(
        self,
        current_category: str,
        question: str,
        topic_id: int = -1,
        all_assignments: list[dict[str, Any]] | None = None,
        skip_cluster: bool = False,
        skip_rules: bool = False,
    ) -> tuple[str, str]:
        """
        Returns (new_category, source) where source is one of:
            'unchanged' — already classified, not OTHER
            'cluster'   — cluster majority signal
            'rules'     — keyword match
            'fallback'  — stayed OTHER
        """
        if current_category != "OTHER":
            return current_category, "unchanged"

        q_lower = question.lower().strip()
        if not skip_rules:
            for cat in _PRIORITY:
                if cat not in self.signals:
                    continue
                if any(_word_match(sig, q_lower) for sig in self.signals[cat]):
                    return cat, "rules"

        if not skip_cluster and all_assignments is not None:
            cluster_cat = _cluster_majority_category(topic_id, all_assignments)
            if cluster_cat:
                return cluster_cat, "cluster"

        return "OTHER", "fallback"
    def extract_entity(self, category: str, question: str) -> str | None:
        """Return the first matching signal for category in question, or None."""
        q_lower = question.lower().strip()
        for sig in self.signals.get(category, []):
            if _word_match(sig, q_lower):
                return sig
        return None

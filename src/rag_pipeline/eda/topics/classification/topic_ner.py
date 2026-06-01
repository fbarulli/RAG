# src/rag_pipeline/eda/topics/classification/topic_ner.py
"""
Domain-aware NER for topic taxonomy using SpaCy EntityRuler.
Seeded from entity_patterns.json — no hardcoded term lists.

Entity labels: TOOL, LANGUAGE, CONCEPT, ERROR, ADMIN
"""
from __future__ import annotations
import csv
import traceback
from pathlib import Path

import spacy
from spacy.language import Language

from rag_pipeline.eda.core.logging import get_logger
from rag_pipeline.eda.core.paths import Paths

logger = get_logger(__name__)


def _load_entity_patterns() -> dict[str, list[str]]:
    import json
    with open(Paths.entity_patterns(), encoding="utf-8") as f:
        return json.load(f)


def _load_tfidf_terms(path: Path) -> set[str]:
    terms = set()
    if not path.exists():
        logger.warning("TF-IDF terms file not found: %s", path)
        return terms
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            terms.add(row["term"].strip().lower())
    return terms


def _load_bridge_terms(path: Path) -> set[str]:
    terms = set()
    if not path.exists():
        logger.warning("Bridge concepts file not found: %s", path)
        return terms
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            terms.add(row["term"].strip().lower())
    return terms


def _classify_term(term: str, patterns: dict[str, list[str]]) -> str:
    """Assign entity label using entity_patterns.json — no hardcoded fallbacks."""
    for cat in ["TOOL", "LANGUAGE", "CONCEPT", "ADMIN", "ERROR"]:
        if term in patterns.get(cat, []):
            return cat
    return "TOOL"  # conservative default for unknown TF-IDF terms


def _resolve_category(question_lower: str, entities: list[dict], error_signals: set[str]) -> str:
    """Priority: ADMIN > ERROR > LANGUAGE > TOOL > CONCEPT > OTHER."""
    labels = {e["label"] for e in entities}
    if "ADMIN" in labels:
        return "ADMIN"
    if any(sig in question_lower for sig in error_signals):
        return "ERROR"
    if "LANGUAGE" in labels:
        return "LANGUAGE"
    if "TOOL" in labels:
        return "TOOL"
    if "CONCEPT" in labels:
        return "CONCEPT"
    return "OTHER"


def _primary_entity(entities: list[dict], category: str) -> str | None:
    priority_label = {
        "TOOL": "TOOL",
        "LANGUAGE": "LANGUAGE",
        "CONCEPT": "CONCEPT",
        "ERROR": "TOOL",
        "ADMIN": None,
        "OTHER": None,
    }.get(category)
    if priority_label is None:
        return None
    for e in entities:
        if e["label"] == priority_label:
            return e["text"]
    return None


def build_ner(tfidf_terms_path: Path, bridge_concepts_path: Path) -> Language:
    """Build a SpaCy pipeline with EntityRuler seeded from entity_patterns.json + TF-IDF."""
    try:
        entity_patterns = _load_entity_patterns()
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
        ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
        patterns: list[dict] = []

        # seed from entity_patterns.json
        for cat, terms in entity_patterns.items():
            for term in terms:
                patterns.append({"label": cat, "pattern": term})
                if term != term.title():
                    patterns.append({"label": cat, "pattern": term.title()})

        # enrich with TF-IDF and bridge terms
        tfidf_terms = _load_tfidf_terms(tfidf_terms_path)
        bridge_terms = _load_bridge_terms(bridge_concepts_path)
        for term in tfidf_terms | bridge_terms:
            label = _classify_term(term, entity_patterns)
            patterns.append({"label": label, "pattern": term})

        ruler.add_patterns(patterns)
        logger.info(
            "EntityRuler built | patterns=%d tfidf=%d bridge=%d",
            len(patterns), len(tfidf_terms), len(bridge_terms),
        )
        return nlp
    except Exception:
        logger.error("build_ner failed", exc_info=True)
        raise


def tag_questions(questions: list[str], nlp: Language) -> list[dict]:
    """Tag each question with entities and top-level category."""
    try:
        entity_patterns = _load_entity_patterns()
        error_signals = set(entity_patterns.get("ERROR", []))
        results = []
        for doc in nlp.pipe(questions, batch_size=64):
            entities = [{"text": ent.text.lower(), "label": ent.label_} for ent in doc.ents]
            category = _resolve_category(doc.text.lower(), entities, error_signals)
            primary = _primary_entity(entities, category)
            results.append({
                "question": doc.text,
                "entities": entities,
                "category": category,
                "primary_entity": primary,
            })
        logger.info("Tagged %d questions", len(results))
        return results
    except Exception:
        logger.error("tag_questions failed", exc_info=True)
        raise
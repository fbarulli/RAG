"""
_topic_ner.py
=============
Domain-aware NER for topic taxonomy using SpaCy EntityRuler.
Seeded from TF-IDF top terms and bridge concepts — no hardcoded guessing.

Entity labels
-------------
TOOL        : specific tools (docker, mlflow, spark, kestra, dbt ...)
LANGUAGE    : programming/query languages (python, sql, java, pyspark ...)
CONCEPT     : ML/data concepts (embedding, regression, precision ...)
ERROR       : questions containing error/exception/warning signals
ADMIN       : course logistics (certificate, cohort, homework, office hours ...)

Usage
-----
    from production_pipeline.p02_eda._topic_ner import build_ner, tag_questions

    nlp = build_ner(
        tfidf_terms_path=Path("...tfidf_terms_long.csv"),
        bridge_concepts_path=Path("...bridge_concepts.csv"),
    )
    tagged = tag_questions(questions, nlp)
"""
from __future__ import annotations

import csv
import traceback
from pathlib import Path

import spacy
from spacy.language import Language

from rag_pipeline.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Static seed lists — only what TF-IDF cannot derive (meta-categories)
# ---------------------------------------------------------------------------

_LANGUAGES: frozenset[str] = frozenset({
    "python", "sql", "java", "pyspark", "bash", "yaml", "json",
    "javascript", "typescript", "r", "scala", "go", "rust",
})

_ADMIN_TERMS: frozenset[str] = frozenset({
    "certificate", "cohort", "deadline", "donate", "donation", "sponsor",
    "office hours", "self-paced", "graduate", "graduation", "leaderboard",
    "peer review", "prerequisite", "homework", "capstone", "recorded",
    "miss a session", "get help", "skip topics", "how much theory",
    "special hardware", "system design", "support",
})

_ERROR_SIGNALS: frozenset[str] = frozenset({
    "error", "exception", "traceback", "failed", "failure", "warning",
    "cannot", "can't", "unable", "not found", "permission denied",
    "attributeerror", "valueerror", "typeerror", "importerror",
    "modulenotfounderror", "filenotfounderror", "keyerror", "indexerror",
    "oserror", "runtimeerror", "nameerror",
})

# Terms that are tools, not concepts, even if TF-IDF scores them similarly
_TOOL_OVERRIDE: frozenset[str] = frozenset({
    "docker", "kubernetes", "terraform", "airflow", "kafka", "spark",
    "kestra", "dbt", "mlflow", "prefect", "mage", "dagster",
    "flask", "fastapi", "streamlit", "jupyter", "github", "git",
    "postgres", "postgresql", "bigquery", "gcs", "gcp", "aws", "ec2",
    "s3", "lambda", "pipenv", "conda", "pip", "poetry",
    "langchain", "openai", "huggingface", "qdrant", "elasticsearch",
    "redis", "mongodb", "mysql", "sqlite",
})

# ML/data concepts — not tools
_CONCEPT_TERMS: frozenset[str] = frozenset({
    "embedding", "embeddings", "regression", "classification", "precision",
    "recall", "accuracy", "overfitting", "underfitting", "gradient",
    "backpropagation", "attention", "transformer", "tokenizer",
    "vectorizer", "tfidf", "tf-idf", "cosine similarity", "clustering",
    "dimensionality reduction", "umap", "pca", "hdbscan", "bertopic",
    "rag", "retrieval", "fine-tuning", "inference", "training",
    "deployment", "monitoring", "batch", "streaming",
})


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_ner(
    tfidf_terms_path: Path,
    bridge_concepts_path: Path,
) -> Language:
    """
    Build a SpaCy pipeline with an EntityRuler seeded from TF-IDF outputs.

    Parameters
    ----------
    tfidf_terms_path    : path to tfidf_terms_long.csv
    bridge_concepts_path: path to bridge_concepts.csv

    Returns
    -------
    SpaCy Language object with entity_ruler added
    """
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])  # replace default NER

        ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})

        patterns: list[dict] = []

        # --- Load TF-IDF top terms ---
        tfidf_terms = _load_tfidf_terms(tfidf_terms_path)
        bridge_terms = _load_bridge_terms(bridge_concepts_path)
        all_domain_terms = tfidf_terms | bridge_terms

        for term in all_domain_terms:
            label = _classify_term(term)
            patterns.append({"label": label, "pattern": term})
            # Also add title-case variant for questions
            if term != term.title():
                patterns.append({"label": label, "pattern": term.title()})

        # --- Static languages ---
        for lang in _LANGUAGES:
            patterns.append({"label": "LANGUAGE", "pattern": lang})
            patterns.append({"label": "LANGUAGE", "pattern": lang.upper()})

        # --- Static admin terms ---
        for term in _ADMIN_TERMS:
            patterns.append({"label": "ADMIN", "pattern": term})
            patterns.append({"label": "ADMIN", "pattern": term.title()})

        ruler.add_patterns(patterns)

        logger.info(
            f"[build_ner] EntityRuler built: {len(patterns)} patterns | "
            f"tfidf_terms={len(tfidf_terms)}, bridge_terms={len(bridge_terms)}"
        )
        return nlp

    except Exception:
        logger.error("[build_ner] Failed.\n" + traceback.format_exc())
        raise


# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------

def tag_questions(
    questions: list[str],
    nlp: Language,
) -> list[dict]:
    """
    Tag each question with detected entities and a top-level category.

    Parameters
    ----------
    questions   : list of question strings
    nlp         : fitted SpaCy pipeline from build_ner()

    Returns
    -------
    List of dicts with keys:
        question    : original string
        entities    : list of {text, label} dicts
        category    : top-level category (ADMIN, ERROR, TOOL, LANGUAGE, CONCEPT, OTHER)
        primary_entity : most prominent entity text (for subtopic grouping)
    """
    try:
        results = []
        for doc in nlp.pipe(questions, batch_size=64):
            entities = [{"text": ent.text.lower(), "label": ent.label_} for ent in doc.ents]

            # Determine top-level category — priority order matters
            question_lower = doc.text.lower()
            category = _resolve_category(question_lower, entities)

            # Primary entity for subtopic grouping
            primary = _primary_entity(entities, category)

            results.append({
                "question": doc.text,
                "entities": entities,
                "category": category,
                "primary_entity": primary,
            })

        logger.info(f"[tag_questions] Tagged {len(results)} questions")
        return results

    except Exception:
        logger.error("[tag_questions] Failed.\n" + traceback.format_exc())
        raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_tfidf_terms(path: Path) -> set[str]:
    terms = set()
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            terms.add(row["term"].strip().lower())
    return terms


def _load_bridge_terms(path: Path) -> set[str]:
    terms = set()
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            terms.add(row["term"].strip().lower())
    return terms


def _classify_term(term: str) -> str:
    """Assign entity label to a TF-IDF term."""
    if term in _TOOL_OVERRIDE:
        return "TOOL"
    if term in _LANGUAGES:
        return "LANGUAGE"
    if term in _CONCEPT_TERMS:
        return "CONCEPT"
    # Bigrams containing known tools -> TOOL
    for tool in _TOOL_OVERRIDE:
        if tool in term:
            return "TOOL"
    # Default domain terms to TOOL (most TF-IDF top terms are tools/frameworks)
    return "TOOL"


def _resolve_category(question_lower: str, entities: list[dict]) -> str:
    """Priority: ADMIN > ERROR > LANGUAGE > TOOL > CONCEPT > OTHER."""
    labels = {e["label"] for e in entities}

    if "ADMIN" in labels:
        return "ADMIN"
    if any(sig in question_lower for sig in _ERROR_SIGNALS):
        return "ERROR"
    if "LANGUAGE" in labels:
        return "LANGUAGE"
    if "TOOL" in labels:
        return "TOOL"
    if "CONCEPT" in labels:
        return "CONCEPT"
    return "OTHER"


def _primary_entity(entities: list[dict], category: str) -> str | None:
    """Return the most relevant entity for subtopic grouping."""
    priority_label = {
        "TOOL": "TOOL",
        "LANGUAGE": "LANGUAGE",
        "CONCEPT": "CONCEPT",
        "ERROR": "TOOL",   # group errors by the tool they relate to
        "ADMIN": None,
        "OTHER": None,
    }.get(category)

    if priority_label is None:
        return None

    for e in entities:
        if e["label"] == priority_label:
            return e["text"]
    return None
"""
LLM-based multi-entity NER replacing spaCy pipeline.

Returns entities: list[str] and categories: list[str] per doc.
Validated by Pydantic. Removes all hand-crafted pattern machinery.

Usage:
    from rag_pipeline.eda.topics.classification.llm_ner import tag_ner_llm
    tagged = tag_ner_llm(questions)  # {question: NERResult}
"""
from __future__ import annotations
import json
import logging
from typing import Optional
from pydantic import BaseModel, field_validator
from rag_pipeline.eda.core.llm_client import call_llm
from rag_pipeline.eda.core.paths import Paths

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {"TOOL", "ERROR", "CONCEPT", "LANGUAGE", "ADMIN", "OTHER"}

_SYSTEM = (
    "You are a technical entity extractor for FAQ questions about ML and "
    "data engineering courses. Return ONLY valid JSON. No explanation, no markdown."
)

_PROMPT_TEMPLATE = """Extract technical entities from each FAQ question. Return a JSON array with one object per question, in the same order.

Schema per object:
{{
  "entities": ["list", "of", "technical", "terms"],
  "categories": ["TOOL"],
  "primary_entity": "most specific single term, or null",
  "primary_category": "single best fit"
}}

Categories (pick from): TOOL  ERROR  CONCEPT  LANGUAGE  ADMIN  OTHER

Few-shot examples:
Q: "Getting ERROR [internal] load metadata for public.ecr.aws/lambda/python:3.8"
A: {{"entities": ["ecr", "lambda", "docker"], "categories": ["ERROR", "TOOL"], "primary_entity": "ecr", "primary_category": "ERROR"}}

Q: "How do I set up a virtual environment in Python for the course?"
A: {{"entities": ["python", "virtualenv"], "categories": ["TOOL", "LANGUAGE"], "primary_entity": "python", "primary_category": "TOOL"}}

Q: "Prefect flow fails with KeyError when reading from S3"
A: {{"entities": ["prefect", "s3"], "categories": ["ERROR", "TOOL"], "primary_entity": "prefect", "primary_category": "ERROR"}}

Questions:
{questions}

Return a JSON array of {n} objects:"""


class NERResult(BaseModel):
    entities: list[str]
    categories: list[str]
    primary_entity: Optional[str]
    primary_category: str

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, v: list[str]) -> list[str]:
        return [c if c in VALID_CATEGORIES else "OTHER" for c in v]

    @field_validator("primary_category")
    @classmethod
    def validate_primary_category(cls, v: str) -> str:
        return v if v in VALID_CATEGORIES else "OTHER"

    @field_validator("entities")
    @classmethod
    def lowercase_entities(cls, v: list[str]) -> list[str]:
        return [e.lower().strip() for e in v if e.strip()]


_FALLBACK = NERResult(
    entities=[], categories=["OTHER"], primary_entity=None, primary_category="OTHER"
)


def _extract_json_array(raw: str) -> list:
    """Strip markdown fences and extract first JSON array."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start < 0 or end <= start:
        raise ValueError("No JSON array found in response")
    return json.loads(raw[start:end])


def _parse_batch(raw: str, batch_size: int) -> list[NERResult]:
    try:
        data = _extract_json_array(raw)
    except Exception as e:
        logger.warning("JSON parse failed: %s | raw=%r", e, raw[:200])
        return [_FALLBACK] * batch_size

    results: list[NERResult] = []
    for item in data[:batch_size]:
        try:
            results.append(NERResult(**item))
        except Exception as e:
            logger.debug("NERResult validation failed: %s | item=%r", e, item)
            results.append(_FALLBACK)

    while len(results) < batch_size:
        results.append(_FALLBACK)

    return results


def tag_ner_llm(questions: list[str], batch_size: int | None = None) -> dict[str, NERResult]:
    """
    Tag all questions with LLM-based multi-entity NER.

    Args:
        questions: list of FAQ question strings
        batch_size: questions per API call; defaults to defaults.json[topic_modeling][ner_batch_size]

    Returns:
        {question: NERResult} — same key order as input
    """
    defaults = Paths.defaults()
    model = defaults.get("static_llm_model") or defaults.get("llm_model")
    if batch_size is None:
        batch_size = defaults.get("topic_modeling", {}).get("ner_batch_size", 5)

    batches = [questions[i : i + batch_size] for i in range(0, len(questions), batch_size)]
    logger.info(
        "LLM NER: %d questions / %d batches / batch_size=%d / model=%s",
        len(questions), len(batches), batch_size, model,
    )

    tagged: dict[str, NERResult] = {}
    for i, batch in enumerate(batches):
        numbered = "\n".join(f"{j + 1}. {q}" for j, q in enumerate(batch))
        prompt = _PROMPT_TEMPLATE.format(questions=numbered, n=len(batch))

        try:
            result = call_llm(prompt, max_tokens=800, model=model, temperature=0.0, system=_SYSTEM)
            parsed = _parse_batch(result.content, len(batch))
        except Exception as e:
            logger.warning("Batch %d/%d failed: %s — using fallback", i + 1, len(batches), e)
            parsed = [_FALLBACK] * len(batch)

        for question, ner in zip(batch, parsed):
            tagged[question] = ner

        if (i + 1) % 20 == 0 or (i + 1) == len(batches):
            logger.info("  NER progress: %d/%d batches", i + 1, len(batches))

    logger.info("LLM NER complete: %d questions tagged", len(tagged))
    return tagged

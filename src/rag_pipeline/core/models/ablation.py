"""Ablation experiment schemas."""
from __future__ import annotations
from pydantic import BaseModel, Field, computed_field

GENERIC_ENTITIES = {"error", "homework", "course", "model", "project", "issue"}


class Patch(BaseModel, frozen=True):
    # payload patches (no re-run needed)
    null_entity: bool = False
    null_category: bool = False
    null_topics: bool = False
    # ner patches (require topic modeling re-run)
    skip_ner: bool = False
    empty_entity_patterns: bool = False
    # reclassify patches (require topic modeling re-run)
    skip_cluster: bool = False
    skip_rules: bool = False
    # selective payload patches
    null_generic_entities: bool = False
    null_low_confidence_topics: bool = False
    topic_prob_threshold: float = 0.5

    @computed_field
    @property
    def needs_rerun(self) -> bool:
        return (
            self.skip_ner
            or self.empty_entity_patterns
            or self.skip_cluster
            or self.skip_rules
        )

    def apply_to_assignments(self, assignments: list[dict]) -> list[dict]:
        for a in assignments:
            if self.null_entity or self.skip_ner:
                a["ner_primary_entity"] = None
            if self.null_category or self.skip_ner:
                a["ner_category"] = "OTHER"
            if self.null_topics:
                a["topic"] = -1
            if self.null_generic_entities:
                if a.get("ner_primary_entity") in GENERIC_ENTITIES:
                    a["ner_primary_entity"] = None
            if self.null_low_confidence_topics:
                if a.get("topic_probability", 1.0) < self.topic_prob_threshold:
                    a["topic"] = -1
        return assignments

    def env(self) -> dict:
        return {}

    def label(self) -> str:
        parts = []
        if self.null_entity:                parts.append("no_entity")
        if self.null_category:              parts.append("no_category")
        if self.null_topics:                parts.append("no_topics")
        if self.skip_ner:                   parts.append("skip_ner")
        if self.empty_entity_patterns:      parts.append("empty_patterns")
        if self.skip_cluster:               parts.append("skip_cluster")
        if self.skip_rules:                 parts.append("skip_rules")
        if self.null_generic_entities:      parts.append("no_generic_entity")
        if self.null_low_confidence_topics: parts.append(f"low_conf_{str(self.topic_prob_threshold).replace('.', '')}")
        return "+".join(parts) if parts else "baseline"


class ExperimentResult(BaseModel):
    name: str
    patch: str
    configs: list[str]
    model: str
    timestamp: str
    metrics: dict
    result_files: list[str]
    git_commit: str = "unknown"
    corpus_size: int | None = None

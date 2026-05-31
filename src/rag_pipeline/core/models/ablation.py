"""Ablation experiment schemas."""
from __future__ import annotations
from pydantic import BaseModel, Field, computed_field
from rag_pipeline.ingestion.benchmark_types import MetricSummary

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
    skip_url_expand: bool = False
    use_llm_ner: bool = False

    @computed_field
    @property
    def needs_rerun(self) -> bool:
        return (
            self.skip_ner
            or self.empty_entity_patterns
            or self.skip_cluster
            or self.skip_rules
            or self.skip_url_expand
        )

    def apply_to_assignments(self, assignments: list[dict]) -> list[dict]:
        llm_map = None
        if self.use_llm_ner:
            import json
            from rag_pipeline.core.paths import Paths
            llm_map = json.load(open(Paths.experiments_dir() / "llm_ner_merged.json"))
        for a in assignments:
            if self.null_entity or self.skip_ner:
                a["ner_primary_entity"] = None
                a["ner_entities"] = []
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
            if llm_map:
                doc_id = a.get("id")
                if doc_id in llm_map:
                    a["ner_primary_entity"] = llm_map[doc_id]["ner_primary_entity"]
                    a["ner_entities"] = llm_map[doc_id]["ner_entities"]
        return assignments

    def env(self) -> dict:
        e = {}
        if self.skip_url_expand:
            e["RAG_SKIP_URL_EXPAND"] = "1"
        return e

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
        if self.use_llm_ner:               parts.append("llm_ner")
        if self.skip_url_expand:            parts.append("no_url_expand")
        return "+".join(parts) if parts else "baseline"


class ExperimentResult(BaseModel):
    name: str
    patch: str
    configs: list[str]
    model: str
    timestamp: str
    metrics: dict[str, MetricSummary]
    result_files: list[str]
    git_commit: str = "unknown"
    corpus_size: int | None = None

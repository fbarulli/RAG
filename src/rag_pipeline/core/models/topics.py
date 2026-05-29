"""Topic modeling schemas."""
from __future__ import annotations
from typing import Optional
import json
from pydantic import BaseModel


class TopicAssignment(BaseModel, frozen=True):
    """A single question with its assigned topic and NER category."""
    question: str
    topic: int
    topic_name: str
    ner_category: str
    model: str


class TopicAssignments(BaseModel):
    """Container for all topic modeling results. Single source of truth."""
    models: list[str]
    results: dict[str, dict]  # model -> {metadata, assignments}

    def save(self, path=None) -> None:
        from rag_pipeline.core.paths import Paths
        if path is None:
            path = Paths.topic_assignments()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load(cls) -> TopicAssignments:
        from rag_pipeline.core.paths import Paths
        path = Paths.topic_assignments()
        if not path.exists():
            raise FileNotFoundError(f"Topic assignments not found at {path}")
        with open(path) as f:
            data = json.load(f)
        return cls(models=data["models"], results=data["results"])

    def iter_assignments(self):
        for model_data in self.results.values():
            yield from model_data.get("assignments", [])

    def get_sample(self, n: int = 3) -> list[dict]:
        results = []
        for assignment in self.iter_assignments():
            results.append(assignment)
            if len(results) >= n:
                break
        return results

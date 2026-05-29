"""Embedding model registry schema. Single source of truth for configs/models.json."""
from __future__ import annotations
import json
from pydantic import BaseModel


class EmbeddingModel(BaseModel, frozen=True):
    name: str
    short_name: str
    collection: str
    es_index: str
    dims: int
    trust_remote_code: bool = False
    enabled: bool = True
    tier: str = "balanced"
    winner: bool = False
    description: str = ""


class ModelRegistry(BaseModel, frozen=True):
    models: list[EmbeddingModel]

    @classmethod
    def load(cls) -> ModelRegistry:
        from rag_pipeline.core.paths import Paths
        with open(Paths.models_config(), encoding="utf-8") as f:
            data = json.load(f)
        return cls(models=data["models"])

    def production(self) -> EmbeddingModel:
        from rag_pipeline.core.paths import Paths
        name = Paths.defaults()["production_model"]
        try:
            return next(m for m in self.models if m.name == name)
        except StopIteration:
            raise ValueError(f"production_model '{name}' not found in models.json")

    def enabled_models(self) -> list[EmbeddingModel]:
        return [m for m in self.models if m.enabled]

    def get(self, name: str) -> EmbeddingModel:
        try:
            return next(m for m in self.models if m.name == name)
        except StopIteration:
            raise ValueError(f"Model '{name}' not found in registry")

"""
rag_pipeline/ingestion/reranker_config.py
Typed config loading for rerankers.json using Pydantic.
All defaults live in rerankers.json — none here.
"""
import json
import logging
from typing import Optional
from pydantic import BaseModel
from ..core.paths import Paths

logger = logging.getLogger(__name__)


class RerankerModelConfig(BaseModel):
    name: str
    model: str
    max_length: int
    reranker: bool
    quantization: bool


class RerankerTrainingConfig(BaseModel):
    default_model_key: str
    max_candidates: int
    num_hard_negatives: int
    sample_size: int
    min_sample_for_full_train: int
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_train_batch_size_gpu: int
    warmup_steps: int
    logging_steps: int
    save_steps: int
    eval_steps: int
    save_strategy: str
    eval_strategy: str
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    save_total_limit: int
    fp16: bool
    dataloader_num_workers: int
    overwrite_output_dir: bool
    course_filter: Optional[str]
    boost_question: float
    boost_text: float
    rrf_k: int


class RerankerInferenceConfig(BaseModel):
    providers_priority: list[str]
    cpu_batch_size_small: int
    cpu_batch_size_medium: int
    cpu_batch_size_large: int
    gpu_batch_size: int
    cpu_threshold_small: int
    cpu_threshold_medium: int


class RerankerConfig(BaseModel):
    models: list[RerankerModelConfig]
    training: RerankerTrainingConfig
    inference: RerankerInferenceConfig


def load_reranker_config() -> RerankerConfig:
    path = Paths.base() / "configs" / "rerankers.json"
    data = json.load(open(path, encoding="utf-8"))
    # strip comment key before validation
    data.pop("_comment", None)
    cfg = RerankerConfig.model_validate(data)
    logger.info("Loaded reranker config from configs/rerankers.json")
    return cfg


def get_model_config(model_key: str) -> RerankerModelConfig:
    cfg = load_reranker_config()
    for m in cfg.models:
        if m.name == model_key or m.model == model_key:
            return m
    raise KeyError(f"Model key '{model_key}' not found in rerankers.json. Available: {[m.name for m in cfg.models]}")

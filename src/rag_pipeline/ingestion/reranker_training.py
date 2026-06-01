"""
reranker_training.py
Fine-tune a cross-encoder reranker on domain triples.

Usage:
    uv run python -m rag_pipeline.ingestion.reranker_training \
        --triples experiments/reranker_training/triples_sample_965.json \
        --model-key TinyBERT \
        --output-dir experiments/reranker_finetuned/tinybert
"""
import argparse
import json
import logging
from pathlib import Path
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_config():
    from rag_pipeline.ingestion.reranker_config import load_reranker_config
    return load_reranker_config()


class TrainingTriple(BaseModel):
    query: str
    positive: str
    hard_negatives: list[str]


def load_triples(path: Path) -> list[TrainingTriple]:
    raw = json.load(open(path))
    triples = [TrainingTriple.model_validate(t) for t in raw]
    logger.info(f"Loaded {len(triples)} triples from {path}")
    return triples


def build_dataset(triples: list[dict], split: float = 0.9):
    from datasets import Dataset

    train_rows, eval_rows = [], []
    cut = int(len(triples) * split)

    import random

    for t in triples[:cut]:
        train_rows.append({"query": t.query, "positive": t.positive, "negative": random.choice(t.hard_negatives)})
    for t in triples[cut:]:
        eval_rows.append({"query": t.query, "positive": t.positive, "negative": t.hard_negatives})

    logger.info(f"Train: {len(train_rows)} | Eval: {len(eval_rows)}")
    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows)


def train(
    triples_path: Path,
    model_name: str,
    output_dir: Path,
    cfg: dict,
):
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
    from sentence_transformers.cross_encoder.losses import MultipleNegativesRankingLoss

    
    t = cfg.training
    triples = load_triples(triples_path)
    train_dataset, eval_dataset = build_dataset(triples)

    logger.info(f"Loading base model: {model_name}")
    model = CrossEncoder(model_name)
    loss = MultipleNegativesRankingLoss(model)  

    # Evaluator from eval split
    eval_samples = [
        {"query": r["query"], "positive": [r["positive"]], "negative": r["negative"]}
        for r in eval_dataset
    ]
    evaluator = CERerankingEvaluator(eval_samples, name="reranker_eval")

    output_dir.mkdir(parents=True, exist_ok=True)

    args = CrossEncoderTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=t.num_train_epochs,
        per_device_train_batch_size=t.per_device_train_batch_size,
        warmup_steps=t.warmup_steps,
        logging_steps=t.logging_steps,
        save_steps=t.save_steps,
        eval_steps=t.eval_steps,
        save_strategy=t.save_strategy,
        eval_strategy=t.eval_strategy,
        load_best_model_at_end=t.load_best_model_at_end,
        metric_for_best_model=t.metric_for_best_model,
        greater_is_better=t.greater_is_better,
        save_total_limit=t.save_total_limit,
        fp16=t.fp16,
        dataloader_num_workers=t.dataloader_num_workers,
        overwrite_output_dir=t.overwrite_output_dir,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        evaluator=evaluator,
        loss=loss,
    )
    
    logger.info("Starting training...")
    trainer.train()
    model.save_pretrained(str(output_dir / "final"))
    logger.info(f"Model saved to {output_dir / 'final'}")


def main():
    from rag_pipeline.ingestion.reranker_config import get_model_config

    cfg = load_config()
    t = cfg.training

    from configs.benchmark_cli import create_reranker_training_parser
    args = create_reranker_training_parser().parse_args()

    from rag_pipeline.ingestion.reranker_config import get_model_config
    for model_key in args.model_keys:
        model_cfg = get_model_config(model_key)
        output_dir = (args.output_dir or Path("experiments/reranker_finetuned")) / model_key.lower()
        logger.info(f"Training {model_key} → {output_dir}")
        train(
            triples_path=args.triples,
            model_name=model_cfg.model,
            output_dir=output_dir,
            cfg=cfg,
        )


if __name__ == "__main__":
    main()

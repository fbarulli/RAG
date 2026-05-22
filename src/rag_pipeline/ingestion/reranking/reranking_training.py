"""
src/rag_pipeline/ingestion/reranking/_reranking_training.py
Main entry point for reranker training.
"""
import sys
import traceback

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments

from ...core.paths import Paths
from ...core.logging import get_logger
from ..reranker_config import get_model_config
from .reranking_triples import load_train_data, get_or_generate_triples
from .reranking_evaluator import create_proper_evaluator

logger = get_logger(__name__)


def prepare_training_examples(triples: list) -> list:
    examples = []
    for t in triples:
        q = t["query"]
        examples.append([q, t["positive"], 1.0])
        for neg in t.get("hard_negatives", [])[:5]:
            examples.append([q, neg, 0.0])
    return examples


def main() -> None:
    sample_size = 50
    logger.info(f"Starting reranker training | sample={sample_size}")

    try:
        target_config = get_model_config("bge-reranker-base")
        model_name = target_config["model"]
        logger.info(f"Target: {model_name}")

        train_data = load_train_data(sample_size=sample_size)
        queries = [item.get("question") or item.get("query", "") for item in train_data]
        corpus = [{"id": item.get("es_id"), "text": item.get("answer", "")} for item in train_data]

        triples = get_or_generate_triples(queries, corpus, sample_size)
        logger.info(f"Using {len(triples)} triples")

        train_examples = prepare_training_examples(triples)

        if sample_size >= 100:
            output_dir = Paths.experiments_dir() / "reranker_models" / "bge-reranker-base-finetuned"
            output_dir.mkdir(parents=True, exist_ok=True)

            model = CrossEncoder(model_name, num_labels=1, max_length=target_config.get("max_length", 512))

            training_args = CrossEncoderTrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=2,
                per_device_train_batch_size=8,
                warmup_steps=30,
                logging_steps=10,
                save_steps=100,
                fp16=True,
                dataloader_num_workers=2,
                overwrite_output_dir=True,
            )

            evaluator = create_proper_evaluator(train_examples)

            trainer = CrossEncoderTrainer(
                model=model,
                args=training_args,
                train_dataset=train_examples,
                evaluator=evaluator,
            )

            logger.info("Starting fine-tuning...")
            trainer.train()
            logger.info(f"✅ Model saved to {output_dir}")
        else:
            logger.info("Practice run complete. Increase sample_size for full training.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
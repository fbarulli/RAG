"""
src/rag_pipeline/ingestion/reranking/reranking_training.py
Main entry point for reranker training.
"""
import sys
import traceback
from datasets import Dataset
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments

from ...core.paths import Paths
from ...core.logging import get_logger
from ..reranker_config import get_model_config, load_reranker_config
from ..benchmark_config import BenchmarkConfig
from ..onnx_bench import setup_bi_encoder_context
from .reranking_triples import load_train_data, get_or_generate_triples
from .reranking_evaluator import create_proper_evaluator

logger = get_logger(__name__)


def prepare_training_examples(triples: list) -> Dataset:
    rows = []
    for t in triples:
        q = t["query"]
        rows.append({"query": q, "document": t["positive"], "label": 1.0})
        for neg in t.get("hard_negatives", [])[:5]:
            rows.append({"query": q, "document": neg, "label": 0.0})
    return Dataset.from_list(rows)


def main() -> None:
    cfg = load_reranker_config()["training"]
    sample_size = cfg["sample_size"]
    logger.info(f"Starting reranker training | sample={sample_size}")
    try:
        config = BenchmarkConfig.from_defaults()
        client = config.qdrant_client
        embedding_entry = setup_bi_encoder_context(config)
        collection = embedding_entry["collection"]
        embedding_model = SentenceTransformer(
            embedding_entry["name"],
            trust_remote_code=embedding_entry.get("trust_remote_code", False)
        )
        retrieval_config = {
            "boost_question": cfg["boost_question"],
            "boost_text": cfg["boost_text"],
            "rrf_k": cfg["rrf_k"],
            "course_filter": cfg["course_filter"],
        }
        target_config = get_model_config(cfg["default_model_key"])
        model_name = target_config["model"]
        logger.info(f"Target: {model_name}")

        train_data = load_train_data(sample_size=sample_size)

        try:
            topic_map = config.get_topic_map(embedding_entry["name"])
            logger.info(f"Loaded topic map with {len(topic_map)} entries")
        except Exception as e:
            logger.warning(f"Could not load topic map, entity boosting disabled: {e}")
            topic_map = None

        triples = get_or_generate_triples(
            train_items=train_data,
            sample_size=sample_size,
            client=client,
            collection=collection,
            embedding_model=embedding_model,
            retrieval_config=retrieval_config,
            topic_map=topic_map,
        )
        logger.info(f"Using {len(triples)} triples")
        train_examples = prepare_training_examples(triples)

        if sample_size >= cfg["min_sample_for_full_train"]:
            output_dir = Paths.experiments_dir() / "reranker_models" / f"{cfg['default_model_key']}-finetuned"
            output_dir.mkdir(parents=True, exist_ok=True)

            import torch
            on_gpu = torch.cuda.is_available()
            batch_size = cfg["per_device_train_batch_size_gpu"] if on_gpu else cfg["per_device_train_batch_size"]
            logger.info(f"Training on {'GPU' if on_gpu else 'CPU'} | batch_size={batch_size} | fp16={cfg['fp16'] and on_gpu}")

            model = CrossEncoder(model_name, num_labels=1, max_length=target_config["max_length"])
            training_args = CrossEncoderTrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=cfg["num_train_epochs"],
                per_device_train_batch_size=batch_size,
                warmup_steps=cfg["warmup_steps"],
                logging_steps=cfg["logging_steps"],
                save_steps=cfg["save_steps"],
                eval_steps=cfg["eval_steps"],
                save_strategy=cfg["save_strategy"],
                evaluation_strategy=cfg["evaluation_strategy"],
                load_best_model_at_end=cfg["load_best_model_at_end"],
                metric_for_best_model=cfg["metric_for_best_model"],
                greater_is_better=cfg["greater_is_better"],
                save_total_limit=cfg["save_total_limit"],
                fp16=cfg["fp16"] and on_gpu,
                dataloader_num_workers=cfg["dataloader_num_workers"],
                overwrite_output_dir=cfg["overwrite_output_dir"],
                resume_from_checkpoint=True,
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
            model.save_pretrained(str(output_dir))
            tokenizer = trainer.processing_class
            if tokenizer:
                tokenizer.save_pretrained(str(output_dir))
            logger.info(f"✅ Model saved to {output_dir}")
        else:
            logger.info("Practice run complete. Increase sample_size in rerankers.json for full training.")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

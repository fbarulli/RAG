# --------------------------------------------------------------
# 1️⃣  Imports & helpers (same as the files you sent)
# --------------------------------------------------------------
import json, logging, pathlib, random, traceback
from pathlib import Path

import ray, torch, numpy as np
from ray import tune
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from rag_pipeline.ingestion.reranking.reranking_config_ray import RayTrainingConfig
from rag_pipeline.ingestion.reranking.reranking_dataset import RerankerDataset, make_collate_fn
from rag_pipeline.ingestion.reranking.reranking_loss import AdaptiveListwiseLoss

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------
# 2️⃣  Utility: compute MRR@10 from scores + mask
# --------------------------------------------------------------
def mrr_at_k(scores: torch.Tensor, mask: torch.Tensor, k: int = 10) -> float:
    # scores: (B, G) ; mask: (B, G) bool
    # Positive is always column 0
    ranks = torch.argsort(scores, dim=-1, descending=True)
    # Where does the positive fall?
    pos_rank = (ranks == 0).nonzero(as_tuple=False)[:, 1] + 1  # 1‑based rank
    # ignore padded candidates
    pos_rank = torch.where(mask[:, 0], pos_rank, torch.tensor(k + 1, device=scores.device))
    recip = (pos_rank <= k).float() / pos_rank.float()
    return recip.mean().item()

# --------------------------------------------------------------
# 3️⃣  Train‑loop callable – works for both screen & full‑train
# --------------------------------------------------------------
def train_loop_per_worker(config: dict) -> None:
    import ray.train.torch as ray_torch
    device = ray_torch.get_device()
    log.info("Worker %s on %s", ray.train.get_context().get_world_rank(), device)

    # ---- tokenizer / model -------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=config["num_labels"]
    )
    model = ray_torch.prepare_model(model)

    # ---- dataset ------------------------------------------------------------
    dataset = RerankerDataset(config["triples_path"], config["max_negatives"])
    collate = make_collate_fn(tokenizer, config["max_length"])
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,                     # DistributedSampler will shuffle later
        collate_fn=collate,
        num_workers=config["dataloader_num_workers"],
        pin_memory=True,
    )
    loader = ray_torch.prepare_data_loader(loader)

    # ---- optimiser / scheduler -----------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    total_steps = len(loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0,
                                                total_iters=warmup_steps)

    loss_fn = AdaptiveListwiseLoss(alpha=config["alpha"]).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=config["fp16"])

    # --------------------------------------------------------------
    # 4️⃣  Training + optional validation every N steps
    # --------------------------------------------------------------
    log_every = config.get("log_every_n_steps", 50)
    val_every = config.get("val_every_n_steps", 200)   # screen: small; full‑train: larger

    global_step = 0
    for epoch in range(config["epochs"]):
        model.train()
        for batch in loader:
            # ---------- move everything to device in ONE go ----------
            for k, v in batch.items():
                batch[k] = v.to(device, non_blocking=True)

            B, G, L = batch["input_ids"].shape
            flat = {
                "input_ids":      batch["input_ids"].view(B * G, L),
                "attention_mask": batch["attention_mask"].view(B * G, L),
            }
            if "token_type_ids" in batch:
                flat["token_type_ids"] = batch["token_type_ids"].view(B * G, L)

            with torch.cuda.amp.autocast(enabled=config["fp16"]):
                logits = model(**flat).logits.squeeze(-1)   # (B, G)
                loss = loss_fn(logits, batch["mask"])

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # --------------------------------------------------
            # 5️⃣  Reporting
            # --------------------------------------------------
            if global_step % log_every == 0:
                log.info("step %d – loss %.4f", global_step, loss.item())
                ray.train.report({"step": global_step, "loss": loss.item()})

            # --------------------------------------------------
            # 6️⃣  Validation checkpoint
            # --------------------------------------------------
            if global_step % val_every == 0:
                model.eval()
                all_scores, all_mask = [], []
                with torch.no_grad():
                    for vbatch in loader:               # reuse same loader (no gradient)
                        for k, v in vbatch.items():
                            vbatch[k] = v.to(device, non_blocking=True)
                        B2, G2, L2 = vbatch["input_ids"].shape
                        flat2 = {
                            "input_ids":      vbatch["input_ids"].view(B2 * G2, L2),
                            "attention_mask": vbatch["attention_mask"].view(B2 * G2, L2),
                        }
                        if "token_type_ids" in vbatch:
                            flat2["token_type_ids"] = vbatch["token_type_ids"].view(B2 * G2, L2)
                        logits2 = model(**flat2).logits.squeeze(-1).view(B2, G2)
                        all_scores.append(logits2.detach())
                        all_mask.append(vbatch["mask"])
                scores = torch.cat(all_scores, dim=0)
                mask   = torch.cat(all_mask,   dim=0)
                mrr10 = mrr_at_k(scores, mask, k=10)
                ray.train.report({"validation/mrr@10": mrr10, "step": global_step})

            global_step += 1

    # --------------------------------------------------------------
    # 7️⃣  Save (only rank 0)
    # --------------------------------------------------------------
    if ray.train.get_context().get_world_rank() == 0:
        out = Path(config["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(out)
        tokenizer.save_pretrained(out)

# --------------------------------------------------------------
# 8️⃣  Helper: build a *tiny* config for the screen step
# --------------------------------------------------------------
def build_screen_cfg(base_cfg: RayTrainingConfig, max_queries: int = 90) -> dict:
    # we reuse the same JSON but override a few fields
    cfg = base_cfg.to_dict()
    cfg.update(
        epochs=2,                     # quick run
        batch_size=8,
        log_every_n_steps=20,
        val_every_n_steps=40,
        # Randomly sample a subset of the triples file (keep the original file)
        triples_path=cfg["triples_path"],   # will be filtered by the dataset below
        max_negatives=cfg["max_negatives"],
    )
    # Store the *sample* size so the dataset can slice itself
    cfg["sample_size"] = max_queries
    return cfg

# --------------------------------------------------------------
# 9️⃣  Custom Dataset that can subsample on the fly (only used for screen)
# --------------------------------------------------------------
class RerankerDataset(RerankerDataset):   # subclass to add sampling
    def __init__(self, triples_path: str, max_negatives: int, sample_size: int | None = None):
        super().__init__(triples_path, max_negatives)
        if sample_size:
            # Randomly keep `sample_size` groups (deterministic seed for reproducibility)
            random.seed(42)
            self.groups = random.sample(self.groups, min(sample_size, len(self.groups)))

# --------------------------------------------------------------
# 10️⃣  Ray‑Tune orchestration
# --------------------------------------------------------------
def run_screen_and_select_top_k(
    model_names: list[str],
    k_top: int = 3,
    queries_for_screen: int = 90,
) -> list[str]:
    """
    1️⃣  Launch a Ray‑Tune trial **per model** on the *screen* dataset.
    2️⃣  Gather the best validation MRR@10 for each model.
    3️⃣  Return the names of the top‑k models.
    """
    ray.init(ignore_reinit_error=True)

    base_cfg = RayTrainingConfig.from_rerankers_json()
    # Override the sample size globally so the dataset knows to truncate
    base_cfg.apply_cli_overrides(
        Namespace(sample_size=queries_for_screen)  # dummy namespace for typing only
    )

    # -----------------------------------------------------------------
    # Define the hyper‑parameter search space – only the model_name changes
    # -----------------------------------------------------------------
    search_space = {
        "model_name": tune.grid_search(model_names),
        # The remainder of the config comes from `base_cfg`
    }

    # -----------------------------------------------------------------
    # Build a Trainable that merges `base_cfg` + Ray‑Tune overrides
    # -----------------------------------------------------------------
    def trainable(config):
        # Merge static base config with the dynamically sampled model_name
        merged_cfg = base_cfg.to_dict()
        merged_cfg.update(config)
        # Attach the screen‑size override
        merged_cfg["sample_size"] = queries_for_screen
        # The trainer will respect `merged_cfg["sample_size"]` via the Dataset subclass
        trainer = TorchTrainer(
            train_loop_per_worker=train_loop_per_worker,
            train_loop_config=merged_cfg,
            scaling_config=ScalingConfig(num_workers=1, use_gpu=True),  # 1 GPU per trial
        )
        result = trainer.fit()
        # Return the best validation metric for Ray‑Tune's scheduler
        best = result.metrics.get("validation/mrr@10", 0.0)
        tune.report(mrr10=best)

    analysis = tune.run(
        trainable,
        name="reranker_screen",
        config=search_space,
        metric="mrr10",
        mode="max",
        resources_per_trial={"cpu": 2, "gpu": 1},
        num_samples=1,
        verbose=1,
    )

    # -----------------------------------------------------------------
    # Pick the top‑k model names
    # -----------------------------------------------------------------
    df = analysis.trials_dataframe()
    top_models = (
        df.sort_values("mrr10", ascending=False)
        .head(k_top)["config/model_name"]
        .tolist()
    )
    log.info("Screen complete – top %d models: %s", k_top, top_models)
    return top_models

# --------------------------------------------------------------
# 11️⃣  Full‑train on the top‑k models (parallel)
# --------------------------------------------------------------
def full_train_top_k(model_names: list[str]):
    base_cfg = RayTrainingConfig.from_rerankers_json()
    # No sample‑size override → use all 900 queries
    search_space = {"model_name": tune.grid_search(model_names)}

    def trainable(config):
        merged_cfg = base_cfg

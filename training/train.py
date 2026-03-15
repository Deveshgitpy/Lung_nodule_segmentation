"""
training/train.py
==================
Entry point for training the 3D U-Net lung nodule segmentation model.

Usage:
    python training/train.py --config configs/training_config.yaml

This script:
  1. Parses CLI arguments and loads YAML config
  2. Sets global random seeds for reproducibility
  3. Builds model, optimizer, scheduler, and loss function
  4. Builds DataLoaders with augmentation
  5. Instantiates the Trainer
  6. Handles checkpoint resumption
  7. Runs training and reports final metrics
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from augmentation.medical_augmentations import get_training_transforms, get_val_transforms
from datasets.luna16_dataset import get_dataloaders
from models.unet import build_unet_from_config
from training.loss_functions import build_loss_from_config
from training.trainer import Trainer, build_scheduler_with_warmup
from utils.checkpoint_utils import CheckpointManager
from utils.logging_utils import MetricLogger, setup_logging

logger = logging.getLogger(__name__)


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for full reproducibility.

    PyTorch documentation recommends setting both the Python random seed,
    NumPy seed, and PyTorch seeds. When deterministic=True, we also disable
    cuDNN benchmark mode (which trades speed for non-determinism).

    Note: Full determinism may reduce GPU throughput by ~10–20%.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # For multi-GPU

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 1.11+ additional determinism
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    logger.info(f"Global seed set to {seed} (deterministic={deterministic})")


# ─── Config Handling ──────────────────────────────────────────────────────────

def load_config(config_path: str, overrides: dict) -> dict:
    """
    Load YAML config and apply CLI overrides.

    CLI overrides take precedence over values in the YAML file.
    This allows launching hyperparameter sweeps without creating
    new config files for each experiment.

    Args:
        config_path: Path to training_config.yaml.
        overrides:   Dict of CLI key-value overrides.

    Returns:
        Merged configuration dict.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply overrides
    for key, value in overrides.items():
        if value is not None:
            # Support nested keys with dot notation (e.g., "training.batch_size")
            keys = key.split(".")
            cfg = config
            for k in keys[:-1]:
                cfg = cfg.setdefault(k, {})
            cfg[keys[-1]] = value

    return config


# ─── Optimizer Builder ────────────────────────────────────────────────────────

def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    opt_cfg = config["optimizer"]
    name = opt_cfg.get("name", "Adam")

    params = [p for p in model.parameters() if p.requires_grad]

    if name == "Adam":
        return torch.optim.Adam(
            params,
            lr=opt_cfg["learning_rate"],
            weight_decay=opt_cfg["weight_decay"],
            betas=tuple(opt_cfg["betas"]),
            eps=opt_cfg["eps"],
        )
    elif name == "AdamW":
        return torch.optim.AdamW(
            params,
            lr=opt_cfg["learning_rate"],
            weight_decay=opt_cfg["weight_decay"],
        )
    elif name == "SGD":
        return torch.optim.SGD(
            params,
            lr=opt_cfg["learning_rate"],
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg["weight_decay"],
            nesterov=opt_cfg.get("nesterov", True),
        )
    else:
        raise ValueError(f"Unknown optimizer: '{name}'")


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    """Main training function."""

    # ── Setup ────────────────────────────────────────────────────────────────
    setup_logging(level=logging.INFO)

    # Load config with CLI overrides
    overrides = {}
    if args.batch_size is not None:
        overrides["training.batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides["optimizer.learning_rate"] = args.learning_rate
    if args.num_epochs is not None:
        overrides["training.num_epochs"] = args.num_epochs
    if args.experiment_name is not None:
        overrides["experiment.name"] = args.experiment_name
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.use_wandb:
        overrides["logging.use_wandb"] = True

    config = load_config(args.config, overrides)
    logger.info(f"Loaded config from: {args.config}")

    # Determine device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available. Training on CPU (extremely slow).")

    logger.info(f"Training device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(
            f"VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB"
        )

    # Set seeds
    set_global_seed(
        config["seed"],
        deterministic=config.get("deterministic", True),
    )

    # ── Logging & Checkpointing ──────────────────────────────────────────────
    output_dir = Path(config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_logger = MetricLogger(config)
    checkpoint_manager = CheckpointManager(config)

    # ── Data Pipeline ────────────────────────────────────────────────────────
    logger.info("Building data pipeline...")

    train_transform = get_training_transforms(config["augmentation"])
    val_transform = get_val_transforms()

    train_loader, val_loader, test_loader = get_dataloaders(
        config=config,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    logger.info(
        f"Dataset sizes — "
        f"Train: {len(train_loader.dataset)} | "
        f"Val: {len(val_loader.dataset)} | "
        f"Test: {len(test_loader.dataset)}"
    )

    # ── Model ────────────────────────────────────────────────────────────────
    logger.info("Building UNet3D model...")
    model = build_unet_from_config(config)
    model = model.to(device)

    param_info = model.get_num_parameters()
    logger.info(
        f"UNet3D — Total params: {param_info['total']:,} | "
        f"Trainable: {param_info['trainable']:,}"
    )

    # ── Loss, Optimizer, Scheduler ───────────────────────────────────────────
    criterion = build_loss_from_config(config).to(device)
    optimizer = build_optimizer(model, config)

    sched_cfg = config["scheduler"]
    scheduler = build_scheduler_with_warmup(
        optimizer=optimizer,
        warmup_epochs=sched_cfg["warmup_epochs"],
        total_epochs=config["training"]["num_epochs"],
        eta_min=sched_cfg["eta_min"],
        warmup_start_lr=sched_cfg["warmup_start_lr"],
        base_lr=config["optimizer"]["learning_rate"],
    )

    # ── Resume from Checkpoint ───────────────────────────────────────────────
    start_epoch = 0
    resume_path = config["checkpointing"].get("resume_from")
    if resume_path and Path(resume_path).exists():
        logger.info(f"Resuming from checkpoint: {resume_path}")
        start_epoch = checkpoint_manager.load(
            resume_path, model, optimizer, scheduler
        )
        logger.info(f"Resumed from epoch {start_epoch}")

    # ── Log Config to WandB ──────────────────────────────────────────────────
    metric_logger.log_config(config)

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        checkpoint_manager=checkpoint_manager,
        metric_logger=metric_logger,
    )

    best_metrics = trainer.fit(start_epoch=start_epoch)

    # ── Final Evaluation on Test Set ─────────────────────────────────────────
    logger.info("Running final evaluation on test set...")

    # Load best checkpoint for evaluation
    best_checkpoint = checkpoint_manager.get_best_checkpoint_path()
    if best_checkpoint and best_checkpoint.exists():
        checkpoint_manager.load(str(best_checkpoint), model)
        logger.info(f"Loaded best checkpoint: {best_checkpoint}")

    trainer.val_loader = test_loader
    test_metrics = trainer._validate_epoch(epoch=-1)
    logger.info("Test Set Results:")
    for metric_name, value in test_metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")

    metric_logger.log_step({"test/" + k: v for k, v in test_metrics.items()})
    metric_logger.finish()

    logger.info("Training pipeline completed successfully.")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 3D U-Net for lung nodule segmentation on LUNA16.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, default="configs/training_config.yaml",
        help="Path to YAML training configuration file.",
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None,
        help="Override experiment name from config.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch_size from config.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None,
        help="Override learning_rate from config.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=None,
        help="Override num_epochs from config.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed from config.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Training device (e.g. 'cuda', 'cuda:1', 'cpu').",
    )
    parser.add_argument(
        "--use_wandb", action="store_true",
        help="Enable Weights & Biases logging.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

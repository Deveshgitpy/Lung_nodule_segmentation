"""
training/trainer.py
====================
Core training loop, validation loop, and learning rate scheduling
for the 3D U-Net lung nodule segmentation model.

Features:
  - Mixed precision training (FP16) via torch.cuda.amp
  - Gradient clipping to stabilise 3D conv training
  - Gradient accumulation to simulate larger batch sizes
  - Learning rate warm-up followed by cosine annealing
  - Early stopping on validation Dice coefficient
  - Comprehensive per-step and per-epoch metric logging
  - Optional WandB integration
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import compute_metrics
from utils.checkpoint_utils import CheckpointManager
from utils.logging_utils import MetricLogger

logger = logging.getLogger(__name__)


# ─── Learning Rate Warmup ─────────────────────────────────────────────────────

def build_scheduler_with_warmup(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    eta_min: float,
    warmup_start_lr: float,
    base_lr: float,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a scheduler that linearly warms up the LR over `warmup_epochs`,
    then applies cosine annealing for the remaining epochs.

    Warm-up prevents large gradient steps at the start of training when
    weight updates would otherwise be driven by random initialisation noise.

    Args:
        optimizer:       The Adam optimizer instance.
        warmup_epochs:   Number of warm-up epochs.
        total_epochs:    Total training epochs.
        eta_min:         Minimum LR for cosine phase.
        warmup_start_lr: LR at epoch 0 of warm-up.
        base_lr:         Peak LR (end of warm-up = start of cosine).

    Returns:
        SequentialLR scheduler wrapping warm-up + cosine stages.
    """
    # Warm-up: linear ramp from warmup_start_lr to base_lr
    def warmup_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return warmup_start_lr / base_lr + (1.0 - warmup_start_lr / base_lr) * (
                epoch / max(1, warmup_epochs)
            )
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # Cosine annealing: base_lr → eta_min over remaining epochs
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=eta_min,
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )


# ─── Trainer Class ────────────────────────────────────────────────────────────

class Trainer:
    """
    Manages the full training pipeline for U-Net segmentation.

    Args:
        model:              UNet3D model.
        train_loader:       Training DataLoader.
        val_loader:         Validation DataLoader.
        criterion:          Loss function (DiceBCELoss).
        optimizer:          Adam optimizer.
        scheduler:          LR scheduler.
        config:             Full training configuration dict.
        device:             torch.device for training.
        checkpoint_manager: Handles saving/loading checkpoints.
        metric_logger:      Handles console + WandB logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        config: dict,
        device: torch.device,
        checkpoint_manager: CheckpointManager,
        metric_logger: MetricLogger,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        self.metric_logger = metric_logger

        # Training config shortcuts
        train_cfg = config["training"]
        self.num_epochs = train_cfg["num_epochs"]
        self.gradient_clip = train_cfg["gradient_clip_norm"]
        self.accum_steps = train_cfg["accumulate_grad_batches"]
        self.mixed_precision = train_cfg["mixed_precision"]
        self.val_every = config["logging"]["val_every_n_epochs"]
        self.log_every = config["logging"]["log_every_n_steps"]

        # Early stopping
        es_cfg = train_cfg
        self.early_stopping = es_cfg.get("early_stopping", True)
        self.es_patience = es_cfg.get("early_stopping_patience", 30)
        self.es_metric = es_cfg.get("early_stopping_metric", "val_dice")
        self.es_mode = es_cfg.get("early_stopping_mode", "max")
        self._es_counter = 0
        self._best_val_metric = -float("inf") if self.es_mode == "max" else float("inf")
        self._should_stop = False

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.mixed_precision)

        # Global step counter for logging
        self.global_step = 0

    # ── Training Loop ─────────────────────────────────────────────────────────

    def fit(self, start_epoch: int = 0) -> Dict[str, float]:
        """
        Run the full training loop.

        Args:
            start_epoch: Starting epoch (for resuming from checkpoint).

        Returns:
            Best validation metrics dict.
        """
        logger.info(
            f"Starting training for {self.num_epochs} epochs "
            f"on device: {self.device}"
        )

        best_metrics: Dict[str, float] = {}

        for epoch in range(start_epoch, self.num_epochs):
            epoch_start = time.time()

            # ── Train epoch ──────────────────────────────────────────────────
            train_metrics = self._train_epoch(epoch)

            # ── Validation epoch ─────────────────────────────────────────────
            val_metrics: Dict[str, float] = {}
            if (epoch + 1) % self.val_every == 0 or epoch == self.num_epochs - 1:
                val_metrics = self._validate_epoch(epoch)

                # Check early stopping condition
                current_metric = val_metrics.get(self.es_metric, 0.0)
                improved = self._check_improvement(current_metric)

                if improved:
                    best_metrics = {**val_metrics}
                    self.checkpoint_manager.save_best(
                        self.model, self.optimizer, epoch, val_metrics
                    )
                    logger.info(
                        f"  → New best {self.es_metric}: {current_metric:.4f}"
                    )

                if self.early_stopping and self._should_stop:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch + 1} "
                        f"(patience={self.es_patience})"
                    )
                    break

            # ── Periodic checkpoint ──────────────────────────────────────────
            ck_cfg = self.config["checkpointing"]
            if (epoch + 1) % ck_cfg["save_every_n_epochs"] == 0:
                self.checkpoint_manager.save_periodic(
                    self.model, self.optimizer, self.scheduler, epoch, val_metrics
                )

            # ── Learning rate step ───────────────────────────────────────────
            self.scheduler.step()

            # ── Epoch summary log ────────────────────────────────────────────
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            log_dict = {
                "epoch": epoch + 1,
                "lr": current_lr,
                "epoch_time_s": epoch_time,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }

            self.metric_logger.log_epoch(log_dict)
            self._print_epoch_summary(epoch, log_dict)

        logger.info("Training complete.")
        return best_metrics

    # ── Single Train Epoch ────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch over the full training DataLoader."""
        self.model.train()

        epoch_loss = 0.0
        epoch_dice = 0.0
        n_batches = 0

        self.optimizer.zero_grad()   # Reset gradients at epoch start

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1:03d}/{self.num_epochs} [Train]",
            leave=False,
        )

        for step, batch in enumerate(pbar):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            # ── Forward pass (mixed precision) ───────────────────────────────
            with autocast(enabled=self.mixed_precision):
                predictions = self.model(images)
                loss = self.criterion(predictions, masks)

                # Scale loss for gradient accumulation
                loss_scaled = loss / self.accum_steps

            # ── Backward pass ────────────────────────────────────────────────
            self.scaler.scale(loss_scaled).backward()

            # ── Gradient accumulation update ─────────────────────────────────
            if (step + 1) % self.accum_steps == 0:
                # Unscale before clipping so clip norm is in the correct scale
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # ── Metrics ──────────────────────────────────────────────────────
            with torch.no_grad():
                # Compute Dice on this batch (detached from graph)
                pred_binary = (predictions.detach() > 0.5).float()
                batch_metrics = compute_metrics(pred_binary, masks)

            epoch_loss += loss.item()
            epoch_dice += batch_metrics["dice"]
            n_batches += 1
            self.global_step += 1

            # Per-step logging
            if self.global_step % self.log_every == 0:
                self.metric_logger.log_step({
                    "train/loss": loss.item(),
                    "train/dice": batch_metrics["dice"],
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "step": self.global_step,
                })

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{batch_metrics['dice']:.3f}",
            })

        avg_loss = epoch_loss / max(1, n_batches)
        avg_dice = epoch_dice / max(1, n_batches)

        return {"loss": avg_loss, "dice": avg_dice}

    # ── Single Validation Epoch ───────────────────────────────────────────────

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one validation epoch over the full validation DataLoader."""
        self.model.eval()

        epoch_loss = 0.0
        all_dice, all_iou, all_precision, all_recall = [], [], [], []

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch+1:03d}/{self.num_epochs} [Val]  ",
            leave=False,
        )

        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with autocast(enabled=self.mixed_precision):
                predictions = self.model(images)
                loss = self.criterion(predictions, masks)

            pred_binary = (predictions > 0.5).float()
            metrics = compute_metrics(pred_binary, masks)

            epoch_loss += loss.item()
            all_dice.append(metrics["dice"])
            all_iou.append(metrics["iou"])
            all_precision.append(metrics["precision"])
            all_recall.append(metrics["recall"])

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{metrics['dice']:.3f}",
            })

        n = max(1, len(self.val_loader))
        val_metrics = {
            "loss": epoch_loss / n,
            "dice": float(torch.tensor(all_dice).mean()),
            "iou": float(torch.tensor(all_iou).mean()),
            "precision": float(torch.tensor(all_precision).mean()),
            "recall": float(torch.tensor(all_recall).mean()),
        }

        self.metric_logger.log_step({
            f"val/{k}": v for k, v in val_metrics.items()
        })

        return val_metrics

    # ── Early Stopping ────────────────────────────────────────────────────────

    def _check_improvement(self, current: float) -> bool:
        """
        Check if the current validation metric is an improvement.
        Updates early stopping counter and best metric.

        Returns True if improved, False otherwise.
        """
        if self.es_mode == "max":
            improved = current > self._best_val_metric
        else:
            improved = current < self._best_val_metric

        if improved:
            self._best_val_metric = current
            self._es_counter = 0
        else:
            self._es_counter += 1
            if self._es_counter >= self.es_patience:
                self._should_stop = True

        return improved

    # ── Logging Helper ────────────────────────────────────────────────────────

    @staticmethod
    def _print_epoch_summary(epoch: int, metrics: dict) -> None:
        """Format and print a concise epoch summary."""
        val_dice = metrics.get("val_dice", float("nan"))
        train_loss = metrics.get("train_loss", float("nan"))
        lr = metrics.get("lr", float("nan"))
        t = metrics.get("epoch_time_s", 0)

        logger.info(
            f"Epoch {epoch+1:4d} | "
            f"loss={train_loss:.4f} | "
            f"val_dice={val_dice:.4f} | "
            f"lr={lr:.2e} | "
            f"time={t:.1f}s"
        )

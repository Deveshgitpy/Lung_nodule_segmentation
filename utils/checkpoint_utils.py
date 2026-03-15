"""
utils/checkpoint_utils.py
==========================
Model checkpoint saving and loading utilities.

Checkpoint files store:
  - Model state dict (all learned weights)
  - Optimizer state dict (Adam momentum buffers — needed for resumption)
  - Scheduler state dict (LR schedule position)
  - Current epoch number
  - Best validation metrics
  - Full training config (for architecture reconstruction)
  - Timestamp and experiment name

The CheckpointManager tracks the top-k checkpoints by a chosen metric
and automatically deletes older, lower-performing checkpoints to
save disk space.

Best practice: save checkpoint every N epochs AND save a separate "best"
checkpoint that is overwritten whenever the validation metric improves.
"""

import heapq
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─── Checkpoint Manager ───────────────────────────────────────────────────────

class CheckpointManager:
    """
    Manages model checkpoints with top-k retention and best-model tracking.

    Args:
        config: Full training configuration dict.
    """

    def __init__(self, config: dict) -> None:
        ck_cfg = config.get("checkpointing", {})
        self.checkpoint_dir = Path(ck_cfg.get("checkpoint_dir", "outputs/checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_best_metric = ck_cfg.get("save_best_metric", "val_dice")
        self.save_best_mode = ck_cfg.get("save_best_mode", "max")
        self.keep_top_k = ck_cfg.get("keep_top_k", 3)
        self.experiment_name = config.get("experiment", {}).get("name", "experiment")
        self.config = config

        # Min-heap for top-k tracking: (metric_value, checkpoint_path)
        # For "max" mode: store negative values to use min-heap as max-heap
        self._checkpoint_heap: List[Tuple[float, str]] = []

        self._best_metric = -float("inf") if self.save_best_mode == "max" else float("inf")
        self._best_checkpoint_path: Optional[Path] = None

    # ── Saving ────────────────────────────────────────────────────────────────

    def save_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
    ) -> Path:
        """
        Save a new best-model checkpoint.

        Overwrites the previous best checkpoint file to avoid accumulating
        too many best-model copies.

        Args:
            model:     Trained model.
            optimizer: Optimizer (for resumption support).
            epoch:     Current epoch number.
            metrics:   Dict of validation metrics.

        Returns:
            Path to the saved checkpoint.
        """
        path = self.checkpoint_dir / f"best_{self.experiment_name}.pth"
        self._save_checkpoint(model, optimizer, None, epoch, metrics, path)
        self._best_checkpoint_path = path
        logger.info(f"Best checkpoint saved: {path}")
        return path

    def save_periodic(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        metrics: Dict[str, float],
    ) -> Path:
        """
        Save a periodic checkpoint (every N epochs).

        Maintains top-k checkpoints by deleting the worst when the
        heap exceeds k entries.

        Args:
            model:     Trained model.
            optimizer: Optimizer.
            scheduler: LR scheduler.
            epoch:     Current epoch.
            metrics:   Dict of validation metrics.

        Returns:
            Path to the saved checkpoint.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.experiment_name}_epoch{epoch:04d}_{ts}.pth"
        path = self.checkpoint_dir / filename

        self._save_checkpoint(model, optimizer, scheduler, epoch, metrics, path)

        # Manage top-k retention
        metric_val = metrics.get(self.save_best_metric, 0.0)
        heap_val = -metric_val if self.save_best_mode == "max" else metric_val

        heapq.heappush(self._checkpoint_heap, (heap_val, str(path)))

        # Remove worst checkpoint if heap exceeds top-k
        if len(self._checkpoint_heap) > self.keep_top_k:
            worst_val, worst_path = heapq.heappop(self._checkpoint_heap)
            worst_path = Path(worst_path)
            if worst_path.exists():
                worst_path.unlink()
                logger.debug(f"Removed checkpoint (top-k exceeded): {worst_path}")

        logger.info(f"Periodic checkpoint saved: {path.name}")
        return path

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        epoch: int,
        metrics: Dict[str, float],
        path: Path,
    ) -> None:
        """Internal: serialise and save checkpoint dict to disk."""
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
            "experiment_name": self.experiment_name,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Atomic write: save to .tmp then rename to avoid corruption
        tmp_path = path.with_suffix(".tmp")
        torch.save(checkpoint, tmp_path)
        shutil.move(str(tmp_path), str(path))

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        strict: bool = True,
    ) -> int:
        """
        Load a checkpoint into the model (and optionally optimizer/scheduler).

        Args:
            checkpoint_path: Path to .pth checkpoint file.
            model:           Model to load weights into.
            optimizer:       Optional optimizer to restore state.
            scheduler:       Optional scheduler to restore state.
            strict:          Whether to require exact key matching in state dict.

        Returns:
            Epoch number from the checkpoint (useful for resuming training).
        """
        device = next(model.parameters()).device
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle DataParallel wrapped models (prefix 'module.' in keys)
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=strict)

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        epoch = checkpoint.get("epoch", 0)
        metrics = checkpoint.get("metrics", {})
        ts = checkpoint.get("timestamp", "unknown")

        logger.info(
            f"Loaded checkpoint from epoch {epoch} "
            f"(saved {ts}) | metrics: {metrics}"
        )

        return epoch + 1   # Return next epoch to continue from

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Return the path of the current best checkpoint."""
        return self._best_checkpoint_path

    def list_checkpoints(self) -> List[Dict]:
        """
        List all checkpoints in the checkpoint directory with metadata.

        Returns:
            List of dicts with path, epoch, metrics, timestamp.
        """
        checkpoint_info = []
        for pth_file in sorted(self.checkpoint_dir.glob("*.pth")):
            try:
                ckpt = torch.load(pth_file, map_location="cpu")
                checkpoint_info.append({
                    "path": str(pth_file),
                    "epoch": ckpt.get("epoch", -1),
                    "metrics": ckpt.get("metrics", {}),
                    "timestamp": ckpt.get("timestamp", ""),
                    "size_mb": pth_file.stat().st_size / 1e6,
                })
            except Exception as e:
                checkpoint_info.append({
                    "path": str(pth_file),
                    "error": str(e),
                })

        return sorted(checkpoint_info, key=lambda x: x.get("epoch", -1))

    def get_best_from_dir(self, metric: str = "dice", mode: str = "max") -> Optional[Path]:
        """
        Scan the checkpoint directory and return the best checkpoint by metric.

        Useful when the in-memory heap is lost (e.g., after a crash + restart).

        Args:
            metric: Metric key to compare (e.g., 'dice', 'iou').
            mode:   'max' or 'min'.

        Returns:
            Path to the best checkpoint found.
        """
        checkpoints = self.list_checkpoints()
        valid = [c for c in checkpoints if metric in c.get("metrics", {})]

        if not valid:
            logger.warning(f"No checkpoints with metric '{metric}' found.")
            return None

        if mode == "max":
            best = max(valid, key=lambda c: c["metrics"][metric])
        else:
            best = min(valid, key=lambda c: c["metrics"][metric])

        logger.info(
            f"Best checkpoint: {Path(best['path']).name} "
            f"({metric}={best['metrics'][metric]:.4f})"
        )
        return Path(best["path"])

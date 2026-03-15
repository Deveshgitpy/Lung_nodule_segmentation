"""
utils/logging_utils.py
=======================
Logging utilities: console logging setup, WandB integration,
and a unified MetricLogger interface.

The MetricLogger acts as a single facade over both console logging
and experiment tracking backends (WandB). Using a facade means
training code never imports WandB directly — swapping to a different
tracker (MLflow, TensorBoard, etc.) only requires changing this file.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ─── Console Logging Setup ────────────────────────────────────────────────────

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_str: Optional[str] = None,
) -> None:
    """
    Configure root logger with a clean, timestamped format.

    Args:
        level:      Logging level (e.g., logging.DEBUG, logging.INFO).
        log_file:   Optional path to write logs to a file simultaneously.
        format_str: Custom format string. Uses a sensible default if None.
    """
    if format_str is None:
        format_str = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a"))

    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    # Suppress overly verbose third-party loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("SimpleITK").setLevel(logging.WARNING)


# ─── Metric Logger ────────────────────────────────────────────────────────────

class MetricLogger:
    """
    Unified interface for experiment metric logging.

    Logs to:
      1. Python's built-in logger (always)
      2. WandB (if use_wandb=True and WANDB_API_KEY is set)
      3. Local JSON file (always, for offline analysis)

    Usage:
        logger = MetricLogger(config)
        logger.log_step({"train/loss": 0.234, "train/dice": 0.812})
        logger.log_epoch({"epoch": 1, "val_dice": 0.819, "lr": 1e-4})
        logger.finish()
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.log_cfg = config.get("logging", {})
        self.use_wandb = self.log_cfg.get("use_wandb", False)
        self.experiment_name = config.get("experiment", {}).get("name", "experiment")
        self.output_dir = Path(config.get("data", {}).get("output_dir", "outputs"))

        # JSON log file for offline metric analysis
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_log_path = self.output_dir / f"metrics_{self.experiment_name}_{ts}.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._json_records = []

        # WandB initialisation
        self._wandb = None
        if self.use_wandb:
            self._init_wandb()

    def _init_wandb(self) -> None:
        """Initialise WandB run. Gracefully handles missing API key."""
        try:
            import wandb

            wandb_key = os.environ.get("WANDB_API_KEY")
            if not wandb_key:
                logger.warning(
                    "WANDB_API_KEY not set. Disabling WandB logging. "
                    "Set the environment variable or run: wandb login"
                )
                self.use_wandb = False
                return

            exp_cfg = self.config.get("experiment", {})
            log_cfg = self.config.get("logging", {})

            self._wandb_run = wandb.init(
                project=log_cfg.get("wandb_project", "lung-nodule-segmentation"),
                entity=log_cfg.get("wandb_entity"),
                name=self.experiment_name,
                tags=exp_cfg.get("tags", []),
                notes=exp_cfg.get("notes", ""),
                config=self.config,
                resume="allow",
            )
            self._wandb = wandb
            logger.info(
                f"WandB run initialised: {self._wandb_run.url}"
            )

        except ImportError:
            logger.warning("WandB not installed. Disabling WandB logging.")
            self.use_wandb = False

    def log_config(self, config: dict) -> None:
        """Log full config to WandB and save as JSON."""
        config_path = self.output_dir / f"config_{self.experiment_name}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        logger.info(f"Config saved to: {config_path}")

        if self.use_wandb and self._wandb:
            self._wandb.config.update(config, allow_val_change=True)

    def log_step(self, metrics: Dict[str, Any]) -> None:
        """
        Log per-step metrics (called every N gradient steps).

        Args:
            metrics: Dict of metric_name → scalar_value.
                     Keys should use 'section/metric' convention
                     (e.g., 'train/loss', 'val/dice').
        """
        # Console
        log_parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in metrics.items() if k != "step"]
        logger.debug("Step metrics: " + "  |  ".join(log_parts))

        # JSON
        self._json_records.append({**metrics, "_type": "step"})

        # WandB
        if self.use_wandb and self._wandb:
            self._wandb.log(metrics)

    def log_epoch(self, metrics: Dict[str, Any]) -> None:
        """
        Log per-epoch summary metrics.

        Args:
            metrics: Dict of metric_name → value.
                     Should include 'epoch' key.
        """
        # Console (already printed by trainer, just log to file/wandb here)
        self._json_records.append({**metrics, "_type": "epoch"})

        # Flush JSON log every epoch
        self._flush_json()

        # WandB
        if self.use_wandb and self._wandb:
            wandb_metrics = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
            self._wandb.log(wandb_metrics)

    def log_media(
        self,
        key: str,
        image_path: str,
        caption: str = "",
    ) -> None:
        """
        Log a visualisation image to WandB.

        Args:
            key:        WandB media key (e.g., 'val/prediction_grid').
            image_path: Path to saved PNG/JPG file.
            caption:    Image caption.
        """
        if self.use_wandb and self._wandb:
            self._wandb.log({
                key: self._wandb.Image(image_path, caption=caption)
            })

    def log_summary(self, summary: Dict[str, float]) -> None:
        """
        Log a run summary (called at end of training with best metrics).

        Args:
            summary: Dict of final best metric values.
        """
        logger.info("=" * 50)
        logger.info("Training Summary:")
        for k, v in summary.items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        logger.info("=" * 50)

        if self.use_wandb and self._wandb:
            for k, v in summary.items():
                self._wandb_run.summary[k] = v

    def finish(self) -> None:
        """Finalise logging: flush JSON, close WandB run."""
        self._flush_json()
        if self.use_wandb and self._wandb and hasattr(self, "_wandb_run"):
            self._wandb_run.finish()
            logger.info("WandB run finished.")

    def _flush_json(self) -> None:
        """Write accumulated metric records to JSON log file."""
        try:
            with open(self.json_log_path, "w") as f:
                json.dump(self._json_records, f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"Failed to flush JSON log: {e}")


# ─── Progress Formatting ──────────────────────────────────────────────────────

def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """
    Format a metrics dict as a compact string for console display.

    Args:
        metrics: Dict of metric_name → float_value.
        prefix:  Optional prefix string.

    Returns:
        Formatted string like "loss=0.1234  dice=0.8741  iou=0.7923"
    """
    parts = []
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")

    result = "  ".join(parts)
    if prefix:
        result = f"{prefix}  {result}"
    return result

"""
evaluation/metrics.py
======================
Evaluation metrics for binary 3D medical image segmentation.

This module implements the standard metrics used in the LUNA16 challenge
and medical segmentation literature:

  - Dice Coefficient (F1 score for segmentation)
  - Intersection over Union (Jaccard index)
  - Precision (Positive Predictive Value)
  - Recall / Sensitivity (True Positive Rate)
  - Specificity (True Negative Rate)

All metrics are computed per-batch and can be averaged across a test set
to produce an overall performance summary.

Notation:
  TP = True Positives  (predicted nodule, is nodule)
  FP = False Positives (predicted nodule, is background) — "ghost nodules"
  FN = False Negatives (predicted background, is nodule) — "missed nodules"
  TN = True Negatives  (predicted background, is background)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# Small epsilon to prevent division by zero in metric computations
EPS = 1e-8


# ─── Core Metric Functions ────────────────────────────────────────────────────

def dice_coefficient(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = EPS,
) -> torch.Tensor:
    """
    Compute the Sørensen-Dice coefficient.

    Dice = (2 * TP + ε) / (2 * TP + FP + FN + ε)
         = (2 * |P ∩ G| + ε) / (|P| + |G| + ε)

    Range: [0, 1]. Higher is better.
    Dice = 1.0 means perfect overlap.
    Dice = 0.0 means no overlap.

    Args:
        predictions: Probability maps or binary masks [B, 1, D, H, W].
        targets:     Binary ground-truth masks [B, 1, D, H, W].
        threshold:   Threshold for converting probabilities to binary.
        smooth:      Additive smoothing to avoid division by zero.

    Returns:
        Per-sample Dice coefficients, shape [B].
    """
    # Binarise predictions
    pred_binary = (predictions >= threshold).float()
    targets_float = targets.float()

    B = pred_binary.shape[0]
    pred_flat = pred_binary.view(B, -1)
    target_flat = targets_float.view(B, -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    return (2.0 * intersection + smooth) / (union + smooth)


def iou_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = EPS,
) -> torch.Tensor:
    """
    Compute the Intersection over Union (Jaccard Index).

    IoU = (TP + ε) / (TP + FP + FN + ε)
        = |P ∩ G| / |P ∪ G|

    Relationship to Dice: IoU = Dice / (2 - Dice)

    Range: [0, 1]. Higher is better.

    Args:
        predictions: Probability maps or binary masks [B, 1, D, H, W].
        targets:     Binary ground-truth masks [B, 1, D, H, W].
        threshold:   Binarisation threshold.
        smooth:      Smoothing constant.

    Returns:
        Per-sample IoU scores, shape [B].
    """
    pred_binary = (predictions >= threshold).float()
    targets_float = targets.float()

    B = pred_binary.shape[0]
    pred_flat = pred_binary.view(B, -1)
    target_flat = targets_float.view(B, -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

    return (intersection + smooth) / (union + smooth)


def precision_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = EPS,
) -> torch.Tensor:
    """
    Compute Precision (Positive Predictive Value).

    Precision = TP / (TP + FP)

    Measures: "Of all voxels predicted as nodule, what fraction actually
    are nodule?" High precision = few false positives.

    Returns:
        Per-sample precision scores, shape [B].
    """
    pred_binary = (predictions >= threshold).float()
    targets_float = targets.float()

    B = pred_binary.shape[0]
    pred_flat = pred_binary.view(B, -1)
    target_flat = targets_float.view(B, -1)

    tp = (pred_flat * target_flat).sum(dim=1)
    fp = (pred_flat * (1 - target_flat)).sum(dim=1)

    return (tp + smooth) / (tp + fp + smooth)


def recall_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = EPS,
) -> torch.Tensor:
    """
    Compute Recall (Sensitivity / True Positive Rate).

    Recall = TP / (TP + FN)

    Measures: "Of all actual nodule voxels, what fraction did we detect?"
    High recall = few missed nodules (clinically critical — missing a
    nodule is more dangerous than a false alarm).

    Returns:
        Per-sample recall scores, shape [B].
    """
    pred_binary = (predictions >= threshold).float()
    targets_float = targets.float()

    B = pred_binary.shape[0]
    pred_flat = pred_binary.view(B, -1)
    target_flat = targets_float.view(B, -1)

    tp = (pred_flat * target_flat).sum(dim=1)
    fn = ((1 - pred_flat) * target_flat).sum(dim=1)

    return (tp + smooth) / (tp + fn + smooth)


def specificity_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = EPS,
) -> torch.Tensor:
    """
    Compute Specificity (True Negative Rate).

    Specificity = TN / (TN + FP)

    Measures: "Of all actual background voxels, what fraction did we
    correctly classify as background?" High specificity = few false positives.

    Note: For highly imbalanced segmentation tasks (nodules ≈ 0.08% of
    volume), specificity is typically >0.999 even for poor models.

    Returns:
        Per-sample specificity scores, shape [B].
    """
    pred_binary = (predictions >= threshold).float()
    targets_float = targets.float()

    B = pred_binary.shape[0]
    pred_flat = pred_binary.view(B, -1)
    target_flat = targets_float.view(B, -1)

    tn = ((1 - pred_flat) * (1 - target_flat)).sum(dim=1)
    fp = (pred_flat * (1 - target_flat)).sum(dim=1)

    return (tn + smooth) / (tn + fp + smooth)


# ─── Combined Metrics ─────────────────────────────────────────────────────────

def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all segmentation metrics for a batch.

    Averages all per-sample metrics over the batch dimension.

    Args:
        predictions: [B, 1, D, H, W] probability maps or binary masks.
        targets:     [B, 1, D, H, W] binary ground-truth masks.
        threshold:   Binarisation threshold (default: 0.5).

    Returns:
        Dict with keys: dice, iou, precision, recall, specificity.
        Each value is a float (batch average).
    """
    with torch.no_grad():
        dice = dice_coefficient(predictions, targets, threshold).mean().item()
        iou = iou_score(predictions, targets, threshold).mean().item()
        prec = precision_score(predictions, targets, threshold).mean().item()
        rec = recall_score(predictions, targets, threshold).mean().item()
        spec = specificity_score(predictions, targets, threshold).mean().item()

    return {
        "dice": dice,
        "iou": iou,
        "precision": prec,
        "recall": rec,
        "sensitivity": rec,    # Alias (recall = sensitivity in medical literature)
        "specificity": spec,
    }


# ─── Volumetric Evaluation ────────────────────────────────────────────────────

def evaluate_volume(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate a single 3D prediction volume against its ground truth.

    This function is used during the final evaluation loop where we
    process complete CT volumes rather than patches.

    Args:
        pred_mask: [D, H, W] probability or binary prediction array.
        gt_mask:   [D, H, W] binary ground-truth mask array.
        threshold: Binarisation threshold for pred_mask.

    Returns:
        Dict of metric scores for this volume.
    """
    pred_bin = (pred_mask >= threshold).astype(np.float32)
    gt = gt_mask.astype(np.float32)

    tp = np.sum(pred_bin * gt)
    fp = np.sum(pred_bin * (1 - gt))
    fn = np.sum((1 - pred_bin) * gt)
    tn = np.sum((1 - pred_bin) * (1 - gt))

    dice = (2 * tp + EPS) / (2 * tp + fp + fn + EPS)
    iou = (tp + EPS) / (tp + fp + fn + EPS)
    precision = (tp + EPS) / (tp + fp + EPS)
    recall = (tp + EPS) / (tp + fn + EPS)
    specificity = (tn + EPS) / (tn + fp + EPS)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
    }


def aggregate_metrics(per_volume_metrics: List[Dict]) -> Dict[str, float]:
    """
    Aggregate per-volume metric dicts into mean ± std statistics.

    Args:
        per_volume_metrics: List of dicts from evaluate_volume().

    Returns:
        Dict with mean and std for each metric.
    """
    if not per_volume_metrics:
        return {}

    scalar_keys = ["dice", "iou", "precision", "recall", "sensitivity", "specificity"]
    summary = {}

    for key in scalar_keys:
        values = [m[key] for m in per_volume_metrics if key in m]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
            summary[f"{key}_median"] = float(np.median(values))

    return summary


def print_metrics_table(metrics: Dict[str, float]) -> None:
    """Pretty-print a metrics summary table."""
    print("\n" + "=" * 55)
    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10}")
    print("=" * 55)

    metric_names = ["dice", "iou", "precision", "recall", "specificity"]
    for name in metric_names:
        mean_key = f"{name}_mean"
        std_key = f"{name}_std"
        if mean_key in metrics:
            mean_val = metrics[mean_key]
            std_val = metrics.get(std_key, float("nan"))
            print(f"  {name.capitalize():<23} {mean_val:>10.4f} {std_val:>10.4f}")

    print("=" * 55)


# ─── CLI Evaluation Script ────────────────────────────────────────────────────

def run_evaluation(
    predictions_dir: str,
    ground_truth_dir: str,
    output_csv: Optional[str] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Evaluate all predictions in a directory against ground-truth masks.

    Args:
        predictions_dir: Directory containing *_pred.npy files.
        ground_truth_dir: Directory containing *_mask.npy files.
        output_csv:      Path to save per-volume metrics CSV.
        threshold:       Binarisation threshold.

    Returns:
        DataFrame with per-volume and aggregate metrics.
    """
    pred_dir = Path(predictions_dir)
    gt_dir = Path(ground_truth_dir)

    pred_files = sorted(pred_dir.glob("*_pred.npy"))
    logger.info(f"Found {len(pred_files)} prediction files.")

    per_volume_results = []

    for pred_path in pred_files:
        series_uid = pred_path.stem.replace("_pred", "")
        gt_path = gt_dir / f"{series_uid}_mask.npy"

        if not gt_path.exists():
            logger.warning(f"No ground truth for {series_uid}, skipping.")
            continue

        pred_mask = np.load(str(pred_path))
        gt_mask = np.load(str(gt_path))

        volume_metrics = evaluate_volume(pred_mask, gt_mask, threshold)
        volume_metrics["series_uid"] = series_uid
        per_volume_results.append(volume_metrics)

    if not per_volume_results:
        logger.error("No volumes evaluated.")
        return pd.DataFrame()

    df = pd.DataFrame(per_volume_results)
    summary = aggregate_metrics(per_volume_results)

    print_metrics_table(summary)

    if output_csv:
        df.to_csv(output_csv, index=False)
        logger.info(f"Per-volume metrics saved to: {output_csv}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions.")
    parser.add_argument("--predictions", required=True, help="Predictions directory.")
    parser.add_argument("--ground_truth", required=True, help="Ground truth directory.")
    parser.add_argument("--output_csv", default=None, help="Output CSV path.")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    run_evaluation(
        predictions_dir=args.predictions,
        ground_truth_dir=args.ground_truth,
        output_csv=args.output_csv,
        threshold=args.threshold,
    )

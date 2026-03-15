"""
training/loss_functions.py
===========================
Loss functions for lung nodule segmentation.

Medical segmentation suffers from severe class imbalance:
  - Lung nodule voxels: ~0.08% of total CT volume
  - Background voxels: ~99.92%

BCE loss alone would converge to predicting all-background (trivially).
Dice loss is overlap-based and intrinsically handles class imbalance
by normalising for the number of positive voxels.

We use a weighted combination:
  L_total = α * L_Dice + β * L_BCE

Where:
  L_Dice = 1 - (2|P∩G| + ε) / (|P| + |G| + ε)
  L_BCE  = -[w_pos * G * log(P) + (1-G) * log(1-P)]
  α = 0.6, β = 0.4 (configurable)
  ε = 1.0 (smoothing factor to avoid division by zero)
  w_pos = 10.0 (positive class weight in BCE)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Sørensen-Dice coefficient loss for binary segmentation.

    The Dice coefficient measures the overlap between prediction P and
    ground truth G:
        Dice(P, G) = (2 * |P ∩ G| + ε) / (|P| + |G| + ε)

    Dice loss = 1 - Dice(P, G)

    Why Dice is preferred for medical segmentation:
      - Scale-invariant: not affected by the ratio of positive to negative voxels
      - Penalises both false positives and false negatives proportionally
      - Typically converges faster on sparse annotation tasks

    Args:
        smooth: Additive smoothing in numerator and denominator.
                Prevents division by zero on empty ground-truth patches.
                Small values (1.0) have minimal effect on non-trivial cases.
        reduction: 'mean' averages over batch; 'sum' sums over batch.
    """

    def __init__(self, smooth: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            predictions: [B, C, D, H, W] sigmoid-activated probability maps,
                         values in [0, 1].
            targets:     [B, C, D, H, W] binary ground-truth masks {0, 1}.

        Returns:
            Scalar Dice loss (1 - Dice coefficient), averaged over batch.
        """
        # Flatten spatial dimensions for each sample in batch
        # Shape: [B, C, D*H*W]
        B = predictions.shape[0]
        pred_flat = predictions.view(B, -1)
        target_flat = targets.view(B, -1)

        # Dice numerator: 2 * |P ∩ G|
        intersection = (pred_flat * target_flat).sum(dim=1)

        # Dice denominator: |P| + |G|
        pred_sum = pred_flat.sum(dim=1)
        target_sum = target_flat.sum(dim=1)

        # Per-sample Dice coefficient
        dice_per_sample = (2.0 * intersection + self.smooth) / (
            pred_sum + target_sum + self.smooth
        )

        # Dice loss = 1 - Dice coefficient
        dice_loss_per_sample = 1.0 - dice_per_sample

        if self.reduction == "mean":
            return dice_loss_per_sample.mean()
        elif self.reduction == "sum":
            return dice_loss_per_sample.sum()
        else:
            return dice_loss_per_sample


class BinaryFocalLoss(nn.Module):
    """
    Focal loss for binary segmentation (Lin et al., 2017).

    Focal loss down-weights easy negatives so training focuses on
    hard examples (small nodules, nodule boundaries).

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    For medical segmentation, γ=2 and α=0.25 are common defaults.

    Args:
        alpha: Weighting factor for positive class. Default: 0.25.
        gamma: Focusing parameter. Higher = more focus on hard examples.
               Default: 2.0.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: [B, 1, D, H, W] sigmoid-activated probabilities.
            targets:     [B, 1, D, H, W] binary targets.

        Returns:
            Scalar focal loss.
        """
        # Binary cross-entropy (per-element, unreduced)
        bce = F.binary_cross_entropy(predictions, targets, reduction="none")

        # p_t: probability of the true class
        p_t = predictions * targets + (1 - predictions) * (1 - targets)

        # Alpha weighting: α for positives, (1-α) for negatives
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal weighting: (1 - p_t)^γ
        focal_weight = (1 - p_t) ** self.gamma

        focal_loss = alpha_t * focal_weight * bce

        return focal_loss.mean()


class DiceBCELoss(nn.Module):
    """
    Combined Dice + Binary Cross-Entropy loss.

    This is the primary training loss for lung nodule segmentation.

    Rationale:
      - Dice loss handles class imbalance and optimises overlap directly
      - BCE loss provides pointwise supervision and stable gradients
      - Combining both gives better convergence and generalisation

    Args:
        dice_weight: Weight for the Dice loss term (default: 0.6).
        bce_weight:  Weight for the BCE loss term (default: 0.4).
        dice_smooth: Dice smoothing constant (default: 1.0).
        pos_weight:  Positive class weight for BCE loss.
                     Set to ~10 to compensate for nodule/background imbalance.
    """

    def __init__(
        self,
        dice_weight: float = 0.6,
        bce_weight: float = 0.4,
        dice_smooth: float = 1.0,
        pos_weight: Optional[float] = 10.0,
    ) -> None:
        super().__init__()

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        self.dice_loss = DiceLoss(smooth=dice_smooth)

        # pos_weight inflates the BCE loss for positive voxels to compensate
        # for their scarcity. Setting pos_weight = N_neg / N_pos ≈ 10.
        if pos_weight is not None:
            pw = torch.tensor([pos_weight])
        else:
            pw = None

        # We use BCELoss (not BCEWithLogitsLoss) because our model outputs
        # sigmoid-activated probabilities.
        self.bce_loss = nn.BCELoss(weight=None)
        self.pos_weight = pos_weight

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined Dice + BCE loss.

        Args:
            predictions: [B, 1, D, H, W] sigmoid probabilities in [0, 1].
            targets:     [B, 1, D, H, W] binary masks in {0.0, 1.0}.

        Returns:
            Scalar combined loss.
        """
        # Ensure float32 for targets
        targets = targets.float()

        # ── Dice Loss ───────────────────────────────────────────────────────
        l_dice = self.dice_loss(predictions, targets)

        # ── BCE Loss with manual positive weighting ──────────────────────────
        # Apply positive class weighting before standard BCE
        if self.pos_weight is not None:
            # weight tensor: pos_weight where target == 1, 1.0 elsewhere
            weight = targets * (self.pos_weight - 1.0) + 1.0
            l_bce = F.binary_cross_entropy(
                predictions, targets, weight=weight, reduction="mean"
            )
        else:
            l_bce = self.bce_loss(predictions, targets)

        # ── Combined Loss ────────────────────────────────────────────────────
        total_loss = self.dice_weight * l_dice + self.bce_weight * l_bce

        return total_loss

    def get_loss_components(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> dict:
        """
        Return individual loss components for logging.

        Useful for debugging: if Dice loss is flat but BCE is decreasing,
        the model may be converging to predicting mostly background.
        """
        targets = targets.float()
        l_dice = self.dice_loss(predictions, targets)

        if self.pos_weight is not None:
            weight = targets * (self.pos_weight - 1.0) + 1.0
            l_bce = F.binary_cross_entropy(
                predictions, targets, weight=weight, reduction="mean"
            )
        else:
            l_bce = self.bce_loss(predictions, targets)

        total = self.dice_weight * l_dice + self.bce_weight * l_bce

        return {
            "loss": total.item(),
            "dice_loss": l_dice.item(),
            "bce_loss": l_bce.item(),
        }


def build_loss_from_config(config: dict) -> nn.Module:
    """
    Instantiate a loss function from the config dictionary.

    Args:
        config: Full config dict or loss sub-dict.

    Returns:
        Instantiated loss module.
    """
    loss_cfg = config.get("loss", config)
    name = loss_cfg.get("name", "DiceBCELoss")

    if name == "DiceBCELoss":
        return DiceBCELoss(
            dice_weight=loss_cfg.get("dice_weight", 0.6),
            bce_weight=loss_cfg.get("bce_weight", 0.4),
            dice_smooth=loss_cfg.get("dice_smooth", 1.0),
            pos_weight=loss_cfg.get("pos_weight", 10.0),
        )
    elif name == "DiceLoss":
        return DiceLoss(smooth=loss_cfg.get("dice_smooth", 1.0))
    elif name == "BCELoss":
        pw = loss_cfg.get("pos_weight", None)
        if pw is not None:
            pw = torch.tensor([pw])
        return nn.BCELoss(weight=pw)
    elif name == "FocalLoss":
        return BinaryFocalLoss(
            alpha=loss_cfg.get("focal_alpha", 0.25),
            gamma=loss_cfg.get("focal_gamma", 2.0),
        )
    else:
        raise ValueError(f"Unknown loss function: '{name}'")

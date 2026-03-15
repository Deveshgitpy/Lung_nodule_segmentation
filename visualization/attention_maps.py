"""
visualization/attention_maps.py
=================================
Gradient-based class activation mapping (Grad-CAM) for U-Net interpretability.

Grad-CAM (Selvaraju et al., 2017) produces a heatmap that highlights the
spatial regions of the input most influential for the model's prediction.

For a segmentation model, we compute Grad-CAM at a chosen encoder layer
and overlay it on the CT slice to answer: "Which regions of the CT patch
drove this prediction?"

Adaptation for U-Net segmentation:
  Unlike classification, segmentation has no single scalar output to
  differentiate. We use the mean of the prediction probability map as
  the target scalar:

    target = mean(sigmoid(output)) over predicted positive voxels

  This measures "how much does activation at this feature map location
  contribute to the average predicted nodule probability?"

The resulting heatmap shows which encoder features were most important,
providing radiologist-interpretable saliency maps.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization," ICCV 2017.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─── Grad-CAM Implementation ──────────────────────────────────────────────────

class GradCAM3D:
    """
    Gradient-weighted Class Activation Mapping for 3D U-Net.

    Registers forward and backward hooks on a target convolutional layer
    to capture activations and gradients during a forward/backward pass.
    The Grad-CAM heatmap is then computed as:

        cam = ReLU(sum_c(alpha_c * A_c))

    where:
        A_c     = activation map for channel c at the target layer
        alpha_c = global average of gradient w.r.t. A_c
        (sum weighted by importance, ReLU keeps only positive contributions)

    Args:
        model:       UNet3D model (eval mode recommended for hooks).
        target_layer: Layer to hook. Should be a Conv3d or the last
                      encoder block for best results.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ) -> None:
        self.model = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(
        self,
        module: nn.Module,
        input: tuple,
        output: torch.Tensor,
    ) -> None:
        """Forward hook: save feature map activations."""
        self._activations = output.detach()

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: tuple,
        grad_output: tuple,
    ) -> None:
        """Backward hook: save gradients flowing through this layer."""
        self._gradients = grad_output[0].detach()

    def compute_cam(
        self,
        input_tensor: torch.Tensor,
        target_voxels: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for the given input.

        Args:
            input_tensor:   [1, 1, D, H, W] input patch tensor.
            target_voxels:  Optional binary mask [1, 1, D, H, W] specifying
                            which output voxels to use as the gradient target.
                            If None, uses all predicted positive voxels.

        Returns:
            Normalised Grad-CAM heatmap, shape [D, H, W], values in [0, 1].
        """
        self.model.eval()
        input_tensor.requires_grad_(False)

        # Ensure hooks are ready
        self._activations = None
        self._gradients = None

        # Forward pass
        output = self.model(input_tensor)   # [1, 1, D, H, W]

        # Define target: mean probability over predicted positive or target voxels
        if target_voxels is not None:
            # Gradient w.r.t. the mean output at target locations
            target_region = output * target_voxels.float()
            scalar_target = target_region.mean()
        else:
            # Gradient w.r.t. all predicted positive voxels
            pred_binary = (output.detach() > 0.5).float()
            if pred_binary.sum() > 0:
                scalar_target = (output * pred_binary).mean()
            else:
                # If nothing predicted, gradient of mean output
                scalar_target = output.mean()

        # Backward pass to compute gradients
        self.model.zero_grad()
        scalar_target.backward()

        if self._activations is None or self._gradients is None:
            logger.warning("Grad-CAM hooks did not capture activations/gradients.")
            return np.zeros(input_tensor.shape[2:])

        # Grad-CAM computation
        # alpha_c: global average of gradients over spatial dims → [C]
        # Shapes: activations = [1, C, d, h, w], gradients = [1, C, d, h, w]
        gradients = self._gradients[0]          # [C, d, h, w]
        activations = self._activations[0]      # [C, d, h, w]

        # Global average pooling over spatial dimensions
        alpha = gradients.mean(dim=(1, 2, 3))   # [C]

        # Weighted combination of activation maps
        # alpha[:, None, None, None] broadcasts C → [C, d, h, w]
        cam = (alpha[:, None, None, None] * activations).sum(dim=0)  # [d, h, w]

        # ReLU: keep only positive contributions
        cam = F.relu(cam)

        # Upsample to input resolution [D, H, W]
        cam = cam.unsqueeze(0).unsqueeze(0)   # [1, 1, d, h, w]
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()     # [D, H, W]

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.astype(np.float32)

    def remove_hooks(self) -> None:
        """Deregister hooks to free memory."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __del__(self) -> None:
        try:
            self.remove_hooks()
        except Exception:
            pass


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_attention_maps(
    ct_patch: np.ndarray,
    cam_map: np.ndarray,
    pred_mask: Optional[np.ndarray] = None,
    gt_mask: Optional[np.ndarray] = None,
    n_slices: int = 5,
    output_path: Optional[str] = None,
    title_prefix: str = "",
) -> plt.Figure:
    """
    Plot Grad-CAM attention maps alongside the CT patch.

    For each selected slice, shows:
      - CT slice (greyscale)
      - Grad-CAM heatmap overlay
      - Binary prediction overlay

    Args:
        ct_patch:    [D, H, W] CT patch array.
        cam_map:     [D, H, W] Grad-CAM heatmap, normalised [0, 1].
        pred_mask:   [D, H, W] binary predicted mask.
        gt_mask:     [D, H, W] binary ground truth mask.
        n_slices:    Number of axial slices to display.
        output_path: Save path for the figure.
        title_prefix: Prefix for the figure title.

    Returns:
        Matplotlib Figure.
    """
    D, H, W = ct_patch.shape

    # Select slices: prefer where cam_map has high values
    if cam_map.max() > 0:
        cam_sums = cam_map.sum(axis=(1, 2))
        top_slices = np.argsort(cam_sums)[-n_slices:]
        slice_indices = sorted(top_slices.tolist())
    else:
        slice_indices = np.linspace(D // 4, 3 * D // 4, n_slices, dtype=int).tolist()

    n_cols = n_slices
    n_rows = 2 + (1 if pred_mask is not None else 0)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 3.5 * n_rows),
        dpi=120,
    )

    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    cam_cmap = plt.cm.jet   # High-contrast colormap for attention maps

    for col_idx, slice_idx in enumerate(slice_indices):
        ct_sl = ct_patch[slice_idx]
        cam_sl = cam_map[slice_idx]

        row = 0
        # Row 0: CT only
        axes[row, col_idx].imshow(ct_sl, cmap="gray", vmin=0, vmax=1)
        axes[row, col_idx].set_title(f"CT [{slice_idx}]", fontsize=8)
        axes[row, col_idx].axis("off")

        row += 1
        # Row 1: Grad-CAM overlay
        axes[row, col_idx].imshow(ct_sl, cmap="gray", vmin=0, vmax=1)
        axes[row, col_idx].imshow(
            cam_sl, cmap=cam_cmap, alpha=0.5,
            vmin=0, vmax=1, interpolation="bilinear"
        )
        axes[row, col_idx].set_title(f"Grad-CAM [{slice_idx}]", fontsize=8)
        axes[row, col_idx].axis("off")

        if pred_mask is not None:
            row += 1
            pred_sl = pred_mask[slice_idx]
            gt_sl = gt_mask[slice_idx] if gt_mask is not None else None

            # Row 2: Prediction + Grad-CAM + Contours
            axes[row, col_idx].imshow(ct_sl, cmap="gray", vmin=0, vmax=1)
            axes[row, col_idx].imshow(
                cam_sl, cmap=cam_cmap, alpha=0.4, vmin=0, vmax=1
            )
            # Prediction contour in white
            if pred_sl.sum() > 0:
                axes[row, col_idx].contour(pred_sl, levels=[0.5], colors=["white"], linewidths=1.5)
            # Ground truth contour in lime green
            if gt_sl is not None and gt_sl.sum() > 0:
                axes[row, col_idx].contour(gt_sl, levels=[0.5], colors=["lime"], linewidths=1.5)
            axes[row, col_idx].set_title(f"Pred + CAM [{slice_idx}]", fontsize=8)
            axes[row, col_idx].axis("off")

    # Colorbar for Grad-CAM
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap=cam_cmap, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1, :], fraction=0.015, pad=0.02)
    cbar.set_label("Attention Weight", fontsize=9)

    title = f"{title_prefix}Grad-CAM Attention Maps — Lung Nodule Segmentation"
    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        logger.info(f"Saved attention map: {output_path}")

    return fig


# ─── Factory: Run Grad-CAM on a Volume ────────────────────────────────────────

def generate_attention_map(
    model: nn.Module,
    ct_patch: np.ndarray,
    pred_mask: Optional[np.ndarray] = None,
    gt_mask: Optional[np.ndarray] = None,
    target_layer_name: str = "encoders.3",
    device: Optional[torch.device] = None,
    output_path: Optional[str] = None,
) -> Tuple[np.ndarray, plt.Figure]:
    """
    Generate and visualise a Grad-CAM attention map for a CT patch.

    Args:
        model:             Trained UNet3D model.
        ct_patch:          [D, H, W] preprocessed CT patch.
        pred_mask:         [D, H, W] predicted binary mask.
        gt_mask:           [D, H, W] ground truth binary mask.
        target_layer_name: Dot-path to the target layer in the model.
                           E.g., "encoders.3" for the last encoder block.
        device:            Inference device.
        output_path:       Save path for visualisation.

    Returns:
        (cam_map, figure) — Grad-CAM array and matplotlib Figure.
    """
    if device is None:
        device = next(model.parameters()).device

    # Resolve target layer by name
    target_layer = _get_layer_by_name(model, target_layer_name)
    if target_layer is None:
        logger.warning(
            f"Layer '{target_layer_name}' not found. "
            f"Falling back to first convolutional layer."
        )
        target_layer = _find_first_conv(model)

    # Set up Grad-CAM
    grad_cam = GradCAM3D(model, target_layer)

    # Prepare input tensor
    input_tensor = torch.from_numpy(
        ct_patch[np.newaxis, np.newaxis, ...].astype(np.float32)
    ).to(device)

    # Compute Grad-CAM
    cam_map = grad_cam.compute_cam(input_tensor)

    # Visualise
    fig = plot_attention_maps(
        ct_patch=ct_patch,
        cam_map=cam_map,
        pred_mask=pred_mask,
        gt_mask=gt_mask,
        output_path=output_path,
    )

    grad_cam.remove_hooks()
    return cam_map, fig


# ─── Layer Discovery Helpers ──────────────────────────────────────────────────

def _get_layer_by_name(
    model: nn.Module, name: str
) -> Optional[nn.Module]:
    """
    Retrieve a sub-module by dot-path name.

    E.g., 'encoders.3.conv.block.3' navigates through model.encoders[3].conv.block[3]
    """
    parts = name.split(".")
    current = model
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        elif part.isdigit():
            try:
                current = current[int(part)]
            except (IndexError, TypeError):
                return None
        else:
            return None
    return current


def _find_first_conv(model: nn.Module) -> nn.Module:
    """Return the first Conv3d layer in the model."""
    for module in model.modules():
        if isinstance(module, nn.Conv3d):
            return module
    return list(model.modules())[0]


def list_hookable_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    List all Conv3d layers in the model for hook selection.

    Returns:
        List of (name, module) tuples for all Conv3d layers.
    """
    hookable = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            hookable.append((name, module))
    return hookable


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models.unet import build_unet_from_config
    from utils.checkpoint_utils import CheckpointManager

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate Grad-CAM attention maps.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--scan", required=True, help="Path to CT patch .npy file.")
    parser.add_argument("--pred_mask", default=None)
    parser.add_argument("--gt_mask", default=None)
    parser.add_argument("--output_dir", default="outputs/attention/")
    parser.add_argument("--target_layer", default="encoders.3",
                        help="Dot-path to target layer for Grad-CAM.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet_from_config(config).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ct = np.load(args.scan)
    pred = np.load(args.pred_mask) if args.pred_mask else None
    gt = np.load(args.gt_mask) if args.gt_mask else None

    uid = Path(args.scan).stem
    out_dir = Path(args.output_dir)

    cam, fig = generate_attention_map(
        model=model,
        ct_patch=ct,
        pred_mask=pred,
        gt_mask=gt,
        target_layer_name=args.target_layer,
        device=device,
        output_path=str(out_dir / f"{uid}_gradcam.png"),
    )
    np.save(out_dir / f"{uid}_gradcam.npy", cam)
    logger.info(f"Attention maps saved to {out_dir}")

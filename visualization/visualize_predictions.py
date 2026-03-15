"""
visualization/visualize_predictions.py
========================================
Visualization utilities for lung nodule segmentation predictions.

Provides:
  - CT slice overlays with predicted and ground-truth masks
  - Multi-plane views (axial, coronal, sagittal)
  - Side-by-side comparison grids
  - 3D surface rendering of predicted nodules (via matplotlib)
  - Prediction confidence maps (probability heatmaps)

All functions save outputs to disk and optionally return matplotlib figures
for use in Jupyter notebooks.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

logger = logging.getLogger(__name__)


# ─── Color Constants ─────────────────────────────────────────────────────────

# Overlay colours (RGBA) for mask rendering on CT slices
PRED_MASK_COLOR = np.array([1.0, 0.2, 0.2, 0.45])   # Semi-transparent red
GT_MASK_COLOR   = np.array([0.2, 1.0, 0.2, 0.45])   # Semi-transparent green
OVERLAP_COLOR   = np.array([1.0, 1.0, 0.0, 0.55])   # Yellow (TP overlap)

# CT display window: [0, 1] normalised (already windowed during preprocessing)
CT_CMAP = "gray"


# ─── Core Slice Overlay ───────────────────────────────────────────────────────

def overlay_mask_on_slice(
    ct_slice: np.ndarray,
    pred_mask_slice: Optional[np.ndarray] = None,
    gt_mask_slice: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "",
) -> plt.Axes:
    """
    Overlay segmentation mask(s) on a single 2D CT slice.

    Colour coding:
      - Red   = prediction only (false positives)
      - Green = ground truth only (false negatives / missed nodules)
      - Yellow = both prediction and ground truth (true positives)

    Args:
        ct_slice:       [H, W] float32 CT slice, normalised to [0, 1].
        pred_mask_slice:[H, W] binary prediction mask (0 or 1).
        gt_mask_slice:  [H, W] binary ground truth mask (0 or 1).
        ax:             Matplotlib axes to draw on. Creates new if None.
        title:          Axes title string.

    Returns:
        Matplotlib Axes with the overlay rendered.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Display CT slice in greyscale
    ax.imshow(ct_slice, cmap=CT_CMAP, vmin=0.0, vmax=1.0, interpolation="bilinear")

    H, W = ct_slice.shape
    # Create RGBA overlay canvas
    overlay = np.zeros((H, W, 4), dtype=np.float32)

    if pred_mask_slice is not None and gt_mask_slice is not None:
        # True positives: both masks agree → yellow
        tp = (pred_mask_slice > 0) & (gt_mask_slice > 0)
        # False positives: only predicted → red
        fp = (pred_mask_slice > 0) & (gt_mask_slice == 0)
        # False negatives: only in GT → green
        fn = (pred_mask_slice == 0) & (gt_mask_slice > 0)

        overlay[tp] = OVERLAP_COLOR
        overlay[fp] = PRED_MASK_COLOR
        overlay[fn] = GT_MASK_COLOR

    elif pred_mask_slice is not None:
        overlay[pred_mask_slice > 0] = PRED_MASK_COLOR

    elif gt_mask_slice is not None:
        overlay[gt_mask_slice > 0] = GT_MASK_COLOR

    ax.imshow(overlay, interpolation="none")

    # Legend
    legend_handles = []
    if pred_mask_slice is not None and gt_mask_slice is not None:
        legend_handles = [
            mpatches.Patch(color=PRED_MASK_COLOR[:3], alpha=0.7, label="FP (Pred only)"),
            mpatches.Patch(color=GT_MASK_COLOR[:3], alpha=0.7, label="FN (GT only)"),
            mpatches.Patch(color=OVERLAP_COLOR[:3], alpha=0.7, label="TP (Both)"),
        ]
    elif pred_mask_slice is not None:
        legend_handles = [
            mpatches.Patch(color=PRED_MASK_COLOR[:3], alpha=0.7, label="Prediction")
        ]
    elif gt_mask_slice is not None:
        legend_handles = [
            mpatches.Patch(color=GT_MASK_COLOR[:3], alpha=0.7, label="Ground Truth")
        ]

    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", fontsize=7,
                  framealpha=0.7)

    ax.set_title(title, fontsize=9)
    ax.axis("off")
    return ax


# ─── Multi-Slice Grid ─────────────────────────────────────────────────────────

def plot_segmentation_grid(
    ct_volume: np.ndarray,
    pred_mask: Optional[np.ndarray] = None,
    gt_mask: Optional[np.ndarray] = None,
    n_slices: int = 5,
    plane: str = "axial",
    output_path: Optional[str] = None,
    series_uid: str = "",
    metrics: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """
    Plot a grid of CT slices with segmentation overlays.

    Slices are selected to cover the full range of the nodule region,
    centred on slices where the mask is non-empty (nodule slices).

    Args:
        ct_volume:   [D, H, W] normalised CT volume.
        pred_mask:   [D, H, W] predicted binary mask.
        gt_mask:     [D, H, W] ground truth binary mask.
        n_slices:    Number of slice columns to show.
        plane:       Viewing plane: 'axial', 'coronal', or 'sagittal'.
        output_path: If provided, save figure to this path.
        series_uid:  Series identifier for figure title.
        metrics:     Optional dict of metric values (dice, iou, etc.) to display.

    Returns:
        Matplotlib Figure.
    """
    # Select which slices to display: prefer slices containing nodule voxels
    reference_mask = gt_mask if gt_mask is not None else pred_mask

    if reference_mask is not None and reference_mask.sum() > 0:
        slice_indices = _select_nodule_slices(reference_mask, n_slices, plane)
    else:
        # Fall back to evenly spaced slices
        dim_size = _get_plane_dim(ct_volume, plane)
        slice_indices = np.linspace(
            dim_size // 10, 9 * dim_size // 10, n_slices, dtype=int
        ).tolist()

    # Build figure with 3 rows: CT only | CT + GT | CT + Pred
    n_rows = 3 if (pred_mask is not None and gt_mask is not None) else \
             2 if (pred_mask is not None or gt_mask is not None) else 1

    fig, axes = plt.subplots(
        n_rows, n_slices,
        figsize=(3 * n_slices, 3.2 * n_rows),
        dpi=120,
    )

    if n_slices == 1:
        axes = axes.reshape(-1, 1) if n_rows > 1 else np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for col_idx, slice_idx in enumerate(slice_indices):
        ct_sl, pred_sl, gt_sl = _get_slices(ct_volume, pred_mask, gt_mask, slice_idx, plane)

        row = 0
        # Row 0: CT only
        overlay_mask_on_slice(ct_sl, ax=axes[row, col_idx],
                              title=f"{plane.capitalize()} [{slice_idx}]")
        row += 1

        # Row 1: CT + GT
        if gt_mask is not None:
            overlay_mask_on_slice(ct_sl, gt_mask_slice=gt_sl, ax=axes[row, col_idx],
                                  title="Ground Truth")
            row += 1

        # Row 2: CT + Prediction (or CT + Pred&GT overlay)
        if pred_mask is not None:
            if gt_mask is not None:
                overlay_mask_on_slice(ct_sl, pred_mask_slice=pred_sl, gt_mask_slice=gt_sl,
                                      ax=axes[row, col_idx], title="Pred vs GT")
            else:
                overlay_mask_on_slice(ct_sl, pred_mask_slice=pred_sl,
                                      ax=axes[row, col_idx], title="Prediction")

    # Figure title
    title_parts = [f"Lung Nodule Segmentation"]
    if series_uid:
        title_parts.append(f"Series: {series_uid[:20]}...")
    if metrics:
        metric_str = "  ".join(
            f"{k.upper()}={v:.3f}" for k, v in metrics.items()
            if k in ["dice", "iou", "precision", "recall"]
        )
        title_parts.append(metric_str)

    fig.suptitle("\n".join(title_parts), fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        logger.info(f"Saved segmentation grid: {output_path}")

    return fig


# ─── Probability Heatmap ──────────────────────────────────────────────────────

def plot_probability_heatmap(
    ct_volume: np.ndarray,
    probability_map: np.ndarray,
    n_slices: int = 5,
    plane: str = "axial",
    threshold: float = 0.5,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualise raw prediction probabilities as a colourmap overlay.

    Unlike the binary mask overlay, this shows the model's uncertainty.
    High-probability voxels (bright yellow) are confident predictions.
    Low-probability voxels (dark blue/purple) are uncertain boundaries.

    Args:
        ct_volume:       [D, H, W] CT volume.
        probability_map: [D, H, W] float32 probability map in [0, 1].
        n_slices:        Number of slices to show.
        plane:           Viewing plane.
        threshold:       Decision boundary line on the colour bar.
        output_path:     Path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    # Find slices with highest predicted probability mass
    if probability_map.max() > 0.0:
        slice_indices = _select_nodule_slices(
            (probability_map > 0.1).astype(np.uint8), n_slices, plane
        )
    else:
        dim_size = _get_plane_dim(ct_volume, plane)
        slice_indices = np.linspace(
            dim_size // 10, 9 * dim_size // 10, n_slices, dtype=int
        ).tolist()

    fig, axes = plt.subplots(
        2, n_slices,
        figsize=(3.5 * n_slices, 7),
        dpi=120,
    )

    cmap = plt.cm.plasma
    norm = Normalize(vmin=0.0, vmax=1.0)

    for col_idx, slice_idx in enumerate(slice_indices):
        ct_sl, _, _ = _get_slices(ct_volume, None, None, slice_idx, plane)
        _, prob_sl, _ = _get_slices(probability_map, None, None, slice_idx, plane)

        # Row 0: CT greyscale
        axes[0, col_idx].imshow(ct_sl, cmap=CT_CMAP, vmin=0, vmax=1)
        axes[0, col_idx].set_title(f"CT [{slice_idx}]", fontsize=8)
        axes[0, col_idx].axis("off")

        # Row 1: Probability heatmap overlay
        axes[1, col_idx].imshow(ct_sl, cmap=CT_CMAP, vmin=0, vmax=1)
        prob_overlay = axes[1, col_idx].imshow(
            prob_sl, cmap=cmap, norm=norm, alpha=0.55, interpolation="bilinear"
        )
        # Threshold contour
        if prob_sl.max() >= threshold:
            axes[1, col_idx].contour(
                prob_sl, levels=[threshold], colors=["white"], linewidths=1.0
            )
        axes[1, col_idx].set_title(f"P(nodule) [{slice_idx}]", fontsize=8)
        axes[1, col_idx].axis("off")

    # Colourbar
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes[1, :],
        orientation="vertical",
        fraction=0.02,
        pad=0.02,
    )
    cbar.set_label("P(nodule)", fontsize=9)
    cbar.ax.axhline(threshold, color="white", linewidth=1.5, linestyle="--")

    fig.suptitle("Nodule Probability Maps", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        logger.info(f"Saved probability heatmap: {output_path}")

    return fig


# ─── 3D Surface Visualisation ─────────────────────────────────────────────────

def plot_3d_nodule_surface(
    pred_mask: np.ndarray,
    ct_volume: Optional[np.ndarray] = None,
    gt_mask: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render predicted nodule(s) as 3D surface meshes using marching cubes.

    Requires scikit-image for marching cubes surface extraction.

    Args:
        pred_mask:   [D, H, W] predicted binary mask.
        ct_volume:   [D, H, W] CT volume for context (optional).
        gt_mask:     [D, H, W] ground truth mask for comparison (optional).
        output_path: Save path.

    Returns:
        Matplotlib Figure with 3D surface plot.
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        logger.warning("scikit-image not found. Cannot generate 3D surface.")
        return plt.figure()

    fig = plt.figure(figsize=(12, 5))

    def render_surface(ax: plt.Axes, mask: np.ndarray, color: str, title: str) -> None:
        """Extract and render a single isosurface."""
        if mask.sum() == 0:
            ax.set_title(f"{title}\n(empty mask)")
            return

        try:
            verts, faces, _, _ = marching_cubes(
                mask.astype(float), level=0.5, spacing=(1.0, 1.0, 1.0)
            )
            mesh = Poly3DCollection(verts[faces], alpha=0.6)
            mesh.set_facecolor(color)
            mesh.set_edgecolor("none")
            ax.add_collection3d(mesh)

            ax.set_xlim(0, mask.shape[0])
            ax.set_ylim(0, mask.shape[1])
            ax.set_zlim(0, mask.shape[2])
            ax.set_xlabel("D (z)")
            ax.set_ylabel("H (y)")
            ax.set_zlabel("W (x)")
            ax.set_title(title, fontsize=10)

            # Equalise axes
            max_dim = max(mask.shape)
            ax.set_xlim(0, max_dim); ax.set_ylim(0, max_dim); ax.set_zlim(0, max_dim)

        except Exception as e:
            logger.warning(f"3D surface rendering failed: {e}")
            ax.set_title(f"{title}\n(render failed)")

    if gt_mask is not None:
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        render_surface(ax1, gt_mask, color="limegreen", title="Ground Truth")
        render_surface(ax2, pred_mask, color="tomato", title="Prediction")
    else:
        ax1 = fig.add_subplot(111, projection="3d")
        render_surface(ax1, pred_mask, color="tomato", title="Predicted Nodule")

    fig.suptitle("3D Nodule Surface Rendering", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        logger.info(f"Saved 3D surface: {output_path}")

    return fig


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _select_nodule_slices(
    mask: np.ndarray,
    n_slices: int,
    plane: str,
) -> List[int]:
    """
    Select `n_slices` indices that cover the nodule region in the given plane.

    Returns indices evenly spaced within the range of slices that contain
    at least one positive voxel.
    """
    axis = {"axial": 0, "coronal": 1, "sagittal": 2}[plane]
    # Sum along the other two axes to find which slices contain nodule voxels
    slice_sums = mask.sum(axis=tuple(i for i in range(3) if i != axis))
    positive_slices = np.where(slice_sums > 0)[0]

    if len(positive_slices) == 0:
        # No positive slices: fall back to centre region
        dim = mask.shape[axis]
        return np.linspace(dim // 4, 3 * dim // 4, n_slices, dtype=int).tolist()

    # Select evenly spaced indices within the positive slice range
    z_min, z_max = positive_slices[0], positive_slices[-1]
    # Add some context around the nodule
    padding = max(1, (z_max - z_min) // 4)
    z_min = max(0, z_min - padding)
    z_max = min(mask.shape[axis] - 1, z_max + padding)

    return np.linspace(z_min, z_max, n_slices, dtype=int).tolist()


def _get_plane_dim(volume: np.ndarray, plane: str) -> int:
    """Return the number of slices along the given plane axis."""
    axis = {"axial": 0, "coronal": 1, "sagittal": 2}[plane]
    return volume.shape[axis]


def _get_slices(
    volume: np.ndarray,
    pred_mask: Optional[np.ndarray],
    gt_mask: Optional[np.ndarray],
    slice_idx: int,
    plane: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract a 2D slice from a volume and optional masks."""
    if plane == "axial":
        ct_sl = volume[slice_idx, :, :]
        pred_sl = pred_mask[slice_idx, :, :] if pred_mask is not None else None
        gt_sl = gt_mask[slice_idx, :, :] if gt_mask is not None else None
    elif plane == "coronal":
        ct_sl = volume[:, slice_idx, :]
        pred_sl = pred_mask[:, slice_idx, :] if pred_mask is not None else None
        gt_sl = gt_mask[:, slice_idx, :] if gt_mask is not None else None
    elif plane == "sagittal":
        ct_sl = volume[:, :, slice_idx]
        pred_sl = pred_mask[:, :, slice_idx] if pred_mask is not None else None
        gt_sl = gt_mask[:, :, slice_idx] if gt_mask is not None else None
    else:
        raise ValueError(f"Unknown plane: {plane}")

    return ct_sl, pred_sl, gt_sl


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Visualise segmentation predictions.")
    parser.add_argument("--scan", required=True, help="Path to CT scan .npy file.")
    parser.add_argument("--mask", required=True, help="Path to predicted mask .npy.")
    parser.add_argument("--gt_mask", default=None, help="Path to ground truth mask .npy.")
    parser.add_argument("--output_dir", default="outputs/visualizations/")
    parser.add_argument("--n_slices", type=int, default=5)
    args = parser.parse_args()

    ct = np.load(args.scan)
    pred = np.load(args.mask)
    gt = np.load(args.gt_mask) if args.gt_mask else None

    uid = Path(args.scan).stem.replace("_image", "")
    out_dir = Path(args.output_dir)

    plot_segmentation_grid(
        ct, pred, gt,
        n_slices=args.n_slices,
        plane="axial",
        output_path=str(out_dir / f"{uid}_axial_grid.png"),
        series_uid=uid,
    )

    plot_probability_heatmap(
        ct, pred,
        n_slices=args.n_slices,
        output_path=str(out_dir / f"{uid}_prob_heatmap.png"),
    )

    plot_3d_nodule_surface(
        pred, ct, gt,
        output_path=str(out_dir / f"{uid}_3d_surface.png"),
    )

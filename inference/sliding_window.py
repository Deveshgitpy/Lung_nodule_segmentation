"""
inference/sliding_window.py
============================
Patch-based sliding window inference for volumetric CT segmentation.

During training, the model sees 64³ patches. At inference time, we need
to segment the full CT volume, which can be 400 × 512 × 512 voxels — far
too large to process in a single forward pass.

The sliding window approach:
  1. Partition the volume into overlapping 3D patches
  2. Run the model on each patch
  3. Accumulate predictions in a probability accumulation buffer
  4. Divide by an importance-weight map to produce the final probability volume

Overlap handling:
  Predictions near patch boundaries are less reliable (the model has
  less context there). We weight each patch's contribution by a Gaussian
  importance map, which downweights predictions near patch edges.
  This produces smoother outputs at patch boundaries compared to uniform
  (top-hat) weighting.

Memory management:
  For large volumes, all patches from a single volume may not fit in GPU
  memory simultaneously. Patches are processed in mini-batches.
"""

import logging
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


# ─── Importance Map ───────────────────────────────────────────────────────────

def create_gaussian_importance_map(
    patch_size: Tuple[int, int, int],
    sigma_scale: float = 0.125,
) -> np.ndarray:
    """
    Create a Gaussian importance map for a patch.

    The importance map assigns higher weight to the centre of each patch
    and lower weight to the edges. When overlapping patch predictions are
    averaged using this weighting, boundary artefacts are minimised.

    Sigma is set as sigma_scale * min(patch_dims), following the convention
    in the nnU-Net framework.

    Args:
        patch_size:  (D, H, W) patch dimensions.
        sigma_scale: Controls the width of the Gaussian. Smaller values
                     create a narrower peak (more aggressive boundary suppression).

    Returns:
        Float32 importance map, shape (D, H, W), values in (0, 1].
    """
    center = np.array(patch_size) / 2.0
    sigma = sigma_scale * min(patch_size)

    # Create coordinate grids
    z, y, x = np.meshgrid(
        np.arange(patch_size[0]) - center[0],
        np.arange(patch_size[1]) - center[1],
        np.arange(patch_size[2]) - center[2],
        indexing="ij",
    )

    # Gaussian: exp(-r² / (2σ²))
    importance = np.exp(
        -(z ** 2 + y ** 2 + x ** 2) / (2.0 * sigma ** 2)
    ).astype(np.float32)

    # Normalise to have max = 1.0
    importance /= importance.max()

    # Ensure minimum weight at boundaries (avoids division by near-zero)
    importance = np.maximum(importance, 1e-6)

    return importance


# ─── Patch Coordinate Generator ───────────────────────────────────────────────

def generate_sliding_window_coords(
    volume_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    overlap: float = 0.5,
) -> List[Tuple[int, int, int, int, int, int]]:
    """
    Generate all (z0, z1, y0, y1, x0, x1) patch coordinates for a volume.

    Patches are placed with a stride determined by the overlap fraction.
    The last patch in each dimension is always aligned to the volume boundary
    (to ensure full coverage without exceeding the volume size).

    Args:
        volume_shape: (D, H, W) of the full volume.
        patch_size:   (pD, pH, pW) of each patch.
        overlap:      Fraction of patch size used as overlap (0 to 1).
                      Higher overlap → more predictions per voxel → smoother
                      output but slower inference.

    Returns:
        List of (z0, z1, y0, y1, x0, x1) tuples defining each patch.
    """
    D, H, W = volume_shape
    pD, pH, pW = patch_size

    # Stride = patch_size * (1 - overlap)
    stride_d = max(1, int(pD * (1 - overlap)))
    stride_h = max(1, int(pH * (1 - overlap)))
    stride_w = max(1, int(pW * (1 - overlap)))

    coords = []

    d_starts = list(range(0, max(1, D - pD), stride_d))
    h_starts = list(range(0, max(1, H - pH), stride_h))
    w_starts = list(range(0, max(1, W - pW), stride_w))

    # Always include the last position aligned to the boundary
    if len(d_starts) == 0 or d_starts[-1] + pD < D:
        d_starts.append(max(0, D - pD))
    if len(h_starts) == 0 or h_starts[-1] + pH < H:
        h_starts.append(max(0, H - pH))
    if len(w_starts) == 0 or w_starts[-1] + pW < W:
        w_starts.append(max(0, W - pW))

    for d0 in d_starts:
        for h0 in h_starts:
            for w0 in w_starts:
                d1 = min(D, d0 + pD)
                h1 = min(H, h0 + pH)
                w1 = min(W, w0 + pW)
                coords.append((d0, d1, h0, h1, w0, w1))

    return coords


# ─── Mini-batch Iterator ──────────────────────────────────────────────────────

def batch_patches(
    coords: List[Tuple],
    volume: np.ndarray,
    patch_size: Tuple[int, int, int],
    batch_size: int = 4,
) -> Iterator[Tuple[torch.Tensor, List[Tuple]]]:
    """
    Yield batches of (padded) patches extracted from the volume.

    For patches near the volume boundary that would extend outside,
    we pad with reflect mode to match the training data distribution.

    Args:
        coords:     List of patch coordinates from generate_sliding_window_coords.
        volume:     [D, H, W] normalised CT volume.
        patch_size: Expected patch dimensions.
        batch_size: Number of patches per batch.

    Yields:
        (batch_tensor, batch_coords) where batch_tensor is [B, 1, pD, pH, pW].
    """
    D, H, W = volume.shape
    pD, pH, pW = patch_size

    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i : i + batch_size]
        patches = []

        for (d0, d1, h0, h1, w0, w1) in batch_coords:
            # Extract patch (may be smaller than patch_size at boundaries)
            patch = volume[d0:d1, h0:h1, w0:w1]

            # Pad if smaller than expected (boundary case)
            pad_d = pD - (d1 - d0)
            pad_h = pH - (h1 - h0)
            pad_w = pW - (w1 - w0)

            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                patch = np.pad(
                    patch,
                    ((0, pad_d), (0, pad_h), (0, pad_w)),
                    mode="reflect",
                )

            # [D, H, W] → [1, D, H, W] (channel dim)
            patches.append(patch[np.newaxis, ...])

        # Stack to [B, 1, pD, pH, pW]
        batch_tensor = torch.from_numpy(
            np.stack(patches, axis=0).astype(np.float32)
        )

        yield batch_tensor, batch_coords


# ─── Sliding Window Inference ─────────────────────────────────────────────────

@torch.no_grad()
def sliding_window_inference(
    volume: np.ndarray,
    model: nn.Module,
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    overlap: float = 0.5,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
    use_gaussian_weights: bool = True,
) -> np.ndarray:
    """
    Run sliding window inference on a full 3D CT volume.

    This is the core inference function that handles the complete
    prediction pipeline for a single CT volume.

    Algorithm:
      1. Generate all patch coordinates
      2. For each mini-batch of patches:
         a. Run model forward pass → per-patch probability map
         b. Multiply by importance weights
         c. Accumulate in prediction_sum buffer
         d. Accumulate importance weights in weight_sum buffer
      3. Final prediction = prediction_sum / weight_sum

    Args:
        volume:               [D, H, W] normalised CT volume (float32, [0,1]).
        model:                Trained UNet3D model (in eval mode).
        patch_size:           (pD, pH, pW) model input patch size.
        overlap:              Fraction of overlap between adjacent patches.
        batch_size:           Number of patches per GPU batch.
        device:               torch.device for inference.
        use_gaussian_weights: Use Gaussian importance map (True) or
                              uniform weighting (False).

    Returns:
        Probability map, shape [D, H, W], float32, values in [0, 1].
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    D, H, W = volume.shape

    # ── Accumulation buffers (float64 for numerical stability) ───────────────
    prediction_sum = np.zeros((D, H, W), dtype=np.float64)
    weight_sum = np.zeros((D, H, W), dtype=np.float64)

    # ── Importance map ───────────────────────────────────────────────────────
    if use_gaussian_weights:
        importance_map = create_gaussian_importance_map(patch_size)
    else:
        importance_map = np.ones(patch_size, dtype=np.float32)

    # ── Generate patch coordinates ───────────────────────────────────────────
    coords = generate_sliding_window_coords(volume.shape, patch_size, overlap)
    pD, pH, pW = patch_size

    logger.debug(
        f"Sliding window inference: {D}×{H}×{W} volume, "
        f"patch={pD}×{pH}×{pW}, overlap={overlap:.0%}, "
        f"{len(coords)} patches total"
    )

    # ── Run model on all patches ─────────────────────────────────────────────
    for batch_tensor, batch_coords in batch_patches(coords, volume, patch_size, batch_size):
        batch_tensor = batch_tensor.to(device)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            output = model(batch_tensor)   # [B, 1, pD, pH, pW]

        # Convert to numpy [B, pD, pH, pW]
        output_np = output.squeeze(1).cpu().numpy()

        # Accumulate each patch's predictions
        for pred_patch, (d0, d1, h0, h1, w0, w1) in zip(output_np, batch_coords):
            actual_pD = d1 - d0
            actual_pH = h1 - h0
            actual_pW = w1 - w0

            # Crop importance map and prediction to actual (non-padded) size
            imp = importance_map[:actual_pD, :actual_pH, :actual_pW]
            pred_crop = pred_patch[:actual_pD, :actual_pH, :actual_pW]

            prediction_sum[d0:d1, h0:h1, w0:w1] += pred_crop * imp
            weight_sum[d0:d1, h0:h1, w0:w1] += imp

    # ── Normalise by accumulated weights ─────────────────────────────────────
    # Avoid division by zero in regions that weren't covered (shouldn't happen)
    weight_sum = np.maximum(weight_sum, 1e-8)
    probability_map = (prediction_sum / weight_sum).astype(np.float32)

    # Clip to valid probability range (numerical safety)
    probability_map = np.clip(probability_map, 0.0, 1.0)

    return probability_map

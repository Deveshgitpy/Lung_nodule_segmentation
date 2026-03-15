"""
postprocessing/refine_masks.py
================================
Post-processing pipeline to refine raw segmentation predictions.

Raw network output often contains:
  1. Small isolated false-positive clusters (model noise)
  2. Holes within true nodule predictions (incomplete coverage)
  3. Jagged boundaries (quantisation artefacts at voxel resolution)

This module applies three sequential operations to clean up predictions:
  1. Connected component analysis (CCA): remove components below a volume threshold
  2. Morphological closing: fill small holes inside predicted regions
  3. Morphological dilation (optional): slightly expand mask borders

These operations are standard in medical image segmentation pipelines
and do not require GPU compute — they run on CPU with scipy/numpy.

Reference: nnU-Net post-processing strategy (Isensee et al., 2021)
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    generate_binary_structure,
    label as nd_label,
)

logger = logging.getLogger(__name__)


# ─── Connected Component Analysis ─────────────────────────────────────────────

def remove_small_components(
    mask: np.ndarray,
    min_size: int = 10,
    connectivity: int = 3,
) -> np.ndarray:
    """
    Remove connected components smaller than `min_size` voxels.

    This is the single most effective post-processing step for medical
    image segmentation. Small isolated clusters (1–9 voxels) are almost
    always false positives caused by noise in the background.

    The minimum size threshold should be set based on the smallest
    clinically relevant nodule (3mm diameter ≈ 14 voxels at 1mm³ spacing).
    A conservative value of 10 removes clear noise without missing small nodules.

    Args:
        mask:         Binary [D, H, W] uint8 mask.
        min_size:     Minimum component size in voxels to keep.
        connectivity: Connectivity structure (1=face, 2=edge, 3=corner).
                      3D connectivity: 1 (6-connected), 2 (18-connected),
                      3 (26-connected). Use 3 for 3D medical images.

    Returns:
        Cleaned binary mask with small components removed.
    """
    if mask.sum() == 0:
        return mask

    # Generate connectivity structure for 3D
    struct = generate_binary_structure(3, connectivity)

    # Label connected components
    labelled, n_components = nd_label(mask, structure=struct)

    if n_components == 0:
        return np.zeros_like(mask)

    # Count voxels in each component
    component_sizes = np.bincount(labelled.ravel())
    # component 0 is background; skip it
    component_sizes[0] = 0

    # Keep only components above the size threshold
    kept_components = np.where(component_sizes >= min_size)[0]

    if len(kept_components) == 0:
        logger.debug(
            f"All {n_components} components below min_size={min_size}. "
            f"Returning empty mask."
        )
        return np.zeros_like(mask)

    # Reconstruct mask from kept components only
    refined = np.zeros_like(mask)
    for comp_id in kept_components:
        refined[labelled == comp_id] = 1

    removed = n_components - len(kept_components)
    if removed > 0:
        logger.debug(
            f"CCA: removed {removed}/{n_components} small components "
            f"(min_size={min_size} voxels)."
        )

    return refined.astype(np.uint8)


# ─── Morphological Operations ─────────────────────────────────────────────────

def apply_morphological_closing(
    mask: np.ndarray,
    radius: int = 2,
) -> np.ndarray:
    """
    Apply 3D morphological closing to fill small holes.

    Closing = dilation followed by erosion.
    Effect: fills holes and gaps within the predicted nodule region
    without significantly changing the overall shape.

    Physiological rationale: nodule interiors should be uniformly
    predicted as positive. Gaps arise because network boundary
    predictions can be incomplete on the first pass.

    Args:
        mask:   Binary [D, H, W] mask.
        radius: Structuring element radius (in voxels). Larger values
                fill larger holes but may also merge adjacent structures.

    Returns:
        Mask with small internal holes filled.
    """
    if mask.sum() == 0:
        return mask

    # Create spherical structuring element
    struct = _create_sphere_struct(radius)

    # binary_closing handles boundary effects with constant padding
    closed = binary_closing(mask.astype(bool), structure=struct)

    return closed.astype(np.uint8)


def apply_morphological_dilation(
    mask: np.ndarray,
    radius: int = 1,
) -> np.ndarray:
    """
    Apply 3D morphological dilation to expand mask borders.

    Dilation slightly grows the predicted region outward.
    Useful when the model consistently under-segments nodule borders
    (common because boundary voxels have mixed tissue composition).

    Use with caution: excessive dilation can merge adjacent structures.

    Args:
        mask:   Binary [D, H, W] mask.
        radius: Dilation radius (in voxels). 1–2 is typical.

    Returns:
        Dilated binary mask.
    """
    if mask.sum() == 0:
        return mask

    struct = _create_sphere_struct(radius)
    dilated = binary_dilation(mask.astype(bool), structure=struct)

    return dilated.astype(np.uint8)


def fill_interior_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes completely enclosed within the predicted mask.

    Uses scipy.ndimage.binary_fill_holes, which identifies voxels
    surrounded on all sides by the mask and fills them.

    This is applied per-slice (2D) and also in 3D.

    Args:
        mask: Binary [D, H, W] mask.

    Returns:
        Mask with interior holes filled.
    """
    if mask.sum() == 0:
        return mask

    # 3D fill
    filled_3d = binary_fill_holes(mask.astype(bool))

    return filled_3d.astype(np.uint8)


# ─── Helper ───────────────────────────────────────────────────────────────────

def _create_sphere_struct(radius: int) -> np.ndarray:
    """
    Create a 3D spherical binary structuring element.

    A spherical element is more isotropic than a cubic one, which
    is important for 3D medical images with isotropic spacing.

    Args:
        radius: Radius of the sphere in voxels.

    Returns:
        Boolean array of shape (2r+1, 2r+1, 2r+1).
    """
    size = 2 * radius + 1
    center = radius
    z, y, x = np.ogrid[:size, :size, :size]
    dist = np.sqrt((z - center) ** 2 + (y - center) ** 2 + (x - center) ** 2)
    return (dist <= radius).astype(bool)


# ─── Combined Pipeline ────────────────────────────────────────────────────────

def refine_segmentation_mask(
    mask: np.ndarray,
    config: Optional[Dict] = None,
) -> np.ndarray:
    """
    Apply the full post-processing pipeline to a raw segmentation mask.

    Pipeline:
      1. CCA: remove small spurious components
      2. Fill holes: complete the interior of larger nodules
      3. Morphological closing: smooth internal boundaries
      4. Morphological dilation (optional): expand border slightly

    Args:
        mask:   Binary [D, H, W] uint8 array from the inference pipeline.
        config: Post-processing config dict (from training_config.yaml
                under 'postprocessing' key). If None, uses defaults.

    Returns:
        Refined binary mask.
    """
    if config is None:
        config = {}

    original_positive = int(mask.sum())

    if original_positive == 0:
        logger.debug("Mask is empty — skipping post-processing.")
        return mask

    # ── Step 1: Remove small components ──────────────────────────────────────
    min_size = config.get("min_component_size", 10)
    mask = remove_small_components(mask, min_size=min_size)

    # ── Step 2: Fill holes ────────────────────────────────────────────────────
    mask = fill_interior_holes(mask)

    # ── Step 3: Morphological closing ─────────────────────────────────────────
    if config.get("morphological_closing", True):
        closing_radius = config.get("closing_radius", 2)
        mask = apply_morphological_closing(mask, radius=closing_radius)

    # ── Step 4: Morphological dilation ────────────────────────────────────────
    if config.get("morphological_dilation", False):
        dilation_radius = config.get("dilation_radius", 1)
        mask = apply_morphological_dilation(mask, radius=dilation_radius)

    final_positive = int(mask.sum())
    delta = final_positive - original_positive
    logger.debug(
        f"Post-processing: {original_positive} → {final_positive} voxels "
        f"({'+'if delta >= 0 else ''}{delta})"
    )

    return mask


# ─── Batch Post-processing ────────────────────────────────────────────────────

def postprocess_volume(
    probability_map: np.ndarray,
    threshold: float = 0.5,
    config: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full post-processing pipeline starting from a probability map.

    Args:
        probability_map: [D, H, W] float32 probability map from inference.
        threshold:       Binarisation threshold.
        config:          Post-processing config dict.

    Returns:
        (refined_mask, raw_mask) tuple of binary [D, H, W] arrays.
    """
    # Binarise
    raw_mask = (probability_map >= threshold).astype(np.uint8)

    # Refine
    refined_mask = refine_segmentation_mask(raw_mask, config)

    return refined_mask, raw_mask

"""
datasets/luna16_dataset.py
===========================
PyTorch Dataset and DataLoader factory for the LUNA16 lung nodule
segmentation task.

This module implements:
  - LUNA16Dataset: Patch-based 3D dataset supporting:
      * Positive patch mining (centred on annotated nodules)
      * Negative patch mining (random background regions)
      * Configurable class balance ratio
      * On-the-fly data augmentation
  - DataLoader factory for train/val/test splits

Patch-based training strategy:
  Processing full CT volumes (280 × 512 × 512 voxels, ~140 MB each)
  is computationally infeasible as a single training sample. Instead
  we extract fixed-size 3D patches (e.g. 64 × 64 × 64 voxels):
    - Positive patches: centred on nodule centroids ± random jitter
    - Negative patches: random background, biased to lung regions
  This strategy also implicitly handles the extreme class imbalance
  (nodule voxels ≈ 0.08% of total CT volume).
"""

import logging
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ─── Helper: Patch Extraction ─────────────────────────────────────────────────

def extract_patch(
    volume: np.ndarray,
    centre: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """
    Extract a 3D patch from a volume, centred at `centre`.

    Handles boundary cases by padding with reflect (mirrors the volume at edges),
    which is preferred over zero-padding for CT to avoid introducing artificial
    boundaries that could confuse the model.

    Args:
        volume:     Input [D, H, W] numpy array.
        centre:     (z, y, x) voxel index of the patch centre.
        patch_size: (pD, pH, pW) patch dimensions. Must be divisible by 16
                    for the 4-level U-Net downsampling path.

    Returns:
        patch:   [pD, pH, pW] array extracted from the volume.
        offsets: (z_start, y_start, x_start) of the patch in the volume
                 (before padding), useful for assembling prediction volumes.
    """
    D, H, W = volume.shape
    pD, pH, pW = patch_size
    cz, cy, cx = centre

    # Half-patch sizes (floor for start, ceil for end to handle odd patch sizes)
    hz, hy, hx = pD // 2, pH // 2, pW // 2

    # Compute patch boundaries (may be out of bounds)
    z0, z1 = cz - hz, cz - hz + pD
    y0, y1 = cy - hy, cy - hy + pH
    x0, x1 = cx - hx, cx - hx + pW

    # Pad amounts if out of bounds
    pad_z0 = max(0, -z0); pad_z1 = max(0, z1 - D)
    pad_y0 = max(0, -y0); pad_y1 = max(0, y1 - H)
    pad_x0 = max(0, -x0); pad_x1 = max(0, x1 - W)

    # Clip to valid range for indexing
    z0c, z1c = max(0, z0), min(D, z1)
    y0c, y1c = max(0, y0), min(H, y1)
    x0c, x1c = max(0, x0), min(W, x1)

    # Extract valid region
    patch = volume[z0c:z1c, y0c:y1c, x0c:x1c]

    # Pad to target size if necessary
    if any([pad_z0, pad_z1, pad_y0, pad_y1, pad_x0, pad_x1]):
        patch = np.pad(
            patch,
            ((pad_z0, pad_z1), (pad_y0, pad_y1), (pad_x0, pad_x1)),
            mode="reflect",
        )

    return patch, (z0, y0, x0)


# ─── Dataset Class ────────────────────────────────────────────────────────────

class LUNA16Dataset(Dataset):
    """
    PyTorch Dataset for LUNA16 lung nodule segmentation.

    Sampling strategy:
      Each call to __getitem__ returns one patch. The dataset pre-computes
      a list of (image_path, mask_path, centre, label) tuples during
      __init__, balancing positive (nodule) and negative (background)
      patches according to `pos_neg_ratio`.

    Directory structure expected:
      preprocessed_dir/
        subset0/
          {series_uid}_image.npy
          {series_uid}_mask.npy
        subset1/
          ...

    Args:
        preprocessed_dir:   Root of preprocessed .npy files.
        subset_indices:     List of subset indices (0-9) to include.
        patch_size:         3D patch dimensions (D, H, W).
        pos_patches_per_scan: Number of positive patches to mine per scan.
        neg_patches_per_scan: Number of negative patches to mine per scan.
        transform:          Optional callable for data augmentation.
        max_jitter:         Max voxel jitter applied to positive patch centres.
                            Prevents the model from always seeing the nodule
                            perfectly centred, improving generalisation.
        seed:               Random seed for reproducible patch sampling.
    """

    def __init__(
        self,
        preprocessed_dir: str,
        subset_indices: List[int],
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        pos_patches_per_scan: int = 8,
        neg_patches_per_scan: int = 8,
        transform: Optional[Callable] = None,
        max_jitter: int = 8,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.preprocessed_dir = Path(preprocessed_dir)
        self.patch_size = patch_size
        self.pos_patches_per_scan = pos_patches_per_scan
        self.neg_patches_per_scan = neg_patches_per_scan
        self.transform = transform
        self.max_jitter = max_jitter

        # Seed the Python random module for reproducible patch sampling
        random.seed(seed)
        np.random.seed(seed)

        # Build the list of samples
        self.samples: List[Dict] = []
        self._build_sample_list(subset_indices)

        logger.info(
            f"LUNA16Dataset: {len(self.samples)} patches "
            f"(subsets {subset_indices})"
        )

    def _build_sample_list(self, subset_indices: List[int]) -> None:
        """
        Pre-compute the list of (scan, patch_centre, is_positive) tuples.

        For positive patches: iterate over all annotated nodule voxels and
        register each nodule centroid as a sample (with random jitter).

        For negative patches: randomly sample centres far from any nodule.
        """
        for subset_idx in subset_indices:
            subset_dir = self.preprocessed_dir / f"subset{subset_idx}"
            if not subset_dir.exists():
                logger.warning(f"Missing subset dir: {subset_dir}")
                continue

            # Find all series UIDs in this subset
            image_files = sorted(subset_dir.glob("*_image.npy"))

            for img_path in image_files:
                series_uid = img_path.stem.replace("_image", "")
                mask_path = subset_dir / f"{series_uid}_mask.npy"

                if not mask_path.exists():
                    logger.warning(f"Missing mask for {series_uid}")
                    continue

                # Load mask to find nodule locations
                # (shape [D, H, W], uint8)
                mask = np.load(str(mask_path))

                # ── Positive patches ────────────────────────────────────────
                # Find connected components (individual nodules)
                pos_centres = self._find_positive_centres(mask)

                for centre in pos_centres:
                    for _ in range(self.pos_patches_per_scan // max(1, len(pos_centres))):
                        # Add random jitter so the nodule isn't always centred
                        jittered_centre = self._jitter_centre(
                            centre, mask.shape, self.max_jitter
                        )
                        self.samples.append({
                            "image_path": str(img_path),
                            "mask_path": str(mask_path),
                            "centre": jittered_centre,
                            "is_positive": True,
                        })

                # ── Negative patches ────────────────────────────────────────
                neg_centres = self._find_negative_centres(
                    mask, self.neg_patches_per_scan
                )
                for centre in neg_centres:
                    self.samples.append({
                        "image_path": str(img_path),
                        "mask_path": str(mask_path),
                        "centre": centre,
                        "is_positive": False,
                    })

    def _find_positive_centres(
        self, mask: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """
        Find the centroid of each connected nodule component in the mask.

        Returns:
            List of (z, y, x) centroids.
        """
        from scipy.ndimage import label, center_of_mass

        labelled, n_components = label(mask)
        centres = []

        for component_idx in range(1, n_components + 1):
            com = center_of_mass(labelled == component_idx)
            centres.append(tuple(int(c) for c in com))

        # If no annotated nodules, return the centre of the volume as fallback
        if not centres:
            centres = [tuple(s // 2 for s in mask.shape)]

        return centres

    def _find_negative_centres(
        self,
        mask: np.ndarray,
        n_patches: int,
    ) -> List[Tuple[int, int, int]]:
        """
        Sample random background patch centres far from nodules.

        Uses a minimum distance constraint to avoid sampling patches that
        partially overlap with nodule regions (which would be ambiguous).
        """
        D, H, W = mask.shape
        pD, pH, pW = self.patch_size
        min_dist = max(pD, pH, pW) // 2   # Must not overlap patch with any nodule

        # Dilate the mask to create an exclusion zone around nodules
        from scipy.ndimage import binary_dilation
        exclude_zone = binary_dilation(mask > 0, iterations=min_dist // 4)

        # Build valid sampling region: inside volume, outside exclusion zone
        # Margins ensure the patch fits within the volume
        z_margin = pD // 2
        y_margin = pH // 2
        x_margin = pW // 2

        valid_z = np.arange(z_margin, D - z_margin)
        valid_y = np.arange(y_margin, H - y_margin)
        valid_x = np.arange(x_margin, W - x_margin)

        centres = []
        attempts = 0
        max_attempts = n_patches * 20

        while len(centres) < n_patches and attempts < max_attempts:
            z = int(np.random.choice(valid_z))
            y = int(np.random.choice(valid_y))
            x = int(np.random.choice(valid_x))

            if not exclude_zone[z, y, x]:
                centres.append((z, y, x))
            attempts += 1

        return centres

    @staticmethod
    def _jitter_centre(
        centre: Tuple[int, int, int],
        volume_shape: Tuple[int, int, int],
        max_jitter: int,
    ) -> Tuple[int, int, int]:
        """Apply random spatial jitter to a patch centre."""
        jitter = np.random.randint(-max_jitter, max_jitter + 1, size=3)
        new_centre = tuple(
            np.clip(c + j, 0, s - 1)
            for c, j, s in zip(centre, jitter, volume_shape)
        )
        return new_centre

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a patch and its corresponding mask patch.

        Returns a dict with:
          - "image": Float32 tensor [1, pD, pH, pW], normalised CT patch.
          - "mask":  Float32 tensor [1, pD, pH, pW], binary nodule mask.
          - "is_positive": int (1 if contains nodule, 0 otherwise).
        """
        sample = self.samples[idx]

        # Load volume and mask from disk
        # Using mmap_mode='r' for memory-mapped reads avoids loading the
        # entire volume into RAM, which is critical for large CT datasets.
        image = np.load(sample["image_path"], mmap_mode="r")
        mask = np.load(sample["mask_path"], mmap_mode="r")

        # Extract patches
        img_patch, _ = extract_patch(image, sample["centre"], self.patch_size)
        mask_patch, _ = extract_patch(mask, sample["centre"], self.patch_size)

        # Ensure correct dtypes
        img_patch = img_patch.astype(np.float32)
        mask_patch = mask_patch.astype(np.float32)

        # Add channel dimension: [D, H, W] → [1, D, H, W]
        img_patch = img_patch[np.newaxis, ...]
        mask_patch = mask_patch[np.newaxis, ...]

        # Apply augmentation if in training mode
        if self.transform is not None:
            img_patch, mask_patch = self.transform(img_patch, mask_patch)

        return {
            "image": torch.from_numpy(img_patch.copy()),
            "mask": torch.from_numpy(mask_patch.copy()),
            "is_positive": int(sample["is_positive"]),
            "series_uid": Path(sample["image_path"]).stem.replace("_image", ""),
        }


# ─── DataLoader Factory ───────────────────────────────────────────────────────

def get_dataloaders(
    config: dict,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test DataLoaders from config.

    Args:
        config:          Full config dict (from training_config.yaml).
        train_transform: Augmentation callable for training set.
        val_transform:   Augmentation callable for validation set (usually None).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data_cfg = config["data"]
    patch_cfg = config["patches"]
    train_cfg = config["training"]

    patch_size = tuple(patch_cfg["patch_size"])

    # ── Training dataset ─────────────────────────────────────────────────────
    train_dataset = LUNA16Dataset(
        preprocessed_dir=data_cfg["preprocessed_dir"],
        subset_indices=data_cfg["train_subsets"],
        patch_size=patch_size,
        pos_patches_per_scan=patch_cfg["pos_patches_per_scan"],
        neg_patches_per_scan=patch_cfg["neg_patches_per_scan"],
        transform=train_transform,
        seed=config["seed"],
    )

    # ── Validation dataset ───────────────────────────────────────────────────
    val_dataset = LUNA16Dataset(
        preprocessed_dir=data_cfg["preprocessed_dir"],
        subset_indices=data_cfg["val_subsets"],
        patch_size=patch_size,
        pos_patches_per_scan=patch_cfg["pos_patches_per_scan"],
        neg_patches_per_scan=patch_cfg["neg_patches_per_scan"],
        transform=val_transform,
        seed=config["seed"] + 1,
    )

    # ── Test dataset ─────────────────────────────────────────────────────────
    test_dataset = LUNA16Dataset(
        preprocessed_dir=data_cfg["preprocessed_dir"],
        subset_indices=data_cfg["test_subsets"],
        patch_size=patch_size,
        pos_patches_per_scan=patch_cfg["pos_patches_per_scan"],
        neg_patches_per_scan=patch_cfg["neg_patches_per_scan"],
        transform=None,    # No augmentation during test
        seed=config["seed"] + 2,
    )

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        drop_last=True,    # Avoid batch-norm issues with size-1 batches
        persistent_workers=train_cfg["num_workers"] > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        drop_last=False,
    )

    logger.info(
        f"DataLoaders ready — "
        f"Train: {len(train_dataset)} patches | "
        f"Val: {len(val_dataset)} patches | "
        f"Test: {len(test_dataset)} patches"
    )

    return train_loader, val_loader, test_loader

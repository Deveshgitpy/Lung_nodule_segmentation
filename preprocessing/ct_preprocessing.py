"""
preprocessing/ct_preprocessing.py
===================================
End-to-end CT preprocessing pipeline for LUNA16 lung nodule segmentation.

This module handles:
1. Loading MetaImage (.mhd/.raw) CT volumes using SimpleITK
2. Resampling to isotropic voxel spacing (default: 1mm³)
3. HU (Hounsfield Unit) windowing for lung tissue contrast
4. Intensity normalisation to [0, 1]
5. Generating binary nodule masks from centroid annotations
6. Saving preprocessed volumes as NumPy arrays

Radiological background:
  CT scanners measure X-ray attenuation and express it in HU:
    - Air:          -1000 HU
    - Lung tissue:  -700 to -600 HU
    - Soft tissue:  20 to 80 HU
    - Bone:         400 to 1000 HU
  Lung windowing (W=1400, L=-500) isolates pulmonary structures.
  Most nodules appear as denser regions (+30 to +100 HU) against
  the dark (-700 HU) lung parenchyma.
"""

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import label as nd_label
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────────────────────

# Lung window: isolates pulmonary tissue from surrounding structures.
# Width 1400, Level -500 → covers -1200 to +200 HU.
LUNG_HU_MIN: float = -1000.0
LUNG_HU_MAX: float = 400.0


# ─── I/O Utilities ────────────────────────────────────────────────────────────

def load_ct_volume(mhd_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a CT volume from a MetaImage (.mhd) file.

    Args:
        mhd_path: Path to the .mhd file.

    Returns:
        volume:  Float32 NumPy array, shape (D, H, W) in HU.
        spacing: Voxel spacing [z, y, x] in mm.
        origin:  Volume origin [x, y, z] in mm (for coordinate conversion).
    """
    image = sitk.ReadImage(str(mhd_path))

    # SimpleITK uses [x, y, z] convention; we convert to [z, y, x] (depth-first)
    volume = sitk.GetArrayFromImage(image).astype(np.float32)  # [D, H, W]

    # Spacing in SimpleITK is [x, y, z]; reverse to [z, y, x]
    spacing = np.array(image.GetSpacing())[::-1].astype(np.float64)

    # Origin in SimpleITK is [x, y, z] — kept as-is for coordinate lookup
    origin = np.array(image.GetOrigin())

    return volume, spacing, origin


def save_preprocessed(
    volume: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
    series_uid: str,
) -> None:
    """Save preprocessed volume and mask as compressed NumPy arrays."""
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"{series_uid}_image.npy", volume)
    np.save(output_dir / f"{series_uid}_mask.npy", mask)


# ─── Resampling ───────────────────────────────────────────────────────────────

def resample_volume(
    volume: np.ndarray,
    original_spacing: np.ndarray,
    target_spacing: np.ndarray = np.array([1.0, 1.0, 1.0]),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample a CT volume to a new isotropic voxel spacing.

    Why resample?
      LUNA16 scans vary in slice thickness (0.6 – 2.5 mm) and in-plane
      resolution (0.5 – 0.9 mm). All models assume a fixed input resolution.
      Resampling to 1mm³ isotropic spacing ensures:
        - Consistent nodule apparent sizes across scanners
        - Predictable receptive fields in convolutional layers

    Method:
      Trilinear interpolation is used for the image (smooth HU values).
      Nearest-neighbour would be used for masks (preserve binary labels).

    Args:
        volume:           Input [D, H, W] float32 array.
        original_spacing: Current voxel spacing [z, y, x] in mm.
        target_spacing:   Desired voxel spacing [z, y, x] in mm.

    Returns:
        resampled_volume: Resampled [D', H', W'] array.
        new_spacing:      Actual achieved spacing (may differ slightly from
                          target_spacing due to integer voxel count rounding).
    """
    # Compute scale factors along each axis
    resize_factor = original_spacing / target_spacing

    # New volume shape (rounded to nearest integer voxel count)
    new_shape = np.round(np.array(volume.shape) * resize_factor).astype(int)

    # Actual spacing after rounding (may differ slightly from target)
    new_spacing = original_spacing * (np.array(volume.shape) / new_shape)

    # Use SimpleITK for high-quality 3D resampling with correct HU interpolation
    sitk_image = sitk.GetImageFromArray(volume)
    sitk_image.SetSpacing(original_spacing[::-1].tolist())  # [x, y, z] for sitk

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing[::-1].tolist())
    resampler.SetSize([int(new_shape[2]), int(new_shape[1]), int(new_shape[0])])
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetDefaultPixelValue(-1024.0)   # Air value for padding

    resampled = resampler.Execute(sitk_image)
    resampled_volume = sitk.GetArrayFromImage(resampled).astype(np.float32)

    return resampled_volume, new_spacing


def resample_mask(
    mask: np.ndarray,
    original_spacing: np.ndarray,
    target_spacing: np.ndarray = np.array([1.0, 1.0, 1.0]),
) -> np.ndarray:
    """
    Resample a binary mask using nearest-neighbour interpolation.

    Using nearest-neighbour for masks is critical: linear interpolation
    would create non-binary values (0.3, 0.7, etc.) at boundaries.

    Args:
        mask:             Input binary [D, H, W] uint8 mask.
        original_spacing: Current spacing [z, y, x] in mm.
        target_spacing:   Target spacing [z, y, x] in mm.

    Returns:
        Resampled binary mask.
    """
    resize_factor = original_spacing / target_spacing
    new_shape = np.round(np.array(mask.shape) * resize_factor).astype(int)

    sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    sitk_mask.SetSpacing(original_spacing[::-1].tolist())

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing[::-1].tolist())
    resampler.SetSize([int(new_shape[2]), int(new_shape[1]), int(new_shape[0])])
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputOrigin(sitk_mask.GetOrigin())
    resampler.SetOutputDirection(sitk_mask.GetDirection())
    resampler.SetDefaultPixelValue(0)

    resampled = resampler.Execute(sitk_mask)
    return sitk.GetArrayFromImage(resampled).astype(np.uint8)


# ─── HU Windowing & Normalisation ─────────────────────────────────────────────

def apply_hu_windowing(
    volume: np.ndarray,
    hu_min: float = LUNG_HU_MIN,
    hu_max: float = LUNG_HU_MAX,
) -> np.ndarray:
    """
    Clip the HU range to the lung window and normalise to [0, 1].

    Lung windowing enhances the contrast between:
      - Lung parenchyma (dark, ~-700 HU)
      - Nodules (bright islands, ~-100 to +100 HU)
      - Vessels (similar density to nodules)

    Args:
        volume: Raw CT volume in HU, shape [D, H, W].
        hu_min: Lower HU clip value (air).
        hu_max: Upper HU clip value (soft tissue).

    Returns:
        Normalised volume with float32 values in [0, 1].
    """
    # Clip to window
    volume = np.clip(volume, hu_min, hu_max)

    # Linear normalisation to [0, 1]
    volume = (volume - hu_min) / (hu_max - hu_min)

    return volume.astype(np.float32)


# ─── Mask Generation ──────────────────────────────────────────────────────────

def world_to_voxel(
    coord_world: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    """
    Convert a world coordinate (mm) to a voxel index.

    LUNA16 annotations store nodule centres in world coordinates [x, y, z].
    We need to convert to voxel indices [z_idx, y_idx, x_idx] to place
    sphere masks in the volume array.

    Formula:
        voxel = round((world - origin) / spacing)

    Args:
        coord_world: [x, y, z] world coordinate in mm.
        origin:      [x, y, z] volume origin in mm (from SimpleITK).
        spacing:     [z, y, x] voxel spacing in mm.

    Returns:
        [z, y, x] voxel index.
    """
    # Subtract origin (both in [x,y,z] convention)
    coord_rel = coord_world - origin           # [x, y, z]

    # Convert [x, y, z] to [z, y, x] for array indexing, then divide by spacing
    voxel_zyx = coord_rel[::-1] / spacing      # [z, y, x]

    return np.round(voxel_zyx).astype(int)


def create_nodule_mask(
    volume_shape: Tuple[int, int, int],
    nodule_annotations: pd.DataFrame,
    origin: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    """
    Generate a binary segmentation mask from nodule centroid annotations.

    For each annotated nodule (centre + diameter), a sphere of the
    appropriate radius is filled with 1s in the mask array.

    Sphere-based mask generation is standard in LUNA16 literature because:
      - The exact nodule boundary is not annotated (only centroid + diameter)
      - Sphere approximation is physically reasonable for roughly spherical nodules
      - It matches the ground truth used in LUNA16 detection challenges

    Args:
        volume_shape:         (D, H, W) shape of the CT volume.
        nodule_annotations:   DataFrame with columns coordX, coordY, coordZ,
                              diameter_mm for nodules in this scan.
        origin:               [x, y, z] volume origin in mm.
        spacing:              [z, y, x] voxel spacing in mm.

    Returns:
        Binary mask array, shape (D, H, W), dtype uint8.
    """
    mask = np.zeros(volume_shape, dtype=np.uint8)
    D, H, W = volume_shape

    for _, row in nodule_annotations.iterrows():
        # World coordinate of nodule centre [x, y, z]
        centre_world = np.array([row["coordX"], row["coordY"], row["coordZ"]])
        diameter_mm = float(row["diameter_mm"])
        radius_mm = diameter_mm / 2.0

        # Convert centre to voxel index [z, y, x]
        centre_vox = world_to_voxel(centre_world, origin, spacing)
        cz, cy, cx = centre_vox

        # Compute radius in voxels along each axis
        r_z = max(1, int(np.ceil(radius_mm / spacing[0])))
        r_y = max(1, int(np.ceil(radius_mm / spacing[1])))
        r_x = max(1, int(np.ceil(radius_mm / spacing[2])))

        # Bounding box around the nodule (clipped to volume bounds)
        z_min, z_max = max(0, cz - r_z), min(D, cz + r_z + 1)
        y_min, y_max = max(0, cy - r_y), min(H, cy + r_y + 1)
        x_min, x_max = max(0, cx - r_x), min(W, cx + r_x + 1)

        # Create coordinate grids in the bounding box
        z_grid = np.arange(z_min, z_max)
        y_grid = np.arange(y_min, y_max)
        x_grid = np.arange(x_min, x_max)
        zz, yy, xx = np.meshgrid(z_grid, y_grid, x_grid, indexing="ij")

        # Ellipsoid equation: check if each voxel falls inside the sphere
        # We use normalised coordinates to handle anisotropic spacing
        inside = (
            ((zz - cz) / r_z) ** 2 +
            ((yy - cy) / r_y) ** 2 +
            ((xx - cx) / r_x) ** 2
        ) <= 1.0

        mask[z_min:z_max, y_min:y_max, x_min:x_max] |= inside.astype(np.uint8)

    return mask


# ─── Per-scan Processing Function ────────────────────────────────────────────

def process_single_scan(
    mhd_path: Path,
    annotations_df: pd.DataFrame,
    output_dir: Path,
    target_spacing: np.ndarray,
    hu_min: float = LUNG_HU_MIN,
    hu_max: float = LUNG_HU_MAX,
) -> Optional[Dict]:
    """
    Full preprocessing pipeline for a single CT scan.

    Steps:
        1. Load .mhd volume
        2. Resample to target isotropic spacing
        3. Apply HU windowing + normalisation
        4. Generate binary nodule mask from annotations
        5. Save image and mask arrays

    Args:
        mhd_path:        Path to the .mhd file.
        annotations_df:  Full LUNA16 annotations DataFrame.
        output_dir:      Directory to save preprocessed files.
        target_spacing:  Target voxel spacing [z, y, x] in mm.
        hu_min:          Lower HU clip value.
        hu_max:          Upper HU clip value.

    Returns:
        Dictionary with metadata, or None if processing failed.
    """
    series_uid = mhd_path.stem

    try:
        # ── Step 1: Load volume ──────────────────────────────────────────────
        volume, original_spacing, origin = load_ct_volume(str(mhd_path))
        original_shape = volume.shape

        # ── Step 2: Resample ────────────────────────────────────────────────
        volume, new_spacing = resample_volume(volume, original_spacing, target_spacing)

        # ── Step 3: HU windowing + normalisation ─────────────────────────────
        volume = apply_hu_windowing(volume, hu_min, hu_max)

        # ── Step 4: Generate mask ────────────────────────────────────────────
        # Filter annotations to this scan
        scan_annotations = annotations_df[
            annotations_df["seriesuid"] == series_uid
        ]

        mask = create_nodule_mask(
            volume_shape=volume.shape,
            nodule_annotations=scan_annotations,
            origin=origin,
            spacing=new_spacing,
        )

        # ── Step 5: Save ─────────────────────────────────────────────────────
        # Determine subset from parent directory name
        subset_dir = output_dir / mhd_path.parent.name
        save_preprocessed(volume, mask, subset_dir, series_uid)

        return {
            "series_uid": series_uid,
            "original_shape": original_shape,
            "resampled_shape": volume.shape,
            "original_spacing": original_spacing.tolist(),
            "new_spacing": new_spacing.tolist(),
            "n_nodules": len(scan_annotations),
            "positive_voxels": int(mask.sum()),
            "status": "ok",
        }

    except Exception as exc:
        logger.error(f"Failed to process {series_uid}: {exc}")
        return {
            "series_uid": series_uid,
            "status": "failed",
            "error": str(exc),
        }


# ─── Batch Preprocessing Entry Point ─────────────────────────────────────────

def preprocess_luna16(
    data_dir: str,
    output_dir: str,
    annotations_csv: str,
    target_spacing: List[float] = [1.0, 1.0, 1.0],
    subsets: Optional[List[int]] = None,
    n_workers: int = 4,
    hu_min: float = LUNG_HU_MIN,
    hu_max: float = LUNG_HU_MAX,
) -> pd.DataFrame:
    """
    Preprocess all LUNA16 CT scans in parallel.

    Args:
        data_dir:        Root LUNA16 directory with subset0..subset9 folders.
        output_dir:      Output directory for preprocessed .npy files.
        annotations_csv: Path to LUNA16 annotations.csv.
        target_spacing:  Target isotropic spacing [z, y, x] in mm.
        subsets:         List of subset indices to process (default: all 0-9).
        n_workers:       Number of parallel worker processes.
        hu_min:          Lower HU clip value.
        hu_max:          Upper HU clip value.

    Returns:
        Metadata DataFrame with one row per scan.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_spacing = np.array(target_spacing, dtype=np.float64)

    # Load annotations
    annotations_df = pd.read_csv(annotations_csv)
    logger.info(f"Loaded {len(annotations_df)} nodule annotations.")

    # Find all .mhd files
    if subsets is None:
        subsets = list(range(10))

    mhd_files: List[Path] = []
    for subset_idx in subsets:
        subset_dir = data_dir / f"subset{subset_idx}"
        if subset_dir.exists():
            mhd_files.extend(sorted(subset_dir.glob("*.mhd")))
        else:
            logger.warning(f"Subset directory not found: {subset_dir}")

    logger.info(f"Found {len(mhd_files)} CT scans to preprocess.")

    # Process in parallel
    metadata_records = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                process_single_scan,
                mhd_path,
                annotations_df,
                output_dir,
                target_spacing,
                hu_min,
                hu_max,
            ): mhd_path
            for mhd_path in mhd_files
        }

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Preprocessing CT scans"):
            result = future.result()
            if result is not None:
                metadata_records.append(result)

    metadata_df = pd.DataFrame(metadata_records)
    metadata_path = output_dir / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    logger.info(f"Preprocessing complete. Metadata saved to {metadata_path}")

    # Print summary
    ok_count = (metadata_df["status"] == "ok").sum()
    fail_count = (metadata_df["status"] == "failed").sum()
    logger.info(f"Processed: {ok_count} success, {fail_count} failed")

    return metadata_df


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess LUNA16 CT scans for U-Net training."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root LUNA16 directory containing subset0..subset9.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save preprocessed .npy files.",
    )
    parser.add_argument(
        "--annotations", type=str, required=True,
        help="Path to LUNA16 annotations.csv file.",
    )
    parser.add_argument(
        "--target_spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
        metavar=("Z", "Y", "X"),
        help="Target voxel spacing in mm (default: 1.0 1.0 1.0).",
    )
    parser.add_argument(
        "--subsets", type=int, nargs="+", default=None,
        help="Subset indices to process (default: all 0-9).",
    )
    parser.add_argument(
        "--n_workers", type=int, default=4,
        help="Number of parallel worker processes (default: 4).",
    )
    parser.add_argument(
        "--hu_min", type=float, default=LUNG_HU_MIN,
        help=f"Lower HU clip value (default: {LUNG_HU_MIN}).",
    )
    parser.add_argument(
        "--hu_max", type=float, default=LUNG_HU_MAX,
        help=f"Upper HU clip value (default: {LUNG_HU_MAX}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    args = parse_args()

    preprocess_luna16(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        annotations_csv=args.annotations,
        target_spacing=args.target_spacing,
        subsets=args.subsets,
        n_workers=args.n_workers,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
    )

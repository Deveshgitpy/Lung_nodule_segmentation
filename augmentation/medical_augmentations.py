"""
augmentation/medical_augmentations.py
======================================
3D medical imaging augmentations for lung nodule segmentation.

Medical imaging augmentation must be applied consistently to both the
CT image and the segmentation mask. All transforms in this module accept
and return (image, mask) pairs.

Why augmentation matters for medical images:
  - Training sets are small (hundreds of CT scans vs millions of natural images)
  - Overfitting is a major risk: the model memorises scan artefacts
  - Augmentation effectively multiplies dataset size and improves generalisation
  - Geometric augmentations must be applied identically to image and mask
  - Intensity augmentations must only be applied to the image (not the mask)

The augmentation pipeline for 3D CT:
  1. Random rotation (±15°)         → geometric
  2. Random axis flips              → geometric
  3. Elastic deformation            → geometric (mimics breathing motion)
  4. Intensity scaling              → intensity (simulates scanner variability)
  5. Intensity shift                → intensity (simulates HU offset)
  6. Gaussian noise                 → intensity (simulates acquisition noise)
  7. Gaussian blur                  → intensity (simulates partial volume effect)

All transforms expect numpy arrays with shape [C, D, H, W] (channel-first).
"""

import logging
import random
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    map_coordinates,
    rotate as scipy_rotate,
)

logger = logging.getLogger(__name__)


# ─── Base Transform ───────────────────────────────────────────────────────────

class MedicalTransform:
    """Base class for all medical imaging transforms."""

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


# ─── Geometric Transforms ─────────────────────────────────────────────────────

class RandomRotation3D(MedicalTransform):
    """
    Random rotation in 3D applied uniformly across all axes.

    Lung nodules can appear at any orientation, so rotational invariance
    is important. Rotations are applied sequentially around z, y, x axes.

    Medical rationale: CT volumes are acquired with the patient in a
    standardised position, but small variations in patient orientation
    and scanner tilt are common.

    Args:
        angle_range: Max rotation angle in degrees (±angle_range).
        prob:        Probability of applying this transform.
    """

    def __init__(
        self,
        angle_range: Tuple[float, float] = (-15.0, 15.0),
        prob: float = 0.5,
    ) -> None:
        self.angle_range = angle_range
        self.prob = prob

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.prob:
            return image, mask

        # Sample random angles for each axis
        axes_pairs = [(0, 1), (0, 2), (1, 2)]   # D-H, D-W, H-W planes
        aug_img = image.copy()
        aug_mask = mask.copy()

        for axes in axes_pairs:
            angle = random.uniform(*self.angle_range)
            if abs(angle) < 1e-3:
                continue

            # Rotate image (trilinear interpolation order=1)
            # Shape: [C, D, H, W] → rotate along spatial axes (1-indexed)
            spatial_axes = (axes[0] + 1, axes[1] + 1)
            aug_img = scipy_rotate(
                aug_img, angle, axes=spatial_axes,
                reshape=False, order=1, cval=0.0,
            )

            # Rotate mask (nearest-neighbour order=0 to preserve binary values)
            aug_mask = scipy_rotate(
                aug_mask, angle, axes=spatial_axes,
                reshape=False, order=0, cval=0.0,
            )

        return aug_img.astype(np.float32), (aug_mask > 0.5).astype(np.float32)


class RandomFlip3D(MedicalTransform):
    """
    Random flipping along specified spatial axes.

    Flipping is the cheapest augmentation and is highly effective.
    For CT scans, left-right (W axis) and anterior-posterior (H axis)
    flips are anatomically plausible. Superior-inferior (D axis) flip
    is less common in practice but included for completeness.

    Args:
        axes: Which axes to potentially flip. 0=D, 1=H, 2=W.
        prob: Per-axis flip probability.
    """

    def __init__(
        self,
        axes: Tuple[int, ...] = (0, 1, 2),
        prob: float = 0.5,
    ) -> None:
        self.axes = axes
        self.prob = prob

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        for axis in self.axes:
            if random.random() < self.prob:
                # Spatial axis in [C, D, H, W] is (axis + 1)
                image = np.flip(image, axis=axis + 1).copy()
                mask = np.flip(mask, axis=axis + 1).copy()

        return image, mask


class ElasticDeformation3D(MedicalTransform):
    """
    3D elastic deformation to simulate tissue deformability.

    Elastic deformations simulate the non-rigid motion of lung tissue
    during breathing, which is the primary source of shape variation
    between scans of the same patient. This augmentation is especially
    effective for improving boundary delineation accuracy.

    Implementation follows Simard et al. (2003) and Çiçek et al. (2016):
      1. Generate random displacement fields along each axis
      2. Smooth with Gaussian filter to create spatially coherent deformation
      3. Map input coordinates through the displacement field
      4. Apply to both image (trilinear) and mask (nearest-neighbour)

    Args:
        num_control_points: Smoothing sigma for the displacement field.
                            Higher = smoother (less aggressive) deformation.
        max_displacement:   Maximum displacement magnitude (voxels).
        prob:               Probability of applying this transform.
    """

    def __init__(
        self,
        num_control_points: int = 7,
        max_displacement: float = 7.5,
        prob: float = 0.3,
    ) -> None:
        self.sigma = num_control_points
        self.alpha = max_displacement
        self.prob = prob

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.prob:
            return image, mask

        # Work on 3D spatial dims [D, H, W] (ignore channel dim)
        shape = image.shape[1:]   # [D, H, W]

        # Random displacement fields for each spatial dimension
        dz = gaussian_filter(
            np.random.randn(*shape) * self.alpha, sigma=self.sigma
        )
        dy = gaussian_filter(
            np.random.randn(*shape) * self.alpha, sigma=self.sigma
        )
        dx = gaussian_filter(
            np.random.randn(*shape) * self.alpha, sigma=self.sigma
        )

        # Create coordinate grids
        z, y, x = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing="ij",
        )

        # Displaced coordinates
        coords = [
            (z + dz).ravel(),
            (y + dy).ravel(),
            (x + dx).ravel(),
        ]

        # Apply deformation to each channel
        deformed_img = np.zeros_like(image)
        deformed_mask = np.zeros_like(mask)

        for c in range(image.shape[0]):
            deformed_img[c] = map_coordinates(
                image[c], coords, order=1, mode="reflect"
            ).reshape(shape)

        for c in range(mask.shape[0]):
            deformed_mask[c] = (
                map_coordinates(mask[c], coords, order=0, mode="constant")
                .reshape(shape) > 0.5
            ).astype(np.float32)

        return deformed_img.astype(np.float32), deformed_mask.astype(np.float32)


# ─── Intensity Transforms ─────────────────────────────────────────────────────

class IntensityScale(MedicalTransform):
    """
    Multiply voxel intensities by a random scalar.

    Simulates scanner-to-scanner variability in HU calibration.
    Only applied to the image (never the mask).

    Args:
        scale_range: (min, max) multiplicative factor.
        prob:        Probability of applying.
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        prob: float = 0.5,
    ) -> None:
        self.scale_range = scale_range
        self.prob = prob

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.prob:
            return image, mask

        scale = random.uniform(*self.scale_range)
        image = np.clip(image * scale, 0.0, 1.0).astype(np.float32)
        return image, mask


class IntensityShift(MedicalTransform):
    """
    Add a random constant offset to voxel intensities.

    Simulates scanner bias field effects and window/level variations.

    Args:
        shift_range: (min, max) additive offset.
        prob:        Probability of applying.
    """

    def __init__(
        self,
        shift_range: Tuple[float, float] = (-0.1, 0.1),
        prob: float = 0.5,
    ) -> None:
        self.shift_range = shift_range
        self.prob = prob

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.prob:
            return image, mask

        shift = random.uniform(*self.shift_range)
        image = np.clip(image + shift, 0.0, 1.0).astype(np.float32)
        return image, mask


class GaussianNoise(MedicalTransform):
    """
    Add zero-mean Gaussian noise to the CT image.

    Models quantum noise (photon statistics) and electronic noise
    from CT detectors. Noise level scales with radiation dose —
    low-dose screening CTs (like LUNA16 scans) have higher noise.

    Args:
        std_range: (min, max) range for noise standard deviation.
        prob:      Probability of applying.
    """

    def __init__(
        self,
        std_range: Tuple[float, float] = (0.0, 0.05),
        prob: float = 0.5,
    ) -> None:
        self.std_range = std_range
        self.prob = prob

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.prob:
            return image, mask

        std = random.uniform(*self.std_range)
        noise = np.random.normal(0.0, std, size=image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0).astype(np.float32)
        return image, mask


class GaussianBlur(MedicalTransform):
    """
    Apply Gaussian blur to the CT image.

    Simulates partial volume effect (voxels at nodule boundaries are
    averages of nodule tissue and surrounding parenchyma) and
    scanner-dependent spatial resolution.

    Args:
        sigma_range: (min, max) Gaussian sigma (voxels).
        prob:        Probability of applying.
    """

    def __init__(
        self,
        sigma_range: Tuple[float, float] = (0.5, 1.0),
        prob: float = 0.3,
    ) -> None:
        self.sigma_range = sigma_range
        self.prob = prob

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.prob:
            return image, mask

        sigma = random.uniform(*self.sigma_range)
        # Apply blur to each channel independently
        blurred = np.zeros_like(image)
        for c in range(image.shape[0]):
            blurred[c] = gaussian_filter(image[c], sigma=sigma)

        return blurred.astype(np.float32), mask


# ─── Composed Pipeline ────────────────────────────────────────────────────────

class Compose:
    """
    Sequential composition of multiple medical augmentation transforms.

    Args:
        transforms: List of MedicalTransform instances to apply in order.
    """

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


# ─── Factory Functions ────────────────────────────────────────────────────────

def get_training_transforms(aug_config: dict) -> Compose:
    """
    Build the training augmentation pipeline from config.

    Args:
        aug_config: Dict from the 'augmentation' section of training_config.yaml.

    Returns:
        Compose pipeline with all enabled transforms.
    """
    if not aug_config.get("enabled", True):
        logger.info("Augmentation disabled (aug_config.enabled=False).")
        return Compose([])

    transforms = []
    aug_prob = aug_config.get("aug_prob", 0.8)

    # ── Geometric ────────────────────────────────────────────────────────────
    rot_range = aug_config.get("rotation_range", [-15, 15])
    transforms.append(
        RandomRotation3D(
            angle_range=tuple(rot_range),
            prob=aug_prob,
        )
    )

    transforms.append(
        RandomFlip3D(
            axes=tuple(aug_config.get("flip_axis", [0, 1, 2])),
            prob=aug_config.get("flip_prob", 0.5),
        )
    )

    if aug_config.get("elastic_enabled", True):
        transforms.append(
            ElasticDeformation3D(
                num_control_points=aug_config.get("elastic_num_control_points", 7),
                max_displacement=aug_config.get("elastic_max_displacement", 7.5),
                prob=aug_prob * 0.5,   # Less aggressive than rotation/flip
            )
        )

    # ── Intensity ────────────────────────────────────────────────────────────
    scale_range = aug_config.get("intensity_scale_range", [0.9, 1.1])
    transforms.append(
        IntensityScale(scale_range=tuple(scale_range), prob=aug_prob)
    )

    shift_range = aug_config.get("intensity_shift_range", [-0.1, 0.1])
    transforms.append(
        IntensityShift(shift_range=tuple(shift_range), prob=aug_prob)
    )

    if aug_config.get("gaussian_noise_enabled", True):
        noise_std = aug_config.get("gaussian_noise_std", 0.02)
        transforms.append(
            GaussianNoise(std_range=(0.0, noise_std), prob=aug_prob)
        )

    if aug_config.get("gaussian_blur_enabled", True):
        blur_sigma = aug_config.get("gaussian_blur_sigma", [0.5, 1.0])
        transforms.append(
            GaussianBlur(sigma_range=tuple(blur_sigma), prob=aug_prob * 0.5)
        )

    logger.info(f"Training augmentation pipeline: {len(transforms)} transforms")
    return Compose(transforms)


def get_val_transforms() -> Compose:
    """
    Validation/test transforms: no augmentation, just return as-is.

    Returns an identity Compose pipeline.
    """
    return Compose([])

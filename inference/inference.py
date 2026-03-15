"""
inference/inference.py
=======================
Entry point for volumetric inference on lung CT scans.

Supports:
  - Single volume inference (one .mhd or .npy file)
  - Batch inference (directory of volumes)
  - Configurable probability threshold
  - Optional post-processing (connected component analysis, morphological ops)
  - Saving predictions as .npy and/or visualisation PNGs

Usage:
  # Single volume
  python inference/inference.py \\
      --checkpoint outputs/checkpoints/best_model.pth \\
      --input /path/to/scan.mhd \\
      --output outputs/predictions/ \\
      --threshold 0.5

  # Batch inference
  python inference/inference.py \\
      --checkpoint outputs/checkpoints/best_model.pth \\
      --input_dir /path/to/scans/ \\
      --output_dir outputs/predictions/ \\
      --batch_size 4
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.sliding_window import sliding_window_inference
from models.unet import UNet3D, build_unet_from_config
from postprocessing.refine_masks import refine_segmentation_mask
from preprocessing.ct_preprocessing import apply_hu_windowing, load_ct_volume, resample_volume

logger = logging.getLogger(__name__)


# ─── Checkpoint Loading ───────────────────────────────────────────────────────

def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Optional[dict] = None,
    device: Optional[torch.device] = None,
) -> Tuple[UNet3D, dict]:
    """
    Load a trained UNet3D model from a checkpoint file.

    The checkpoint stores both the model state dict and the full config
    used for training, ensuring the architecture is reconstructed correctly.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        config:          Optional config override (uses saved config if None).
        device:          Target device.

    Returns:
        (model, config) tuple.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Use config from checkpoint if not provided
    if config is None:
        config = checkpoint.get("config", {})

    model = build_unet_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "unknown")
    val_dice = checkpoint.get("metrics", {}).get("dice", "unknown")
    logger.info(
        f"Loaded checkpoint from epoch {epoch} "
        f"(val_dice={val_dice}): {checkpoint_path}"
    )

    return model, config


# ─── Single Volume Inference ──────────────────────────────────────────────────

def run_inference_on_volume(
    volume_path: str,
    model: UNet3D,
    config: dict,
    device: torch.device,
    threshold: float = 0.5,
    apply_postprocessing: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run full inference pipeline on a single CT volume.

    Steps:
      1. Load CT volume (handles .mhd and .npy formats)
      2. Resample to model's expected spacing
      3. Apply HU windowing + normalisation
      4. Run sliding window inference
      5. Apply post-processing (optional)
      6. Return probability map and binary mask

    Args:
        volume_path:          Path to the CT scan.
        model:                Trained UNet3D model.
        config:               Training config dict.
        device:               Inference device.
        threshold:            Probability threshold for binary mask.
        apply_postprocessing: Whether to apply CCA + morphological ops.

    Returns:
        (probability_map, binary_mask, original_volume) — all [D, H, W].
    """
    t0 = time.time()
    volume_path = Path(volume_path)
    logger.info(f"Processing: {volume_path.name}")

    # ── Load ─────────────────────────────────────────────────────────────────
    if volume_path.suffix == ".mhd":
        volume, original_spacing, origin = load_ct_volume(str(volume_path))
        prep_cfg = config.get("preprocessing", {})
        target_spacing = np.array(prep_cfg.get("target_spacing", [1.0, 1.0, 1.0]))

        # Resample to isotropic spacing
        volume, _ = resample_volume(volume, original_spacing, target_spacing)

        # HU windowing + normalisation
        volume = apply_hu_windowing(
            volume,
            hu_min=prep_cfg.get("hu_min", -1000.0),
            hu_max=prep_cfg.get("hu_max", 400.0),
        )

    elif volume_path.suffix == ".npy":
        # Pre-preprocessed volume
        volume = np.load(str(volume_path)).astype(np.float32)

    else:
        raise ValueError(f"Unsupported file format: {volume_path.suffix}")

    original_volume = volume.copy()
    logger.debug(f"Volume shape: {volume.shape}, range: [{volume.min():.3f}, {volume.max():.3f}]")

    # ── Sliding Window Inference ──────────────────────────────────────────────
    inf_cfg = config.get("inference", {})
    patch_size = tuple(inf_cfg.get("patch_size", [64, 64, 64]))
    overlap = inf_cfg.get("overlap", 0.5)
    batch_size = inf_cfg.get("batch_size", 4)
    use_gaussian = inf_cfg.get("use_gaussian_weights", True)

    probability_map = sliding_window_inference(
        volume=volume,
        model=model,
        patch_size=patch_size,
        overlap=overlap,
        batch_size=batch_size,
        device=device,
        use_gaussian_weights=use_gaussian,
    )

    # ── Binary Mask ───────────────────────────────────────────────────────────
    binary_mask = (probability_map >= threshold).astype(np.uint8)

    # ── Post-processing ───────────────────────────────────────────────────────
    if apply_postprocessing:
        pp_cfg = config.get("postprocessing", {})
        binary_mask = refine_segmentation_mask(binary_mask, pp_cfg)

    elapsed = time.time() - t0
    n_nodule_voxels = binary_mask.sum()
    logger.info(
        f"  Inference complete in {elapsed:.1f}s | "
        f"Nodule voxels: {n_nodule_voxels:,}"
    )

    return probability_map, binary_mask, original_volume


# ─── Batch Inference ──────────────────────────────────────────────────────────

def run_batch_inference(
    input_dir: str,
    output_dir: str,
    model: UNet3D,
    config: dict,
    device: torch.device,
    threshold: float = 0.5,
    apply_postprocessing: bool = True,
    file_extensions: Tuple[str, ...] = (".mhd", ".npy"),
) -> List[Dict]:
    """
    Run inference on all CT scans in a directory.

    Args:
        input_dir:            Directory containing CT scan files.
        output_dir:           Directory to save predictions.
        model:                Trained UNet3D model.
        config:               Training config.
        device:               Inference device.
        threshold:            Binarisation threshold.
        apply_postprocessing: Apply post-processing pipeline.
        file_extensions:      Tuple of supported file extensions.

    Returns:
        List of result dicts with series_uid, n_nodule_voxels, etc.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all volume files
    volume_files = []
    for ext in file_extensions:
        volume_files.extend(sorted(input_dir.rglob(f"*{ext}")))

    # Deduplicate (.mhd and .raw are pairs; skip .raw)
    volume_files = [f for f in volume_files if f.suffix != ".raw"]
    logger.info(f"Found {len(volume_files)} volumes to process.")

    results = []
    n_success = 0
    n_failed = 0

    for vol_path in volume_files:
        series_uid = vol_path.stem.replace("_image", "")

        try:
            prob_map, binary_mask, orig_volume = run_inference_on_volume(
                volume_path=str(vol_path),
                model=model,
                config=config,
                device=device,
                threshold=threshold,
                apply_postprocessing=apply_postprocessing,
            )

            # Save outputs
            np.save(output_dir / f"{series_uid}_pred.npy", prob_map)
            np.save(output_dir / f"{series_uid}_mask.npy", binary_mask)

            results.append({
                "series_uid": series_uid,
                "status": "ok",
                "n_nodule_voxels": int(binary_mask.sum()),
                "volume_shape": str(binary_mask.shape),
            })
            n_success += 1

        except Exception as exc:
            logger.error(f"Failed to process {series_uid}: {exc}")
            results.append({
                "series_uid": series_uid,
                "status": "failed",
                "error": str(exc),
            })
            n_failed += 1

    logger.info(
        f"Batch inference complete: {n_success} success, {n_failed} failed."
    )
    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Volumetric inference for lung nodule segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint .pth file.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/training_config.yaml",
        help="Path to training config YAML.",
    )
    # Single volume mode
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to a single CT scan (.mhd or .npy).",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/predictions/",
        help="Output directory for single scan mode.",
    )
    # Batch mode
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Directory of CT scans for batch inference.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/predictions/",
        help="Output directory for batch mode.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Probability threshold for binary mask.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override patch batch size from config.",
    )
    parser.add_argument(
        "--no_postprocessing", action="store_true",
        help="Disable post-processing (CCA, morphological ops).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Inference device (e.g. cuda, cpu).",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    args = parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.batch_size is not None:
        config.setdefault("inference", {})["batch_size"] = args.batch_size

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Inference device: {device}")

    # Load model
    model, config = load_model_from_checkpoint(
        args.checkpoint, config=config, device=device
    )

    apply_pp = not args.no_postprocessing

    if args.input:
        # Single volume mode
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        series_uid = Path(args.input).stem.replace("_image", "")
        prob_map, binary_mask, _ = run_inference_on_volume(
            volume_path=args.input,
            model=model,
            config=config,
            device=device,
            threshold=args.threshold,
            apply_postprocessing=apply_pp,
        )

        np.save(output_dir / f"{series_uid}_pred.npy", prob_map)
        np.save(output_dir / f"{series_uid}_mask.npy", binary_mask)
        logger.info(f"Predictions saved to: {output_dir}")

    elif args.input_dir:
        # Batch mode
        run_batch_inference(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model=model,
            config=config,
            device=device,
            threshold=args.threshold,
            apply_postprocessing=apply_pp,
        )

    else:
        logger.error("Please provide either --input or --input_dir.")
        sys.exit(1)


if __name__ == "__main__":
    main()

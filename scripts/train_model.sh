#!/usr/bin/env bash
# =============================================================================
# scripts/train_model.sh
# Full training launch script for U-Net lung nodule segmentation
# =============================================================================
#
# Usage:
#   # Default training run
#   bash scripts/train_model.sh
#
#   # With WandB logging enabled
#   WANDB_API_KEY=your_key bash scripts/train_model.sh --use_wandb
#
#   # Resume from checkpoint
#   bash scripts/train_model.sh --resume outputs/checkpoints/best_model.pth
#
# =============================================================================

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
CONFIG="configs/training_config.yaml"
EXPERIMENT_NAME="unet_luna16_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="outputs/logs"
USE_WANDB=false
RESUME_FROM=""

# ─── Parse Arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)          CONFIG="$2"; shift 2 ;;
    --experiment_name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --use_wandb)       USE_WANDB=true; shift 1 ;;
    --resume)          RESUME_FROM="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ─── Environment Checks ───────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║      U-Net Lung Nodule Segmentation — Training           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Config:          $CONFIG"
echo "  Experiment:      $EXPERIMENT_NAME"
echo "  WandB:           $USE_WANDB"
echo "  Resume from:     ${RESUME_FROM:-none}"
echo ""

# Check Python environment
if ! python3 -c "import torch" &>/dev/null; then
    echo "ERROR: PyTorch not found. Run: bash setup.sh"
    exit 1
fi

# Check CUDA
CUDA_AVAILABLE=$(python3 -c "import torch; print(int(torch.cuda.is_available()))")
if [ "$CUDA_AVAILABLE" -eq 1 ]; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "  GPU:             $GPU_NAME"
    VRAM=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')")
    echo "  VRAM:            $VRAM"
else
    echo "  ⚠️  WARNING: No GPU detected. Training will be very slow on CPU."
    echo "  Press Ctrl+C to cancel or wait 5 seconds to continue..."
    sleep 5
fi

echo ""

# ─── Create Log Directory ─────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}.log"
echo "  Log file:        $LOG_FILE"
echo ""

# ─── Build Training Command ───────────────────────────────────────────────────
TRAIN_CMD="python3 training/train.py"
TRAIN_CMD="$TRAIN_CMD --config $CONFIG"
TRAIN_CMD="$TRAIN_CMD --experiment_name $EXPERIMENT_NAME"

if [ "$USE_WANDB" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use_wandb"
fi

if [ -n "$RESUME_FROM" ]; then
    # Inject resume path into config override
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME_FROM"
fi

# ─── Pre-training Checks ──────────────────────────────────────────────────────
echo "Running pre-training checks..."
python3 -c "
import yaml
import sys

with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)

# Check preprocessed data exists
import os
preproc_dir = cfg['data']['preprocessed_dir']
if not os.path.exists(preproc_dir):
    print(f'ERROR: Preprocessed data not found: {preproc_dir}')
    print('Run: python preprocessing/ct_preprocessing.py --help')
    sys.exit(1)

# Count available scans
import glob
n_scans = len(glob.glob(os.path.join(preproc_dir, '**', '*_image.npy'), recursive=True))
print(f'  Found {n_scans} preprocessed CT scans.')

if n_scans == 0:
    print('ERROR: No preprocessed scans found!')
    sys.exit(1)

print('  Pre-training checks passed.')
" 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "Starting training..."
echo "Command: $TRAIN_CMD"
echo "========================================"
echo ""

# ─── Launch Training ──────────────────────────────────────────────────────────
# Tee output to both stdout and log file
$TRAIN_CMD 2>&1 | tee -a "$LOG_FILE"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================"
if [ "$TRAIN_EXIT_CODE" -eq 0 ]; then
    echo "✓ Training completed successfully!"
    echo ""
    echo "  Checkpoints: outputs/checkpoints/"
    echo "  Logs:        $LOG_FILE"
    echo ""
    echo "  To run inference:"
    echo "  python inference/inference.py \\"
    echo "      --checkpoint outputs/checkpoints/best_${EXPERIMENT_NAME}.pth \\"
    echo "      --input_dir /path/to/test_scans/ \\"
    echo "      --output_dir outputs/predictions/"
else
    echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
    echo "  Check logs: $LOG_FILE"
    exit $TRAIN_EXIT_CODE
fi

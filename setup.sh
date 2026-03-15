#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-command environment setup for lung-nodule-segmentation-unet
# =============================================================================
# Usage:
#   bash setup.sh [--no-conda] [--cpu-only]
#
# Options:
#   --no-conda    Skip conda environment creation (use existing env)
#   --cpu-only    Install CPU-only PyTorch (no CUDA)
# =============================================================================

set -euo pipefail

# ─── Parse arguments ──────────────────────────────────────────────────────────
NO_CONDA=false
CPU_ONLY=false

for arg in "$@"; do
  case $arg in
    --no-conda) NO_CONDA=true ;;
    --cpu-only) CPU_ONLY=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC}  $1"; }
log_success() { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ─── Header ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Lung Nodule Segmentation U-Net — Environment Setup     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ─── Check Python version ─────────────────────────────────────────────────────
log_info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
  log_error "Python 3.9+ is required. Found: Python $PYTHON_VERSION"
fi
log_success "Python $PYTHON_VERSION detected."

# ─── Create Conda environment (optional) ──────────────────────────────────────
if [ "$NO_CONDA" = false ]; then
  if command -v conda &>/dev/null; then
    log_info "Creating conda environment 'lungnet' with Python 3.9..."
    conda create -n lungnet python=3.9 -y || log_warn "Environment may already exist."
    log_success "Conda environment 'lungnet' ready."
    log_warn "Activate with: conda activate lungnet"
    log_warn "Re-run setup.sh --no-conda after activating."
    # Source conda and activate
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate lungnet
  else
    log_warn "Conda not found. Proceeding with current Python environment."
  fi
fi

# ─── Upgrade pip ──────────────────────────────────────────────────────────────
log_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel -q
log_success "pip upgraded."

# ─── Install PyTorch ──────────────────────────────────────────────────────────
if [ "$CPU_ONLY" = true ]; then
  log_info "Installing CPU-only PyTorch..."
  pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu -q
else
  log_info "Installing PyTorch with CUDA 11.8 support..."
  pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118 -q
fi
log_success "PyTorch installed."

# ─── Install remaining requirements ───────────────────────────────────────────
log_info "Installing project dependencies from requirements.txt..."
# Skip torch/torchvision lines since we installed them above
grep -v "^torch" requirements.txt | pip install -r /dev/stdin -q
log_success "All dependencies installed."

# ─── Verify CUDA availability ──────────────────────────────────────────────────
log_info "Checking CUDA availability..."
CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" = "True" ]; then
  CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
  GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
  log_success "CUDA $CUDA_VERSION available on GPU: $GPU_NAME"
else
  log_warn "CUDA not available. Training will run on CPU (very slow for 3D U-Net)."
fi

# ─── Create output directories ────────────────────────────────────────────────
log_info "Creating output directories..."
mkdir -p outputs/checkpoints outputs/predictions outputs/visualizations outputs/attention
log_success "Output directories created."

# ─── Install the project as editable package ──────────────────────────────────
log_info "Installing project in editable mode..."
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
  pip install -e . -q
  log_success "Project installed."
else
  log_warn "No setup.py found. Skipping editable install."
fi

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                   Setup Complete! 🎉                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Next steps:"
echo "  1. Download LUNA16 data:    cat data/dataset_instructions.md"
echo "  2. Preprocess data:         python preprocessing/ct_preprocessing.py --help"
echo "  3. Start training:          bash scripts/train_model.sh"
echo "  4. Run inference:           python inference/inference.py --help"
echo ""

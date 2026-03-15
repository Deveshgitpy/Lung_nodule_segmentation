#!/usr/bin/env bash
# =============================================================================
# scripts/download_luna16.sh
# Download LUNA16 dataset subsets from Grand Challenge
# =============================================================================
#
# Usage:
#   bash scripts/download_luna16.sh \
#       --output_dir /data/LUNA16 \
#       --subsets "0 1 2 3 4 5 6 7 8 9"
#
# Note: You must register at https://luna16.grand-challenge.org/ and
#       accept the data use agreement before downloading.
#       The download links are provided after registration.
#
# After downloading, verify with:
#   find /data/LUNA16 -name "*.mhd" | wc -l   # Should print 888
# =============================================================================

set -euo pipefail

# ─── Defaults ─────────────────────────────────────────────────────────────────
OUTPUT_DIR="/data/LUNA16"
SUBSETS="0 1 2 3 4 5 6 7 8 9"

# ─── Argument Parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --subsets)    SUBSETS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ─── Setup ────────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║            LUNA16 Dataset Download Helper                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "⚠️  IMPORTANT: You must first register and accept the data use"
echo "   agreement at https://luna16.grand-challenge.org/"
echo ""

# ─── Download annotations (always needed) ─────────────────────────────────────
echo "[1/2] Downloading annotation files..."
echo ""
echo "Please download the following files manually from the LUNA16 portal:"
echo "  - annotations.csv"
echo "  - candidates_V2.csv"
echo ""
echo "Save them to: $OUTPUT_DIR/"
echo ""

# ─── Download subsets ─────────────────────────────────────────────────────────
echo "[2/2] Subset download instructions:"
echo ""
echo "Download the following zip files and extract to $OUTPUT_DIR/:"
echo ""

for subset in $SUBSETS; do
  echo "  subset${subset}.zip  →  $OUTPUT_DIR/subset${subset}/"
done

echo ""
echo "After downloading all files, run the verification script:"
echo ""
echo "  # Verify file count (should be 888)"
echo "  find $OUTPUT_DIR -name '*.mhd' | wc -l"
echo ""
echo "  # Run preprocessing"
echo "  python preprocessing/ct_preprocessing.py \\"
echo "      --data_dir $OUTPUT_DIR \\"
echo "      --output_dir $OUTPUT_DIR/preprocessed \\"
echo "      --annotations $OUTPUT_DIR/annotations.csv \\"
echo "      --n_workers 8"
echo ""

# ─── If files already exist, verify them ──────────────────────────────────────
if [ -d "$OUTPUT_DIR/subset0" ]; then
    echo "Found existing data. Verifying..."
    MHD_COUNT=$(find "$OUTPUT_DIR" -name "*.mhd" | wc -l)
    echo "  .mhd files found: $MHD_COUNT"

    if [ "$MHD_COUNT" -eq 888 ]; then
        echo "  ✓ All 888 CT scans present."
    else
        echo "  ⚠ Expected 888, found $MHD_COUNT. Some files may be missing."
    fi

    if [ -f "$OUTPUT_DIR/annotations.csv" ]; then
        echo "  ✓ annotations.csv present."
        ANNOT_COUNT=$(tail -n +2 "$OUTPUT_DIR/annotations.csv" | wc -l)
        echo "  Annotation count: $ANNOT_COUNT"
    else
        echo "  ✗ annotations.csv missing!"
    fi
fi

echo ""
echo "For detailed instructions, see: data/dataset_instructions.md"
echo ""

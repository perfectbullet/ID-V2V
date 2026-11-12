#!/bin/bash
# ID-V2V preprocessing (idv2v): derive the VACE condition — foreground-on-gray pixels.
#
# Derives the control signal the idv2v model consumes:
#   SAM3 (person segmentation) -> orig_pixel (foreground-on-gray pixels).
#
# Reads:  <SAMPLE_DIR>/source.mp4
# Writes: <SAMPLE_DIR>/preprocessing/orig_pixel.mp4  (+ a SAM3 overlay for debugging)
#
# Usage: edit the variables below (or override via env), then run from anywhere:
#     bash scripts/preprocess.sh
set -e
cd "$(dirname "$0")/.."           # repo root — so relative paths and .venv resolve
source .venv/bin/activate

# ====================================================================
# EDIT THESE 3 VARIABLES BEFORE RUNNING (or override any via environment variable,
# e.g. `SAMPLE_DIR=... bash scripts/preprocess.sh`). See checkpoints/README.md.
# ====================================================================
SAMPLE_DIR="${SAMPLE_DIR:-test_samples/restylization/two_sitting_woman}"   # dir containing source.mp4
# Default assumes you ran `bash scripts/download_checkpoints.sh` (and optionally symlinked
# SAM 2.1 weights from /data/chenna/BadgeGroove-OOD-Seg/). Override if needed.
# Match checkpoint to its Hydra config:
#   sam2.1_hiera_base_plus.pt (309M) ↔ configs/sam2.1/sam2.1_hiera_b+.yaml
#   sam2.1_hq_hiera_large.pt (858M) ↔ no HQ config in pip sam2 — don't use
SAM2_CKPT="${SAM2_CKPT:-checkpoints/VACE-Annotators/sam2/sam2.1_hiera_base_plus.pt}"
# Built-in Hydra config (pip-installed sam2 ships pkg://sam2/configs/sam2.1/*.yaml).
# Pass as "configs/<subdir>/<name>.yaml" — Hydra resolves inside pkg://sam2.
SAM2_CFG="${SAM2_CFG:-configs/sam2.1/sam2.1_hiera_b+.yaml}"
GPU_ID="${GPU_ID:-0}"

# Optional
CLEANUP=true          # delete per-frame frame_*/ folders after pipeline finishes
MIN_AREA_RATIO="${MIN_AREA_RATIO:-0.005}"   # discard masks smaller than 0.5% of frame
MAX_AREA_RATIO="${MAX_AREA_RATIO:-0.95}"    # discard masks larger than 95% of frame
TOP_N="${TOP_N:-5}"                         # keep at most 5 largest masks per frame

SOURCE_VIDEO="${SAMPLE_DIR}/source.mp4"
PREPROC_DIR="${SAMPLE_DIR}/preprocessing"
mkdir -p "$PREPROC_DIR"

export CUDA_VISIBLE_DEVICES=$GPU_ID

# Step 1: SAM 2.1 automatic segmentation + Secret Panda mask cleanup
python -m idv2v.preprocess.sam2_preprocess \
    --video_path "$SOURCE_VIDEO" \
    --output_dir "$PREPROC_DIR" \
    --model_path "$SAM2_CKPT" \
    --config_path "$SAM2_CFG" \
    --min_area_ratio "$MIN_AREA_RATIO" \
    --max_area_ratio "$MAX_AREA_RATIO" \
    --top_n "$TOP_N"

# Step 2: foreground-on-gray — the single VACE condition idv2v consumes
python -m idv2v.preprocess.orig_pixel \
    --video_path "$SOURCE_VIDEO" \
    --mask_folder "$PREPROC_DIR" \
    --mask_image_file_name "sam3Mask_id_all.png" \
    --result_save_path "$PREPROC_DIR/orig_pixel.mp4"

if [ "$CLEANUP" = "true" ]; then
    rm -rf "$PREPROC_DIR"/frame_*/
fi

echo ""
echo "Done. Condition video in: $PREPROC_DIR"
echo "  orig_pixel.mp4   (the VACE condition for idv2v)"
echo "  *sam3_overlay.mp4 (debug viz)"

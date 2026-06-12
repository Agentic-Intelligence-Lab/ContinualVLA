#!/usr/bin/env bash
# ============================================================
# Download ContinualVLA datasets from HuggingFace Hub.
#
# Datasets are part of the RoboNever project:
#   https://github.com/Agentic-Intelligence-Lab/RoboNever
#
# Prerequisites:
#   - pip install huggingface_hub
#   - huggingface-cli login (for authentication)
#   - Set OPENPI_DATA_ROOT to your dataset storage directory
#     (default: /data/datasets)
# ============================================================

set -euo pipefail

OPENPI_DATA_ROOT="${OPENPI_DATA_ROOT:-/data/datasets}"

DATASETS=(
    "Ray0v0/cl-piper-single-stack-bowls:stack_bowls_20260413"
    "Ray0v0/cl-piper-single-hang-cup:hang_cup_20260413"
    "Ray0v0/cl-piper-single-fold-towel:fold_towel_20260417"
    "Ray0v0/cl-piper-single-press-button:press_button_20260414"
)

echo "Downloading ContinualVLA datasets to: ${OPENPI_DATA_ROOT}/realworld_piper/"
echo ""

for entry in "${DATASETS[@]}"; do
    IFS=':' read -r REPO_ID LOCAL_DIR <<< "$entry"
    TARGET="${OPENPI_DATA_ROOT}/realworld_piper/${LOCAL_DIR}"

    if [[ -d "$TARGET/meta" ]]; then
        echo "[SKIP] Already exists: ${TARGET}"
        continue
    fi

    echo "[DOWNLOAD] ${REPO_ID} -> ${TARGET}"
    mkdir -p "$(dirname "$TARGET")"
    huggingface-cli download "$REPO_ID" \
        --repo-type dataset \
        --local-dir "$TARGET" \
        --resume-download
    echo "[DONE] ${REPO_ID}"
    echo ""
done

echo "=========================================="
echo "All datasets downloaded to: ${OPENPI_DATA_ROOT}/realworld_piper/"
echo "=========================================="

#!/usr/bin/env bash
# ============================================================
# Download dataset recipes for ContinualVLA tasks.
#
# Prerequisites:
#   - Set OPENPI_DATA_ROOT to your dataset storage directory
#     (default: /data/datasets)
#   - For gated datasets (e.g., OpenGalaxea), set HF_TOKEN or
#     run `huggingface-cli login` first.
# ============================================================

set -euo pipefail

OPENPI_DATA_ROOT="${OPENPI_DATA_ROOT:-/data/datasets}"

# ============================================================
# Example: LIBERO dataset download
# ============================================================
# huggingface-cli download physical-intelligence/libero \
#   --repo-type dataset \
#   --local-dir "${OPENPI_DATA_ROOT}/physical-intelligence/libero" \
#   --resume-download \
#   --max-workers 1

# ============================================================
# Galaxea Open World Dataset: download selected task files
# ============================================================
# Repo: OpenGalaxea/Galaxea-Open-World-Dataset
# The dataset is gated — you must request access on the HF dataset page.
#
# To download a specific task:
#   huggingface-cli download OpenGalaxea/Galaxea-Open-World-Dataset \
#     --repo-type dataset \
#     --local-dir "${OPENPI_DATA_ROOT}/galaxea_open_world_datasets" \
#     --include "lerobot/<task_file>.tar.gz"

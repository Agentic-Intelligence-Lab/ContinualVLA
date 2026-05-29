#!/usr/bin/env bash
set -euo pipefail

# Download Pi0.5 base model weights from HuggingFace.
# Set OPENPI_CACHE_DIR to control where model files are stored.
# Default: ~/.cache/openpi

CACHE_DIR="${OPENPI_CACHE_DIR:-$HOME/.cache/openpi}"

huggingface-cli download lerobot/pi05_base \
  --repo-type model \
  --local-dir "${CACHE_DIR}/pi05_base" \
  --resume-download \
  --max-workers 1

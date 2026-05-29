#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------
# R1Lite offline evaluation launcher
# ----------------------------------------
# Usage examples:
#   bash scripts/run_eval_r1lite.sh
#   EPISODE_IDS="65" bash scripts/run_eval_r1lite.sh
#   EPISODE_IDS="0,6,65" MAX_STEPS=1200 bash scripts/run_eval_r1lite.sh
# ----------------------------------------

DATASET_ROOT="${DATASET_ROOT:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

# If EPISODE_IDS is empty -> use EPISODES (first N episodes).
EPISODES="${EPISODES:-1}"
EPISODE_IDS="${EPISODE_IDS:-}"

MAX_STEPS="${MAX_STEPS:-600}"
PROMPT="${PROMPT:-Fold the green T-shirt.}"

# Output directory for evaluation results
OUT_DIR="${OUT_DIR:-examples/r1lite/eval_out}"

# Video backend: pyav (recommended) or opencv
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

cmd=(
  uv run examples/r1lite/main.py
  --dataset-root "$DATASET_ROOT"
  --host "$HOST"
  --port "$PORT"
  --prompt "$PROMPT"
  --max-steps "$MAX_STEPS"
  --out-dir "$OUT_DIR"
  --video-backend "$VIDEO_BACKEND"
)

if [[ -n "$EPISODE_IDS" ]]; then
  cmd+=( --episode-ids "$EPISODE_IDS" )
else
  cmd+=( --episodes "$EPISODES" )
fi

echo "[RUN] ${cmd[*]}"
"${cmd[@]}"


#!/usr/bin/env bash
set -euo pipefail

# Single-task training for fold_towel (resume 4000 -> 6000 steps)

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-openpi-cl}"

export OPENPI_NORM_STATS_DIR="./assets/pi05_piper_stack_bowls_20260413_4cam/piper_stack_bowls_20260413_4cam"
export OPENPI_NUM_TRAIN_STEPS=4000

bash scripts/train_pytorch.sh \
  --config-name pi05_piper_fold_towel_20260417_4cam_hold_dim7_13 \
  --exp-name fold_towel_with_stack_bowls_norm_stats_0510_2105 \
  --gpus 0,1,2,3,4,5,6,7 \
  --nproc 8 \
  --nnodes 1 \
  --standalone 1 \
  --resume 0 \
  --mode file

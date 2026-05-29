#!/usr/bin/env bash
set -euo pipefail

# Single-task training for find_cube

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-openpi-cl}"

export OPENPI_NUM_TRAIN_STEPS=6000

bash scripts/train_pytorch.sh \
  --config-name pi05_piper_find_cube_100_4cam_hold_dim7_13 \
  --exp-name find_cube_0527 \
  --gpus 0,1,2,3,4,5,6,7 \
  --nproc 8 \
  --nnodes 1 \
  --standalone 1 \
  --resume 0 \
  --mode file

#!/usr/bin/env bash
set -euo pipefail

# Single-task training for press_button (4000 steps)

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-openpi-cl}"

export OPENPI_NUM_TRAIN_STEPS=4000

bash scripts/train_pytorch.sh \
  --config-name pi05_piper_press_button_20260414_4cam_hold_dim7_13 \
  --exp-name press_button_$(date +%m%d_%H%M) \
  --gpus 0,1,2,3,4,5,6,7 \
  --nproc 8 \
  --nnodes 1 \
  --standalone 1 \
  --resume 0 \
  --mode file

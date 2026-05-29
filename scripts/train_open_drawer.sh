#!/usr/bin/env bash
set -euo pipefail

# Single-task training for open_drawer (resume 4000 -> 6000 steps)

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-openpi-cl}"

export OPENPI_NUM_TRAIN_STEPS=8000

bash scripts/train_pytorch.sh \
  --config-name pi05_piper_open_the_drawer_500_4cam_hold_dim7_13 \
  --exp-name open_drawer_0502_0617 \
  --gpus 0,1,2,3 \
  --nproc 4 \
  --nnodes 1 \
  --standalone 1 \
  --resume 1 \
  --mode file

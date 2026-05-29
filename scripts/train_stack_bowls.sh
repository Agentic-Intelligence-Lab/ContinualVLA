#!/usr/bin/env bash
set -euo pipefail
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export OPENPI_CONFIG_NAME="${OPENPI_CONFIG_NAME:-RC_Stack_Bowls}"
export OPENPI_EXP_NAME="${OPENPI_EXP_NAME:-rc_stack_bowls_chunk10}"
export OPENPI_GPUS="${OPENPI_GPUS:-0,1,2,9}"
export OPENPI_NPROC="${OPENPI_NPROC:-4}"
export OPENPI_NNODES="${OPENPI_NNODES:-1}"
export OPENPI_STANDALONE="${OPENPI_STANDALONE:-1}"
export OPENPI_RESUME="${OPENPI_RESUME:-0}"
export OPENPI_LOG_MODE="${OPENPI_LOG_MODE:-file}"
export OPENPI_LOG_DIR="${OPENPI_LOG_DIR:-./logs/training}"
export OPENPI_PROJECT_NAME="${OPENPI_PROJECT_NAME:-Robochallenge}"
export OPENPI_CHECKPOINT_PATH="${OPENPI_CHECKPOINT_PATH:-ELUBrain/RC/agilex_aloha_stack_bowls}"
export OPENPI_SAVE_INTERVAL="${OPENPI_SAVE_INTERVAL:-1000}"

bash scripts/train_pytorch.sh "$@"

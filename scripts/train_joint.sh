#!/usr/bin/env bash
set -euo pipefail

############################
# Joint (Multi-Task) Training Script
#
# Trains all 4 tasks mixed together in a single run. This is the baseline
# for continual learning experiments.
#
# Usage:
#   bash scripts/train_joint.sh
#   bash scripts/train_joint.sh --steps 10000
#   bash scripts/train_joint.sh --gpus "0,1,2,3" --dry-run
############################

############################
# 0) Defaults (can be overridden via env or CLI)
############################
export PATH="$HOME/.local/bin:$PATH"
export OPENPI_DATA_ROOT="${OPENPI_DATA_ROOT:-/data/datasets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${OPENPI_DATA_ROOT}}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-openpi-joint}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

############################
# 1) Training Hyperparameters
############################
CONFIG_NAME="${CONFIG_NAME:-pi05_piper_4tasks_joint}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"

############################
# 2) Hardware / Distributed
############################
GPUS="${OPENPI_GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${OPENPI_NPROC:-8}"
NNODES="${OPENPI_NNODES:-1}"
STANDALONE="${OPENPI_STANDALONE:-1}"

############################
# 3) Paths
############################
EXPERIMENT_NAME="${EXPERIMENT_NAME:-joint_$(date +'%m%d_%H%M')_steps${TRAIN_STEPS}}"
EXPERIMENT_DIR="./results/joint_training/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="${EXPERIMENT_DIR}/checkpoints"
LOG_DIR="${EXPERIMENT_DIR}/logs"

# Shared norm stats (computed across all 4 datasets)
NORM_STATS_DIR="./assets/pi05_piper_4tasks_joint/piper_4tasks_joint"
NORM_STATS_FILE="${NORM_STATS_DIR}/norm_stats.json"

############################
# 4) Parse CLI args
############################
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift;;
        --config-name)
            CONFIG_NAME="$2"; shift 2;;
        --steps)
            TRAIN_STEPS="$2"; shift 2;;
        --gpus)
            GPUS="$2"; shift 2;;
        --exp-name)
            EXPERIMENT_NAME="$2"; shift 2;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1;;
    esac
done

############################
# 5) Create directories
############################
mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}"

############################
# 6) Compute shared norm stats (if not already present)
############################
if [[ ! -f "$NORM_STATS_FILE" ]]; then
    echo ""
    echo "Shared norm stats not found: ${NORM_STATS_FILE}"
    echo "Computing norm stats across all 4 datasets..."
    echo "  config: ${CONFIG_NAME}"
    echo ""
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY RUN] python scripts/compute_norm_stats.py --config-name ${CONFIG_NAME}"
    else
        python scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
        if [[ ! -f "$NORM_STATS_FILE" ]]; then
            echo "[ERROR] Norm stats file was not created: ${NORM_STATS_FILE}"
            exit 1
        fi
        echo "Norm stats saved to: ${NORM_STATS_FILE}"
    fi
else
    echo "Shared norm stats found: ${NORM_STATS_FILE}"
fi

############################
# 7) Experiment Info
############################
cat > "${EXPERIMENT_DIR}/experiment_info.txt" <<INFO_EOF
# Joint Training Experiment (Baseline)
# Generated: $(date +'%Y-%m-%d %H:%M:%S')

## Config
  config_name:       ${CONFIG_NAME}
  num_train_steps:   ${TRAIN_STEPS}
  save_interval:     ${SAVE_INTERVAL}

## Datasets (4 tasks mixed)
  1. stack_bowls_20260413
  2. hang_cup_20260413
  3. press_button_20260414_trimmed
  4. fold_towel_20260417

## Hardware
  gpus:              ${GPUS}
  nproc:             ${NPROC}
  norm_stats_dir:    ${NORM_STATS_DIR}

## Output
  experiment_dir:    ${EXPERIMENT_DIR}
  checkpoints:       ${CHECKPOINT_DIR}
  logs:              ${LOG_DIR}
INFO_EOF
echo "Experiment dir: ${EXPERIMENT_DIR}"

############################
# 8) Export env vars for training
############################
export OPENPI_NORM_STATS_DIR="$NORM_STATS_DIR"
export OPENPI_NUM_TRAIN_STEPS="$TRAIN_STEPS"
export OPENPI_SAVE_INTERVAL="$SAVE_INTERVAL"

############################
# 9) Launch training
############################
TRAIN_ARGS=(
    --config-name "$CONFIG_NAME"
    --exp-name "$EXPERIMENT_NAME"
    --gpus "$GPUS"
    --nproc "$NPROC"
    --nnodes "$NNODES"
    --standalone "$STANDALONE"
    --mode file
    --log-dir "$LOG_DIR"
    --checkpoint-path "${CHECKPOINT_DIR}/${EXPERIMENT_NAME}"
    --save-interval "$SAVE_INTERVAL"
)

echo ""
echo "=========================================="
echo "  Joint Training (4-Task Baseline)"
echo "  Config:   ${CONFIG_NAME}"
echo "  Steps:    ${TRAIN_STEPS}"
echo "  GPUs:     ${GPUS}"
echo "  Save:     every ${SAVE_INTERVAL} steps"
echo "=========================================="
echo ""

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY RUN] bash scripts/train_pytorch.sh ${TRAIN_ARGS[*]}"
    echo "[DRY RUN] OPENPI_NORM_STATS_DIR=${OPENPI_NORM_STATS_DIR}"
    echo "[DRY RUN] OPENPI_NUM_TRAIN_STEPS=${OPENPI_NUM_TRAIN_STEPS}"
    echo "[DRY RUN] OPENPI_SAVE_INTERVAL=${OPENPI_SAVE_INTERVAL}"
else
    bash scripts/train_pytorch.sh "${TRAIN_ARGS[@]}"
fi

echo ""
echo "=========================================="
echo "  Joint Training Complete!"
echo "  Experiment dir: ${EXPERIMENT_DIR}"
echo "  Checkpoint dir: ${CHECKPOINT_DIR}/${EXPERIMENT_NAME}"
echo "=========================================="

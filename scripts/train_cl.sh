#!/usr/bin/env bash
set -euo pipefail

############################
# Continual Learning via Data Replay
#
# Trains sequentially on multiple tasks. After each task, a replay buffer
# is saved to disk. During subsequent tasks, old data is replayed at a
# configurable ratio to mitigate catastrophic forgetting.
#
# Usage:
#   bash scripts/train_cl.sh
#   bash scripts/train_cl.sh --dry-run
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
export WANDB_PROJECT="${WANDB_PROJECT:-openpi-cl}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

############################
# 1) Task Sequence
############################
# Each entry is: "config_name:data_root"
# - config_name: must match a TrainConfig name in config.py
# - data_root: local path to the dataset for this task
TASKS=(
    "pi05_piper_stack_bowls_20260413_4cam_hold_dim7_13:${OPENPI_DATA_ROOT}/realworld_piper/stack_bowls_20260413"
    "pi05_piper_hang_cup_20260413_4cam_hold_dim7_13:${OPENPI_DATA_ROOT}/realworld_piper/hang_cup_20260413"
    "pi05_piper_fold_towel_20260417_4cam_hold_dim7_13:${OPENPI_DATA_ROOT}/realworld_piper/fold_towel_20260417"
    "pi05_piper_press_button_20260414_4cam_hold_dim7_13:${OPENPI_DATA_ROOT}/realworld_piper/press_button_20260414"
)

# TASKS=(
#     "pi05_piper_hang_cup_20260413_4cam_hold_dim7_13:${OPENPI_DATA_ROOT}/realworld_piper/hang_cup_20260413"
#     "pi05_piper_fold_towel_20260417_4cam_hold_dim7_13:${OPENPI_DATA_ROOT}/realworld_piper/fold_towel_20260417"
#     "pi05_piper_stack_bowls_20260413_4cam_hold_dim7_13:${OPENPI_DATA_ROOT}/realworld_piper/stack_bowls_20260413"
#     "pi05_piper_press_button_20260414_4cam_hold_dim7_13:${OPENPI_DATA_ROOT}/realworld_piper/press_button_20260414"
# )


############################
# 2) Continual Learning Hyperparameters
############################
DATA_BUFFER_SIZE="${DATA_BUFFER_SIZE:-0.2}"     # Total fraction of data retained across all old tasks
DATA_REPLAY_RATIO="${DATA_REPLAY_RATIO:-0.2}"   # Per-step probability of sampling from replay buffer
REPLAY_MODE="${REPLAY_MODE:-episode}"            # "transition" or "episode"
# TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"               # Epochs per task
TRAIN_STEPS="${TRAIN_STEPS:-4000}"

############################
# 3) Hardware / Distributed
############################
GPUS="${OPENPI_GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${OPENPI_NPROC:-8}"
NNODES="${OPENPI_NNODES:-1}"
STANDALONE="${OPENPI_STANDALONE:-1}"

############################
# 4) Paths
############################
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(date +'%m%d_%H%M')_buffer${DATA_BUFFER_SIZE}_replay${DATA_REPLAY_RATIO}_steps${TRAIN_STEPS}_${REPLAY_MODE}_with_joint_norm_stats}"
EXPERIMENT_DIR="./results/continual_learning/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="${EXPERIMENT_DIR}/checkpoints"
BUFFER_DIR="${EXPERIMENT_DIR}/replay_buffers"
LOG_DIR="${EXPERIMENT_DIR}/logs"

mkdir -p "${CHECKPOINT_DIR}" "${BUFFER_DIR}" "${LOG_DIR}"

############################
# 5) Parse CLI args
############################
DRY_RUN=0
# NORM_STATS_DIR="./assets/pi05_piper_stack_bowls_20260413_4cam/piper_stack_bowls_20260413_4cam"

NORM_STATS_DIR="./assets/pi05_piper_4tasks_joint/piper_4tasks_joint"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift;;
        --norm-stats-dir)
            NORM_STATS_DIR="$2"; shift 2;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1;;
    esac
done

# Validate --norm-stats-dir if provided
if [[ -n "$NORM_STATS_DIR" ]]; then
    if [[ ! -d "$NORM_STATS_DIR" ]]; then
        echo "[ERROR] --norm-stats-dir directory does not exist: $NORM_STATS_DIR"
        exit 1
    fi
    if [[ ! -f "${NORM_STATS_DIR}/norm_stats.json" ]]; then
        echo "[ERROR] No norm_stats.json found in --norm-stats-dir: $NORM_STATS_DIR"
        exit 1
    fi
    export OPENPI_NORM_STATS_DIR="$NORM_STATS_DIR"
    echo "[INFO] Using shared norm stats from: ${OPENPI_NORM_STATS_DIR}"
fi

############################
# 6) Experiment Info
############################
cat > "${EXPERIMENT_DIR}/experiment_info.txt" <<INFO_EOF
# Continual Learning Experiment
# Generated: $(date +'%Y-%m-%d %H:%M:%S')

## Task Sequence (${#TASKS[@]} tasks)
$(for idx in "${!TASKS[@]}"; do
    IFS=':' read -r cfg root <<< "${TASKS[$idx]}"
    echo "  $((idx+1)). ${cfg} (data: ${root})"
done)

## Hyperparameters
  buffer_size:       ${DATA_BUFFER_SIZE}
  replay_ratio:      ${DATA_REPLAY_RATIO}
  replay_mode:       ${REPLAY_MODE}
  steps_per_task:    ${TRAIN_STEPS}

## Hardware
  gpus:              ${GPUS}
  nproc:             ${NPROC}
  norm_stats_dir:    ${NORM_STATS_DIR:-<per-task default>}

## Output
  checkpoints:       ${CHECKPOINT_DIR}
  replay_buffers:    ${BUFFER_DIR}
INFO_EOF
echo "Experiment dir: ${EXPERIMENT_DIR}"

############################
# 7) Helper: find latest checkpoint
############################
find_latest_checkpoint() {
    local ckpt_dir="$1"
    local latest
    latest=$(find "$ckpt_dir" -name "model.safetensors" -type f 2>/dev/null \
        | sed 's|.*/\([0-9]*\)/model.safetensors|\1|' \
        | sort -n \
        | tail -1)
    if [[ -n "$latest" ]]; then
        echo "${ckpt_dir}/${latest}"
    fi
}

############################
# 8) Sequential training loop
############################
echo ""
echo "Starting continual learning with ${#TASKS[@]} tasks"
echo "  buffer_size=${DATA_BUFFER_SIZE}, replay_ratio=${DATA_REPLAY_RATIO}"
echo "  replay_mode=${REPLAY_MODE}, steps=${TRAIN_STEPS}"
echo ""

PRETRAINED_CKPT=""
PRETRAINED_WEIGHT=""

for i in "${!TASKS[@]}"; do
    IFS=':' read -r CONFIG_NAME DATA_ROOT <<< "${TASKS[$i]}"
    TASK_INDEX=$((i + 1))
    NUM_OLD_TASKS=$i
    NUM_TOTAL_TASKS=$TASK_INDEX

    EXP_NAME="cl_task${TASK_INDEX}_$(date +%m%d_%H%M)"

    echo ""
    echo "=========================================="
    echo "  Task ${TASK_INDEX}/${#TASKS[@]}: ${CONFIG_NAME}"
    echo "  Data root: ${DATA_ROOT}"
    echo "  Old tasks: ${NUM_OLD_TASKS}"
    echo "  Per-task buffer fraction: $(awk "BEGIN {printf \"%.4f\", ${DATA_BUFFER_SIZE} / ${NUM_TOTAL_TASKS}}")"
    echo "=========================================="

    # Build training command
    TRAIN_ARGS=(
        --config-name "$CONFIG_NAME"
        --exp-name "$EXP_NAME"
        --gpus "$GPUS"
        --nproc "$NPROC"
        --nnodes "$NNODES"
        --standalone "$STANDALONE"
        --mode file
        --log-dir "$LOG_DIR"
        --checkpoint-path "$CHECKPOINT_DIR"
    )

    # Set CL-specific env vars for the training script
    export OPENPI_CONFIG_NAME="$CONFIG_NAME"
    export OPENPI_EXP_NAME="$EXP_NAME"
    export OPENPI_GPUS="$GPUS"
    export OPENPI_NPROC="$NPROC"
    export OPENPI_NNODES="$NNODES"
    export OPENPI_STANDALONE="$STANDALONE"
    export OPENPI_LOG_MODE="file"
    export OPENPI_LOG_DIR="$LOG_DIR"

    # Set CL replay parameters via environment
    # The training script will read these from env
    if [[ $NUM_OLD_TASKS -gt 0 ]]; then
        export OPENPI_REPLAY_BUFFER_DIR="$BUFFER_DIR"
        export OPENPI_REPLAY_RATIO="$DATA_REPLAY_RATIO"
    else
        unset OPENPI_REPLAY_BUFFER_DIR 2>/dev/null || true
        unset OPENPI_REPLAY_RATIO 2>/dev/null || true
    fi

    # Epoch-based training
    # export OPENPI_EPOCHS="$TRAIN_EPOCHS"

    export OPENPI_NUM_TRAIN_STEPS="$TRAIN_STEPS"

    # Optional: override total training steps (disables epoch-based auto-calc)
    if [[ -n "${TRAIN_STEPS:-}" ]]; then
        export OPENPI_NUM_TRAIN_STEPS="$TRAIN_STEPS"
    fi

    # Optional: override save interval
    if [[ -n "${SAVE_INTERVAL:-}" ]]; then
        export OPENPI_SAVE_INTERVAL="$SAVE_INTERVAL"
    fi

    # Resume from pretrained checkpoint (load weights only, not optimizer state)
    # The training script handles weight loading via pytorch_weight_path
    # For CL, we need to load the previous task's checkpoint
    if [[ -n "$PRETRAINED_WEIGHT" ]]; then
        export OPENPI_PYTORCH_WEIGHT_PATH="$PRETRAINED_WEIGHT"
    fi

    echo "Launching training for task: ${CONFIG_NAME} (${TRAIN_STEPS} steps)"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY RUN] Would run: bash scripts/train_pytorch.sh ${TRAIN_ARGS[*]}"
        echo "[DRY RUN] OPENPI_REPLAY_BUFFER_DIR=${OPENPI_REPLAY_BUFFER_DIR:-<none>}"
        echo "[DRY RUN] OPENPI_REPLAY_RATIO=${OPENPI_REPLAY_RATIO:-<none>}"
        echo "[DRY RUN] OPENPI_EPOCHS=${OPENPI_EPOCHS}"
        echo "[DRY RUN] OPENPI_PYTORCH_WEIGHT_PATH=${PRETRAINED_WEIGHT:-<none>}"
        echo "[DRY RUN] OPENPI_NORM_STATS_DIR=${OPENPI_NORM_STATS_DIR:-<per-task default>}"
    else
        bash scripts/train_pytorch.sh "${TRAIN_ARGS[@]}"
    fi

    echo "Training complete for task: ${CONFIG_NAME}"

    # Post-training: build new buffer + downsample old buffers
    echo "Building/downsampling replay buffers (mode=${REPLAY_MODE})..."

    BUILD_ARGS=(
        --config-name "$CONFIG_NAME"
        --data-root "$DATA_ROOT"
        --buffer-dir "$BUFFER_DIR"
        --buffer-size "$DATA_BUFFER_SIZE"
        --num-total-tasks "$NUM_TOTAL_TASKS"
        --mode "$REPLAY_MODE"
        --seed 42
    )

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY RUN] Would run: python scripts/build_replay_buffers.py ${BUILD_ARGS[*]}"
    else
        python scripts/build_replay_buffers.py "${BUILD_ARGS[@]}"
    fi

    # Find the latest checkpoint from this task
    # Checkpoints are saved in ${CHECKPOINT_DIR}/${EXP_NAME}/
    LATEST_EXP_DIR="${CHECKPOINT_DIR}/${EXP_NAME}"

    if [[ -n "$LATEST_EXP_DIR" ]]; then
        PRETRAINED_CKPT=$(find_latest_checkpoint "$LATEST_EXP_DIR")
    fi

    if [[ -n "$PRETRAINED_CKPT" ]]; then
        PRETRAINED_WEIGHT="$PRETRAINED_CKPT"
        echo "Next pretrained checkpoint: ${PRETRAINED_WEIGHT}"
    else
        echo "WARNING: No checkpoint found for task ${CONFIG_NAME}"
        echo "  The next task will train from the base pretrained weights."
        PRETRAINED_WEIGHT=""
    fi

    echo "Task ${TASK_INDEX} (${CONFIG_NAME}) complete!"
done

echo ""
echo "=========================================="
echo "  Continual Learning Complete!"
echo "  Tasks trained: ${#TASKS[@]}"
echo "  Experiment dir: ${EXPERIMENT_DIR}"
echo "  Buffer dir: ${BUFFER_DIR}"
echo "  Final checkpoint: ${PRETRAINED_WEIGHT:-<none>}"
echo "=========================================="

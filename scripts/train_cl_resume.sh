#!/usr/bin/env bash
set -euo pipefail

############################
# Continual Learning Resume Script
#
# Resume CL training from a specific task, loading a checkpoint from a
# previous run and optionally reusing its replay buffers.
#
# Usage:
#   bash scripts/train_cl_resume.sh --start-task 3 --resume-checkpoint /path/to/ckpt
#   bash scripts/train_cl_resume.sh --start-task 5 --resume-checkpoint /path/to/ckpt --resume-buffer-dir /path/to/buffers
#   bash scripts/train_cl_resume.sh --start-task 3 --resume-checkpoint /path/to/ckpt --dry-run
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
TASKS=(
    "pi05_piper_stack_bowls_20260413_4cam_hold_dim7_13:${OPENPI_DATA_ROOT}/realworld_piper/stack_bowls_20260413"
    "pi05_piper_hang_cup_20260413_4cam_hold_dim7_13:${OPENPI_DATA_ROOT}/realworld_piper/hang_cup_20260413"
    "pi05_piper_press_button_20260414_4cam_hold_dim7_13:${OPENPI_DATA_ROOT}/realworld_piper/press_button_20260414"
    "pi05_piper_fold_towel_20260417_4cam_hold_dim7_13:${OPENPI_DATA_ROOT}/realworld_piper/fold_towel_20260417"
)
NUM_TOTAL_TASKS=${#TASKS[@]}

############################
# 2) Continual Learning Hyperparameters
############################
DATA_BUFFER_SIZE="${DATA_BUFFER_SIZE:-0.2}"
DATA_REPLAY_RATIO="${DATA_REPLAY_RATIO:-0}"
REPLAY_MODE="${REPLAY_MODE:-episode}"
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
# EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(date +'%m%d_%H%M')_buffer${DATA_BUFFER_SIZE}_replay${DATA_REPLAY_RATIO}_steps${TRAIN_STEPS}_${REPLAY_MODE}_resume}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(date +'%m%d_%H%M')_buffer${DATA_BUFFER_SIZE}_replay${DATA_REPLAY_RATIO}_steps${TRAIN_STEPS}_${REPLAY_MODE}_self_norm_stats}"
EXPERIMENT_DIR="./results/continual_learning/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="${EXPERIMENT_DIR}/checkpoints"
LOG_DIR="${EXPERIMENT_DIR}/logs"

############################
# 5) Parse CLI args
############################
DRY_RUN=0
START_TASK="2"
RESUME_CHECKPOINT="./results/continual_learning/0428_2350_buffer0.2_replay0.3_steps4000_episode/checkpoints/cl_task1_0428_2350/4000"
RESUME_BUFFER_DIR=""
# NORM_STATS_DIR="./assets/pi05_piper_stack_bowls_20260413_4cam/piper_stack_bowls_20260413_4cam"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift;;
        --start-task)
            START_TASK="$2"; shift 2;;
        --resume-checkpoint)
            RESUME_CHECKPOINT="$2"; shift 2;;
        --resume-buffer-dir)
            RESUME_BUFFER_DIR="$2"; shift 2;;
        --norm-stats-dir)
            NORM_STATS_DIR="$2"; shift 2;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1;;
    esac
done

# --- Validate required args ---
if [[ -z "$START_TASK" ]]; then
    echo "[ERROR] --start-task is required (e.g., --start-task 3 to resume from task 3)"
    exit 1
fi
if ! [[ "$START_TASK" =~ ^[0-9]+$ ]] || [[ "$START_TASK" -lt 1 ]] || [[ "$START_TASK" -gt "$NUM_TOTAL_TASKS" ]]; then
    echo "[ERROR] --start-task must be between 1 and ${NUM_TOTAL_TASKS}"
    exit 1
fi
if [[ -z "$RESUME_CHECKPOINT" ]]; then
    echo "[ERROR] --resume-checkpoint is required (path to pretrained weights)"
    exit 1
fi
if [[ ! -d "$RESUME_CHECKPOINT" ]]; then
    echo "[ERROR] --resume-checkpoint must be a directory containing model.safetensors: $RESUME_CHECKPOINT"
    exit 1
fi
if [[ ! -f "${RESUME_CHECKPOINT}/model.safetensors" ]]; then
    echo "[ERROR] No model.safetensors found in --resume-checkpoint: ${RESUME_CHECKPOINT}/model.safetensors"
    exit 1
fi

# --- Validate optional args ---
if [[ -n "$RESUME_BUFFER_DIR" ]]; then
    if [[ ! -d "$RESUME_BUFFER_DIR" ]]; then
        echo "[ERROR] --resume-buffer-dir directory does not exist: $RESUME_BUFFER_DIR"
        exit 1
    fi
    BUFFER_DIR="$RESUME_BUFFER_DIR"
else
    BUFFER_DIR="${EXPERIMENT_DIR}/replay_buffers"
fi

if [[ -n "${NORM_STATS_DIR:-}" ]]; then
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

# --- Create directories ---
mkdir -p "${CHECKPOINT_DIR}" "${BUFFER_DIR}" "${LOG_DIR}"

############################
# 6) Experiment Info
############################
COMPLETED_TASKS=$((START_TASK - 1))
cat > "${EXPERIMENT_DIR}/experiment_info.txt" <<INFO_EOF
# Continual Learning Resume Experiment
# Generated: $(date +'%Y-%m-%d %H:%M:%S')

## Resume Info
  start_task:         ${START_TASK}
  completed_tasks:    ${COMPLETED_TASKS}
  resume_checkpoint:  ${RESUME_CHECKPOINT}
  resume_buffer_dir:  ${RESUME_BUFFER_DIR:-<new>}

## Full Task Sequence (${NUM_TOTAL_TASKS} tasks)
$(for idx in "${!TASKS[@]}"; do
    IFS=':' read -r cfg root <<< "${TASKS[$idx]}"
    task_num=$((idx + 1))
    if [[ $task_num -lt $START_TASK ]]; then
        echo "  ${task_num}. [DONE] ${cfg}"
    elif [[ $task_num -eq $START_TASK ]]; then
        echo "  ${task_num}. [RESUME] ${cfg} (data: ${root})"
    else
        echo "  ${task_num}. ${cfg} (data: ${root})"
    fi
done)

## Hyperparameters
  buffer_size:       ${DATA_BUFFER_SIZE}
  replay_ratio:      ${DATA_REPLAY_RATIO}
  replay_mode:       ${REPLAY_MODE}
  train_steps:       ${TRAIN_STEPS}

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
# 8) Rebuild replay buffers for completed tasks
############################
rebuild_task_buffer() {
    local task_idx="$1"
    IFS=':' read -r cfg root <<< "${TASKS[$((task_idx - 1))]}"
    local buf_file="${BUFFER_DIR}/task${task_idx}_buffer.json"

    if [[ -f "$buf_file" ]]; then
        echo "  [SKIP] task${task_idx}_buffer.json already exists"
        return
    fi

    echo "  [BUILD] task${task_idx}: ${cfg}"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [DRY RUN] python scripts/build_replay_buffers.py --config-name ${cfg} --data-root ${root} --buffer-dir ${BUFFER_DIR} --buffer-size ${DATA_BUFFER_SIZE} --num-total-tasks ${task_idx} --mode ${REPLAY_MODE} --seed 42"
    else
        python scripts/build_replay_buffers.py \
            --config-name "$cfg" \
            --data-root "$root" \
            --buffer-dir "$BUFFER_DIR" \
            --buffer-size "$DATA_BUFFER_SIZE" \
            --num-total-tasks "$task_idx" \
            --mode "$REPLAY_MODE" \
            --seed 42
    fi
}

if [[ $COMPLETED_TASKS -gt 0 ]]; then
    echo ""
    echo "Rebuilding replay buffers for ${COMPLETED_TASKS} completed task(s)..."
    for (( t=1; t<=COMPLETED_TASKS; t++ )); do
        rebuild_task_buffer "$t"
    done
    echo "Replay buffer rebuild complete."
fi

############################
# 9) Sequential training loop (from START_TASK)
############################
echo ""
echo "Starting continual learning (resume from task ${START_TASK})"
echo "  buffer_size=${DATA_BUFFER_SIZE}, replay_ratio=${DATA_REPLAY_RATIO}"
echo "  replay_mode=${REPLAY_MODE}, steps=${TRAIN_STEPS}"
echo "  pretrained: ${RESUME_CHECKPOINT}"
echo ""

PRETRAINED_WEIGHT="$RESUME_CHECKPOINT"

# Loop from START_TASK to the end
for (( i=START_TASK-1; i<NUM_TOTAL_TASKS; i++ )); do
    IFS=':' read -r CONFIG_NAME DATA_ROOT <<< "${TASKS[$i]}"
    TASK_INDEX=$((i + 1))
    NUM_OLD_TASKS=$i

    EXP_NAME="cl_task${TASK_INDEX}_$(date +%m%d_%H%M)"

    echo ""
    echo "=========================================="
    echo "  Task ${TASK_INDEX}/${NUM_TOTAL_TASKS}: ${CONFIG_NAME}"
    echo "  Data root: ${DATA_ROOT}"
    echo "  Old tasks: ${NUM_OLD_TASKS}"
    echo "  Per-task buffer fraction: $(awk "BEGIN {printf \"%.4f\", ${DATA_BUFFER_SIZE} / ${TASK_INDEX}}")"
    echo "  Pretrained weight: ${PRETRAINED_WEIGHT}"
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
        --checkpoint-path "${CHECKPOINT_DIR}/${EXP_NAME}"
    )

    # Set CL-specific env vars
    export OPENPI_CONFIG_NAME="$CONFIG_NAME"
    export OPENPI_EXP_NAME="$EXP_NAME"
    export OPENPI_GPUS="$GPUS"
    export OPENPI_NPROC="$NPROC"
    export OPENPI_NNODES="$NNODES"
    export OPENPI_STANDALONE="$STANDALONE"
    export OPENPI_LOG_MODE="file"
    export OPENPI_LOG_DIR="$LOG_DIR"

    # Replay buffers: always enabled for resumed tasks
    export OPENPI_REPLAY_BUFFER_DIR="$BUFFER_DIR"
    export OPENPI_REPLAY_RATIO="$DATA_REPLAY_RATIO"

    # Step-based training
    export OPENPI_NUM_TRAIN_STEPS="$TRAIN_STEPS"

    # Optional: override save interval
    if [[ -n "${SAVE_INTERVAL:-}" ]]; then
        export OPENPI_SAVE_INTERVAL="$SAVE_INTERVAL"
    fi

    # Load pretrained weights from previous task's checkpoint
    export OPENPI_PYTORCH_WEIGHT_PATH="$PRETRAINED_WEIGHT"

    echo "Launching training for task: ${CONFIG_NAME} (${TRAIN_STEPS} steps)"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY RUN] Would run: bash scripts/train_pytorch.sh ${TRAIN_ARGS[*]}"
        echo "[DRY RUN] OPENPI_REPLAY_BUFFER_DIR=${OPENPI_REPLAY_BUFFER_DIR}"
        echo "[DRY RUN] OPENPI_REPLAY_RATIO=${OPENPI_REPLAY_RATIO}"
        echo "[DRY RUN] OPENPI_NUM_TRAIN_STEPS=${OPENPI_NUM_TRAIN_STEPS}"
        echo "[DRY RUN] OPENPI_PYTORCH_WEIGHT_PATH=${PRETRAINED_WEIGHT}"
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
        --num-total-tasks "$TASK_INDEX"
        --mode "$REPLAY_MODE"
        --seed 42
    )

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY RUN] Would run: python scripts/build_replay_buffers.py ${BUILD_ARGS[*]}"
    else
        python scripts/build_replay_buffers.py "${BUILD_ARGS[@]}"
    fi

    # Find the latest checkpoint from this task
    PRETRAINED_CKPT=$(find_latest_checkpoint "${CHECKPOINT_DIR}/${EXP_NAME}")

    if [[ -n "${PRETRAINED_CKPT:-}" ]]; then
        PRETRAINED_WEIGHT="$PRETRAINED_CKPT"
        echo "Next pretrained checkpoint: ${PRETRAINED_WEIGHT}"
    else
        echo "[ERROR] No checkpoint found for task ${CONFIG_NAME} in ${CHECKPOINT_DIR}/${EXP_NAME}"
        exit 1
    fi

    echo "Task ${TASK_INDEX} (${CONFIG_NAME}) complete!"
done

echo ""
echo "=========================================="
echo "  Continual Learning Resume Complete!"
echo "  Tasks trained: $((NUM_TOTAL_TASKS - COMPLETED_TASKS)) (from task ${START_TASK} to ${NUM_TOTAL_TASKS})"
echo "  Experiment dir: ${EXPERIMENT_DIR}"
echo "  Buffer dir: ${BUFFER_DIR}"
echo "  Final checkpoint: ${PRETRAINED_WEIGHT:-<none>}"
echo "=========================================="

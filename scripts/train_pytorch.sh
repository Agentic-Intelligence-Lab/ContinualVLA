#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# PyTorch DDP Training Launcher for OpenPI (uv + torchrun)
# ------------------------------------------------------------------------------
# Add-on:
#   - --resume {0|1} : resume from latest checkpoint under the experiment ckpt dir
#     (default from OPENPI_RESUME, fallback 1)
# ==============================================================================

# ----------------------------
# Defaults (can be overridden)
# ----------------------------
export PATH="$HOME/.local/bin:$PATH"
export OPENPI_DATA_ROOT="${OPENPI_DATA_ROOT:-/data/datasets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${OPENPI_DATA_ROOT}}"
export WANDB_MODE="${WANDB_MODE:-online}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
# PyTorch training only needs JAX for utility code. Keep JAX on CPU so it does not
# initialize CUDA contexts on GPUs outside CUDA_VISIBLE_DEVICES.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

# Training defaults (env overridable)
DEFAULT_CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_galaxea_r1lite_Fold_Green_Tshirt}"
DEFAULT_EXP_NAME="${OPENPI_EXP_NAME:-pi05_galaxea_fold_green_Tshirt_300_chunk10}"
DEFAULT_GPUS="${OPENPI_GPUS:-4,5,6,7}"  # e.g., "0,1,2,"
DEFAULT_NPROC="${OPENPI_NPROC:-4}"
DEFAULT_NNODES="${OPENPI_NNODES:-1}"
DEFAULT_STANDALONE="${OPENPI_STANDALONE:-1}"   # 1: use --standalone, 0: do not
DEFAULT_LOG_DIR="${OPENPI_LOG_DIR:-./logs/training}"
DEFAULT_PROJECT_NAME="${OPENPI_PROJECT_NAME:-}"
DEFAULT_CHECKPOINT_PATH="${OPENPI_CHECKPOINT_PATH:-}"
DEFAULT_SAVE_INTERVAL="${OPENPI_SAVE_INTERVAL:-}"

# Resume (env overridable)
# 1: add --resume to training command; 0: do not resume
DEFAULT_RESUME="${OPENPI_RESUME:-0}"

# Log mode (env overridable)
# Supported: file | terminal | both
DEFAULT_MODE="${OPENPI_LOG_MODE:-file}"

# ----------------------------
# Helper: usage
# ----------------------------
usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_pytorch.sh [options]

Options:
  --mode {file|terminal|both}   Logging mode. CLI overrides env OPENPI_LOG_MODE.
  --exp-name NAME               Experiment name (default from OPENPI_EXP_NAME or pi05_galaxea_test)
  --config-name NAME            Config name (default from OPENPI_CONFIG_NAME or pi05_galaxea_r1lite)
  --gpus "0,1,2,3"              CUDA_VISIBLE_DEVICES (default from OPENPI_GPUS or 4,5,6,7)
  --nproc N                     torchrun --nproc_per_node (default from OPENPI_NPROC or 4)
  --nnodes N                    torchrun --nnodes (default from OPENPI_NNODES or 1)
  --standalone {0|1}            Whether to pass torchrun --standalone (default from OPENPI_STANDALONE or 1)
  --resume {0|1}                Whether to resume from latest checkpoint (default from OPENPI_RESUME or 1)
  --log-dir DIR                 Base directory to store log files (default from OPENPI_LOG_DIR or ./logs)
  --project-name NAME           W&B project name override
  --checkpoint-path DIR         Explicit checkpoint directory override
  --save-interval N             Save a checkpoint every N training steps
  --dry-run                     Print the final command and exit
  -h, --help                    Show this help

Environment variables (optional):
  OPENPI_LOG_MODE, OPENPI_EXP_NAME, OPENPI_CONFIG_NAME, OPENPI_GPUS,
  OPENPI_NPROC, OPENPI_NNODES, OPENPI_STANDALONE, OPENPI_RESUME, OPENPI_LOG_DIR,
  OPENPI_PROJECT_NAME, OPENPI_CHECKPOINT_PATH, OPENPI_SAVE_INTERVAL,
  HF_LEROBOT_HOME, WANDB_MODE

Examples:
  # resume (default)
  bash scripts/train_pytorch.sh

  # explicitly resume
  bash scripts/train_pytorch.sh --resume 1

  # from scratch
  bash scripts/train_pytorch.sh --resume 0

  # resume + both logging
  bash scripts/train_pytorch.sh --resume 1 --mode both
EOF
}

# ----------------------------
# Parse args
# ----------------------------
MODE="$DEFAULT_MODE"
EXP_NAME="$DEFAULT_EXP_NAME"
CONFIG_NAME="$DEFAULT_CONFIG_NAME"
GPUS="$DEFAULT_GPUS"
NPROC="$DEFAULT_NPROC"
NNODES="$DEFAULT_NNODES"
STANDALONE="$DEFAULT_STANDALONE"
RESUME="$DEFAULT_RESUME"
LOG_DIR="$DEFAULT_LOG_DIR"
PROJECT_NAME="$DEFAULT_PROJECT_NAME"
CHECKPOINT_PATH="$DEFAULT_CHECKPOINT_PATH"
SAVE_INTERVAL="$DEFAULT_SAVE_INTERVAL"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"; shift 2;;
    --exp-name|--exp_name)
      EXP_NAME="${2:-}"; shift 2;;
    --config-name|--config_name)
      CONFIG_NAME="${2:-}"; shift 2;;
    --gpus)
      GPUS="${2:-}"; shift 2;;
    --nproc)
      NPROC="${2:-}"; shift 2;;
    --nnodes)
      NNODES="${2:-}"; shift 2;;
    --standalone)
      STANDALONE="${2:-}"; shift 2;;
    --resume)
      RESUME="${2:-1}"; shift 2;;
    --log-dir|--log_dir)
      LOG_DIR="${2:-}"; shift 2;;
    --project-name|--project_name)
      PROJECT_NAME="${2:-}"; shift 2;;
    --checkpoint-path|--checkpoint_path)
      CHECKPOINT_PATH="${2:-}"; shift 2;;
    --save-interval|--save_interval)
      SAVE_INTERVAL="${2:-}"; shift 2;;
    --dry-run)
      DRY_RUN=1; shift 1;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1;;
  esac
done

# ----------------------------
# Validate inputs
# ----------------------------
if [[ -z "$MODE" ]]; then
  echo "[ERROR] --mode is empty. Supported: file | terminal | both"
  exit 1
fi

case "$MODE" in
  file|terminal|both) ;;
  *)
    echo "[ERROR] Invalid mode: $MODE. Supported: file | terminal | both"
    exit 1;;
esac

if [[ -z "$EXP_NAME" || -z "$CONFIG_NAME" ]]; then
  echo "[ERROR] exp-name or config-name is empty."
  exit 1
fi

if ! [[ "$NPROC" =~ ^[0-9]+$ ]] || ! [[ "$NNODES" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] --nproc and --nnodes must be integers."
  exit 1
fi

if ! [[ "$STANDALONE" =~ ^[01]$ ]]; then
  echo "[ERROR] --standalone must be 0 or 1."
  exit 1
fi

if ! [[ "$RESUME" =~ ^[01]$ ]]; then
  echo "[ERROR] --resume must be 0 or 1."
  exit 1
fi

if [[ -n "$SAVE_INTERVAL" ]] && ! [[ "$SAVE_INTERVAL" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] --save-interval must be an integer."
  exit 1
fi

# ----------------------------
# Normalize / validate GPU list
# ----------------------------
GPUS="${GPUS//[[:space:]]/}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
declare -A GPU_SEEN=()
GPU_LIST=()
for gpu in "${GPU_ARRAY[@]}"; do
  if [[ -z "$gpu" ]]; then
    continue
  fi
  if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid GPU id in --gpus: '$gpu'"
    exit 1
  fi
  if [[ -n "${GPU_SEEN[$gpu]:-}" ]]; then
    echo "[ERROR] Duplicate GPU id in --gpus: '$gpu'"
    exit 1
  fi
  GPU_SEEN[$gpu]=1
  GPU_LIST+=("$gpu")
done

if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "[ERROR] --gpus resolved to an empty GPU list."
  exit 1
fi

if (( NPROC > ${#GPU_LIST[@]} )); then
  echo "[ERROR] --nproc ($NPROC) cannot exceed the number of visible GPUs (${#GPU_LIST[@]})."
  exit 1
fi

GPUS="$(IFS=,; echo "${GPU_LIST[*]}")"
VISIBLE_GPU_COUNT="${#GPU_LIST[@]}"
JAX_VISIBLE_LOCAL_GPUS=""
for i in "${!GPU_LIST[@]}"; do
  if [[ -n "$JAX_VISIBLE_LOCAL_GPUS" ]]; then
    JAX_VISIBLE_LOCAL_GPUS+=","
  fi
  JAX_VISIBLE_LOCAL_GPUS+="$i"
done

# ----------------------------
# Export CUDA devices
# ----------------------------
export CUDA_VISIBLE_DEVICES="$GPUS"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-$GPUS}"
export JAX_CUDA_VISIBLE_DEVICES="${JAX_CUDA_VISIBLE_DEVICES:-$JAX_VISIBLE_LOCAL_GPUS}"

# ----------------------------
# Build torchrun args
# ----------------------------
TORCHRUN_ARGS=(--nnodes="$NNODES" --nproc_per_node="$NPROC")
if [[ "$STANDALONE" == "1" ]]; then
  TORCHRUN_ARGS=(--standalone "${TORCHRUN_ARGS[@]}")
fi

# ----------------------------
# Prepare log paths (only used when mode=file/both)
# ----------------------------
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${LOG_DIR%/}/${EXP_NAME}"
LOG_FILE="${RUN_DIR}/${TS}.log"

if [[ "$MODE" == "file" || "$MODE" == "both" ]]; then
  mkdir -p "$RUN_DIR"
fi

# ----------------------------
# Final command to run
# ----------------------------
# Use the repo-local Python interpreter directly to avoid picking up a torchrun wrapper
# whose shebang points at a different virtual environment.
CMD=(
  env
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
  NVIDIA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"
  CUDA_DEVICE_ORDER="$CUDA_DEVICE_ORDER"
  JAX_PLATFORMS="$JAX_PLATFORMS"
  JAX_PLATFORM_NAME="$JAX_PLATFORM_NAME"
  JAX_CUDA_VISIBLE_DEVICES="$JAX_CUDA_VISIBLE_DEVICES"
  XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PYTHON_CLIENT_PREALLOCATE"
  XLA_PYTHON_CLIENT_ALLOCATOR="$XLA_PYTHON_CLIENT_ALLOCATOR"
  HF_LEROBOT_HOME="$HF_LEROBOT_HOME"
  WANDB_MODE="$WANDB_MODE"
  python3
  -m
  torch.distributed.run
  "${TORCHRUN_ARGS[@]}"
  scripts/train_pytorch.py
  "$CONFIG_NAME"
  --exp-name
  "$EXP_NAME"
)
if [[ "$RESUME" == "1" ]]; then
  CMD+=(--resume)
fi
if [[ -n "$PROJECT_NAME" ]]; then
  CMD+=(--project-name "$PROJECT_NAME")
fi
if [[ -n "$CHECKPOINT_PATH" ]]; then
  CMD+=(--checkpoint-path "$CHECKPOINT_PATH")
fi
if [[ -n "$SAVE_INTERVAL" ]]; then
  CMD+=(--save-interval "$SAVE_INTERVAL")
fi

echo "============================================================"
echo "[INFO] mode          : $MODE"
echo "[INFO] exp_name       : $EXP_NAME"
echo "[INFO] config_name    : $CONFIG_NAME"
echo "[INFO] gpus           : $CUDA_VISIBLE_DEVICES"
echo "[INFO] visible_gpus   : $VISIBLE_GPU_COUNT"
echo "[INFO] nnodes/nproc   : $NNODES / $NPROC"
echo "[INFO] standalone     : $STANDALONE"
echo "[INFO] resume         : $RESUME"
echo "[INFO] WANDB_MODE     : ${WANDB_MODE:-}"
echo "[INFO] HF_LEROBOT_HOME: ${HF_LEROBOT_HOME:-}"
echo "[INFO] JAX_PLATFORMS  : ${JAX_PLATFORMS:-}"
echo "[INFO] JAX_PLATFORM   : ${JAX_PLATFORM_NAME:-}"
echo "[INFO] JAX CUDA vis   : ${JAX_CUDA_VISIBLE_DEVICES:-}"
if [[ -n "$PROJECT_NAME" ]]; then
  echo "[INFO] project_name   : $PROJECT_NAME"
fi
if [[ -n "$CHECKPOINT_PATH" ]]; then
  echo "[INFO] checkpoint_path: $CHECKPOINT_PATH"
fi
if [[ -n "$SAVE_INTERVAL" ]]; then
  echo "[INFO] save_interval  : $SAVE_INTERVAL"
fi
if [[ "$MODE" == "file" || "$MODE" == "both" ]]; then
  echo "[INFO] log_file       : $LOG_FILE"
fi
echo "============================================================"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY RUN] ${CMD[*]}"
  exit 0
fi

# ----------------------------
# Launch according to mode
# ----------------------------
if [[ "$MODE" == "terminal" ]]; then
  # Foreground, terminal only
  exec "${CMD[@]}"

elif [[ "$MODE" == "both" ]]; then
  # Foreground, terminal + file
  if command -v stdbuf >/dev/null 2>&1; then
    exec stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
  else
    exec "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
  fi

else
  # MODE == file — run in foreground, tee output to log file
  if command -v stdbuf >/dev/null 2>&1; then
    stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
  else
    "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
  fi
fi

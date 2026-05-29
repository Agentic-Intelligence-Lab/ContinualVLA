#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

CONFIG_NAME="${OPENPI_CONFIG_NAME:-RC_Stack_Bowls}"
LOG_DIR="${OPENPI_COMPUTE_LOG_DIR:-logs/compute}"
MAX_FRAMES="${OPENPI_MAX_FRAMES:-}"

mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR%/}/compute_norm_stats_stack_bowls_${TS}.log"
PID_FILE="${LOG_DIR%/}/compute_norm_stats_stack_bowls.pid"

CMD=(uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME")
if [[ -n "$MAX_FRAMES" ]]; then
  CMD+=(--max-frames "$MAX_FRAMES")
fi

echo "[INFO] config_name : $CONFIG_NAME"
echo "[INFO] log_file    : $LOG_FILE"
echo "[INFO] pid_file    : $PID_FILE"
echo "[INFO] command     : ${CMD[*]}"

nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"

echo "[INFO] Started compute_norm_stats in background."
echo "[INFO] PID=$(cat "$PID_FILE")"
echo "[INFO] tail -f \"$LOG_FILE\""

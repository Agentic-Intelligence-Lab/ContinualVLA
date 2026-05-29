#!/usr/bin/env python3
"""
LeRobot Parquet quick inspection (NO video decoding).

Supports:
- GALAXEA style: observation.state.* and action.*
- LIBERO style: state and actions (plus image columns, which we DO NOT load)

Features:
(A) Print Parquet schema column names
(B) Print per-block dims (when blocks exist) and packed dims
(C) Print first N rows of actual values for state/actions to infer semantics
(D) Print an OpenPI-ready spec snippet (keys + dims) for copy/paste

Dependencies:
  pip install -U pyarrow numpy
"""

import os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

# =============================================================================
# User-editable paths
# =============================================================================
_DATA_ROOT = os.environ.get("OPENPI_DATA_ROOT", "/data/datasets")
ROOT = Path(_DATA_ROOT) / "physical-intelligence";      TASK_NAME = "libero"

PARQUET_PATH = None  # optionally set a specific parquet file path
PREVIEW_ROWS = 5     # print first N rows of values
FLOAT_PREC = 5       # printing precision


# =============================================================================
# Helpers
# =============================================================================
def find_any_parquet(task_root: Path) -> Path:
    data_dir = task_root / "data"
    parquets = sorted(data_dir.rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet found under: {data_dir}")
    return parquets[0]


def print_parquet_columns(task_root: Path) -> Path:
    parquet_file = find_any_parquet(task_root)
    print(f"[Schema] Using parquet: {parquet_file}")

    pf = pq.ParquetFile(parquet_file)
    names = pf.schema_arrow.names

    print("\n=== Columns (Parquet schema) ===")
    for name in names:
        print(name)

    # IMPORTANT: avoid loading row groups blindly (LIBERO has image columns in parquet).
    print("\n=== (Skipped) First row group read to avoid loading image columns ===")
    return parquet_file


def detect_state_action_keys(column_names: list[str]) -> tuple[list[str], list[str], str]:
    """Return (state_keys, action_keys, style)"""
    names = column_names

    # LIBERO-style: packed vectors
    if "state" in names and "actions" in names:
        return ["state"], ["actions"], "LIBERO"

    # GALAXEA-style: block columns
    has_obs = any(n.startswith("observation.state.") for n in names)
    has_act = any(n.startswith("action.") for n in names)
    if has_obs and has_act:
        # Use schema order for stable, reproducible packing
        state_keys = [n for n in names if n.startswith("observation.state.")]
        action_keys = [n for n in names if n.startswith("action.")]
        return state_keys, action_keys, "GALAXEA"

    raise RuntimeError(
        "Unrecognized schema: cannot find (state, actions) nor (observation.state.*, action.*).\n"
        "Columns are:\n" + "\n".join(names)
    )


def arr_at(col, i: int) -> np.ndarray:
    """
    Convert the i-th row of a pydict column into a flat 1D numpy array.
    Works for both:
      - list-of-lists (typical)
      - fixed-size list arrays
    """
    # after to_pydict(), col is usually a Python list with length = num_rows
    if isinstance(col, list):
        v = col[i]
    else:
        v = col
    return np.asarray(v).reshape(-1)


def fmt_vec(x: np.ndarray) -> str:
    """Pretty print small vectors without scientific noise."""
    x = np.asarray(x)
    with np.printoptions(precision=FLOAT_PREC, suppress=True):
        return np.array2string(x, separator=", ")


def print_openpi_spec(state_keys: list[str], action_keys: list[str], state_dim: int, action_dim: int) -> None:
    print("\n\n================ OpenPI Spec (copy/paste) ================")
    print("observation:")
    print("  state_keys:")
    for k in state_keys:
        print(f"    - {k}")
    print(f"  state_dim: {state_dim}")
    print("action:")
    print("  action_keys:")
    for k in action_keys:
        print(f"    - {k}")
    print(f"  action_dim: {action_dim}")
    print("==========================================================\n")


# =============================================================================
# Core inspect
# =============================================================================
def inspect_state_action_dims_and_preview(parquet_file: Path, preview_rows: int = 5) -> None:
    pf = pq.ParquetFile(parquet_file)
    names = pf.schema_arrow.names

    state_keys, action_keys, style = detect_state_action_keys(names)
    need_cols = state_keys + action_keys

    # Read only needed columns: avoids LIBERO image columns
    table = pf.read_row_group(0, columns=need_cols)
    num_rows = table.num_rows
    show_n = min(preview_rows, num_rows)

    row = table.slice(0, show_n).to_pydict()

    print(f"\n[Detect] Dataset style: {style}")
    print(f"[Dims] Using parquet for dim inspection: {parquet_file}")

    # --- Print dims for row0 (per-block) ---
    print("\n=== State blocks (dims from row 0) ===")
    for k in state_keys:
        v0 = arr_at(row[k], 0)
        print(f"{k}: {v0.shape}")

    print("\n=== Action blocks (dims from row 0) ===")
    for k in action_keys:
        v0 = arr_at(row[k], 0)
        print(f"{k}: {v0.shape}")

    # --- Packed dims (row0) ---
    state0 = np.concatenate([arr_at(row[k], 0) for k in state_keys], axis=0)
    action0 = np.concatenate([arr_at(row[k], 0) for k in action_keys], axis=0)
    print("\n=== Packed dimensions (row 0) ===")
    print("Packed state dim:", state0.shape)
    print("Packed action dim:", action0.shape)

    # --- Print OpenPI-ready spec snippet ---
    print_openpi_spec(state_keys, action_keys, int(state0.shape[-1]), int(action0.shape[-1]))

    # --- Preview first N rows actual values ---
    print(f"\n=== Preview first {show_n} rows: packed state/actions values ===")
    for i in range(show_n):
        si = np.concatenate([arr_at(row[k], i) for k in state_keys], axis=0)
        ai = np.concatenate([arr_at(row[k], i) for k in action_keys], axis=0)

        print(f"\n--- row {i} ---")
        print(f"state[{i}] (dim={si.shape[-1]}): {fmt_vec(si)}")
        print(f"actions[{i}] (dim={ai.shape[-1]}): {fmt_vec(ai)}")

    # --- If LIBERO and state_dim==8, help you infer semantics quickly ---
    if style == "LIBERO" and int(state0.shape[-1]) == 8:
        print("\n[Hint] LIBERO state_dim=8 is commonly either:")
        print("  A) 3(xyz) + 4(quat) + 1(gripper) = 8  (EE pose + gripper)")
        print("  B) 7(joints) + 1(gripper) = 8")
        print("Check the printed values: quat components typically lie in [-1, 1].")


# =============================================================================
# Main
# =============================================================================
def main():
    task_root = ROOT / TASK_NAME
    if not task_root.exists():
        raise FileNotFoundError(f"Task root does not exist: {task_root}")

    picked_parquet = print_parquet_columns(task_root)
    parquet_file = Path(PARQUET_PATH) if PARQUET_PATH is not None else picked_parquet

    inspect_state_action_dims_and_preview(parquet_file, preview_rows=PREVIEW_ROWS)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
from pathlib import Path
import argparse
import numpy as np
import pyarrow.parquet as pq

def main():
    _DATA_ROOT = os.environ.get("OPENPI_DATA_ROOT", "/data/datasets")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=f"{_DATA_ROOT}/galaxea_open_world_datasets/lerobot")
    ap.add_argument("--task", type=str, default="Dry_Clothes_In_A_Dryer")
    ap.add_argument("--col", type=str, default="action.chassis.velocities")
    ap.add_argument("--thresh", type=float, default=50.0)
    ap.add_argument("--max_files", type=int, default=0, help="0 means all files")
    args = ap.parse_args()

    task_dir = Path(args.root) / args.task / "data"
    files = sorted(task_dir.rglob("*.parquet"))
    print(f"[INFO] task_dir = {task_dir}")
    print(f"[INFO] parquet_files_found = {len(files)}")
    if len(files) == 0:
        raise FileNotFoundError(f"No parquet under {task_dir}")

    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]
        print(f"[INFO] limiting to first {len(files)} files")

    cols = [
        args.col,
        "episode_index", "frame_index", "index",
        "timestamp", "task_index", "quality_index",
        "coarse_task_index", "coarse_quality_index",
    ]

    bad = 0
    total_rows = 0
    total_rgs = 0
    max_abs = np.zeros(3, dtype=np.float64)
    max_where = [None, None, None]  # (fp, rg, row, value)

    # quick sanity: show schema of first file
    pf0 = pq.ParquetFile(files[0])
    schema_names = set(pf0.schema_arrow.names)
    print(f"[INFO] first_file = {files[0]}")
    print(f"[INFO] has_col({args.col}) = {args.col in schema_names}")
    if args.col not in schema_names:
        print("[WARN] Column not found in schema. Available columns (first 40):")
        print(list(pf0.schema_arrow.names)[:40])
        raise KeyError(f"Missing column {args.col} in parquet schema")

    for fi, fp in enumerate(files):
        pf = pq.ParquetFile(fp)
        total_rgs += pf.num_row_groups

        # only read columns that exist in this file
        present = set(pf.schema_arrow.names)
        use_cols = [c for c in cols if c in present]

        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=use_cols)
            d = table.to_pydict()

            vals = d.get(args.col, None)
            if vals is None:
                continue

            n = len(vals)
            total_rows += n

            # scan each row
            for i in range(n):
                v = vals[i]
                if v is None:
                    continue
                a = np.asarray(v, dtype=np.float32).reshape(-1)
                if a.size < 3:
                    continue

                # update global max tracker (3 dims)
                for k in range(3):
                    av = float(abs(a[k]))
                    if av > max_abs[k]:
                        max_abs[k] = av
                        max_where[k] = (str(fp), rg, i, float(a[k]))

                if abs(a[2]) > args.thresh:
                    bad += 1
                    def get_meta(name):
                        arr = d.get(name, None)
                        return None if arr is None else arr[i]

                    print("\n[BAD]", fp, "row_group", rg, "row", i)
                    print("  chassis:", a[:3].tolist())
                    print("  episode_index:", get_meta("episode_index"))
                    print("  frame_index:", get_meta("frame_index"))
                    print("  index:", get_meta("index"))
                    print("  timestamp:", get_meta("timestamp"))
                    print("  task_index:", get_meta("task_index"))
                    print("  quality_index:", get_meta("quality_index"))

        # progress every few files
        if (fi + 1) % 10 == 0 or (fi + 1) == len(files):
            print(f"[PROG] scanned_files={fi+1}/{len(files)}  total_rows={total_rows}  bad={bad}")

    print("\n[RESULT] Done.")
    print(f"[RESULT] scanned_files = {len(files)}")
    print(f"[RESULT] total_row_groups = {total_rgs}")
    print(f"[RESULT] total_rows = {total_rows}")
    print(f"[RESULT] bad_count (abs(chassis[2])>{args.thresh}) = {bad}")
    print(f"[RESULT] max_abs chassis = {max_abs.tolist()}")
    for k in range(3):
        print(f"[RESULT] max_where chassis[{k}] = {max_where[k]}")

if __name__ == "__main__":
    main()

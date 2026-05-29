"""
Scan dataset outputs (after the same transforms as training) for outliers in actions/state.

Outlier rules (per-sample):
  - any |z| > z_thresh  where z=(x-mean)/max(std, eps)
  OR
  - any |x| > abs_thresh (optional hard bound)

Prints: dataset __idx__, key, time step (if exists), dim, value, z
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import tqdm
import tyro

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class IndexDataset:
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        ex = self.ds[idx]
        if isinstance(ex, dict):
            ex["__idx__"] = np.int64(idx)
        return ex


def _as_np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _flatten_time(x: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Return (X, has_time) where
      - if x is (B, T, D) -> X=(B*T, D), has_time=True
      - if x is (B, D)    -> X=(B, D),   has_time=False
    """
    if x.ndim == 3:
        B, T, D = x.shape
        return x.reshape(B * T, D), True
    if x.ndim == 2:
        return x, False
    raise ValueError(f"Unexpected tensor shape: {x.shape}")


def main(
    config_name: str = "pi05_galaxea_r1lite",
    norm_path: str = "./assets/pi05_galaxea_r1lite/galaxea_r1lite/norm_stats.json",
    batch_size: int = 256,
    num_workers: int = 8,
    max_batches: int | None = 200,     # 快速扫描：先扫 200 个 batch
    z_thresh: float = 8.0,             # 经验上 6~10 之间
    abs_thresh: float | None = None,   # 例如动作本应[-1,1]可设 abs_thresh=5
    max_print: int = 50,
    eps: float = 1e-6,
):
    norm = json.load(open(norm_path, "r"))["norm_stats"]

    cfg = _config.get_config(config_name)
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)

    ds = _data_loader.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)
    ds = _data_loader.TransformedDataset(
        ds,
        [
            *data_cfg.repack_transforms.inputs,
            *data_cfg.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    ds = IndexDataset(ds)

    num_batches = max_batches if max_batches is not None else (len(ds) // batch_size)
    num_batches = max(1, int(num_batches))

    dl = _data_loader.TorchDataLoader(
        ds,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        num_batches=num_batches,
        framework="pytorch",
    )

    print(f"[INFO] config={config_name}")
    print(f"[INFO] norm={norm_path}")
    print(f"[INFO] batch_size={batch_size} num_workers={num_workers} num_batches={num_batches}")
    print(f"[INFO] z_thresh={z_thresh} abs_thresh={abs_thresh} max_print={max_print}")

    printed = 0

    for bi, batch in enumerate(tqdm.tqdm(dl, total=num_batches, desc="OUTLIER scan", dynamic_ncols=True)):
        idx = np.asarray(batch["__idx__"]).astype(np.int64)

        for key in ["actions", "state"]:
            x = _as_np(batch[key])  # (B,T,D) or (B,D)
            mean = np.asarray(norm[key]["mean"], dtype=np.float32)
            std  = np.asarray(norm[key]["std"],  dtype=np.float32)
            std = np.maximum(std, eps)

            X, has_time = _flatten_time(x)  # (N, D)
            Z = (X - mean[None, :]) / std[None, :]

            # rule1: z outlier
            mask_z = np.abs(Z) > z_thresh

            # rule2: abs outlier (optional)
            if abs_thresh is not None:
                mask_abs = np.abs(X) > abs_thresh
                mask = mask_z | mask_abs
            else:
                mask = mask_z

            if not np.any(mask):
                continue

            bad_pos = np.argwhere(mask)  # rows: [row, dim]
            for row, dim in bad_pos:
                row = int(row); dim = int(dim)
                # map back to dataset idx + timestep if has_time
                if has_time:
                    B = x.shape[0]
                    T = x.shape[1]
                    b = row // T
                    t = row % T
                    ds_idx = int(idx[b])
                else:
                    b = row
                    t = None
                    ds_idx = int(idx[b])

                val = float(X[row, dim])
                z   = float(Z[row, dim])
                print(f"[OUTLIER] key={key} ds_idx={ds_idx} t={t} dim={dim} val={val:.6g} z={z:.3f}")

                printed += 1
                if printed >= max_print:
                    print("[DONE] reached max_print, stop early.")
                    return

    print("[DONE] no outliers found under current thresholds.")


if __name__ == "__main__":
    tyro.cli(main)

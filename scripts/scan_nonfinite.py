"""
Scan dataset outputs (after the same transforms as training) for NaN/Inf in actions/state.

Writes progress + findings to stdout/stderr; you can redirect/tee to a log file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
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
    """Wraps a map-style dataset to also return __idx__ for debugging."""
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        ex = self.ds[idx]
        # attach index for later reverse lookup
        if isinstance(ex, dict):
            ex["__idx__"] = np.int64(idx)
        return ex


def _all_finite_per_sample(x: np.ndarray) -> np.ndarray:
    """Return bool mask (B,) indicating whether each sample is all-finite."""
    if x.ndim == 0:
        return np.isfinite(x)[None]
    if x.ndim == 1:
        # treat as (D,) => single sample
        return np.isfinite(x).all()[None]
    # (B, ...)
    axes = tuple(range(1, x.ndim))
    return np.isfinite(x).all(axis=axes)


def main(
    config_name: str = "pi05_galaxea_r1lite",
    batch_size: int = 256,
    num_workers: int = 8,
    max_frames: int | None = None,   # None = full scan; else only scan first max_frames samples
    max_bad: int = 50,
    print_every_sec: float = 15.0,
):
    cfg = _config.get_config(config_name)
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)

    # Build dataset (same transforms as your earlier code)
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

    n_total = len(ds)
    n_scan = min(n_total, max_frames) if (max_frames is not None) else n_total
    num_batches = n_scan // batch_size
    if num_batches <= 0:
        num_batches = 1

    dl = _data_loader.TorchDataLoader(
        ds,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        num_batches=num_batches,
        framework="pytorch",
    )

    print(f"[INFO] config_name={config_name}")
    print(f"[INFO] dataset_len={n_total}, scanning={n_scan} (max_frames={max_frames})")
    print(f"[INFO] batch_size={batch_size}, num_workers={num_workers}, num_batches={num_batches}")

    bad = 0
    last_print = time.time()
    seen = 0

    pbar = tqdm.tqdm(dl, total=num_batches, desc="NONFINITE scan", dynamic_ncols=True)
    for bi, batch in enumerate(pbar):
        # batch["__idx__"] should exist (B,)
        idx = np.asarray(batch.get("__idx__", None)) if isinstance(batch, dict) else None

        a = np.asarray(batch["actions"], np.float32)
        s = np.asarray(batch["state"], np.float32)

        fa = _all_finite_per_sample(a)
        fs = _all_finite_per_sample(s)

        # mark bad per-sample
        good = fa & fs
        if not np.all(good):
            bad_pos = np.where(~good)[0]
            for j in bad_pos:
                _idx = int(idx[j]) if idx is not None else f"batch{bi}_pos{int(j)}"
                aa = a[j]
                ss = s[j]
                print(
                    "[NONFINITE] idx=",
                    _idx,
                    " actions_finite=",
                    bool(np.isfinite(aa).all()),
                    " state_finite=",
                    bool(np.isfinite(ss).all()),
                    " a_maxabs=",
                    float(np.nanmax(np.abs(aa))) if aa.size else None,
                    " s_maxabs=",
                    float(np.nanmax(np.abs(ss))) if ss.size else None,
                )
                bad += 1
                if bad >= max_bad:
                    print(f"[DONE] reached max_bad={max_bad}, stopping early.")
                    print("done. nonfinite_count =", bad)
                    return

        seen += a.shape[0] if a.ndim >= 1 else 1
        now = time.time()
        if now - last_print >= print_every_sec:
            print(f"[PROG] batches={bi+1}/{num_batches} seen~={seen} bad={bad}")
            last_print = now

    print("done. nonfinite_count =", bad)


if __name__ == "__main__":
    tyro.cli(main)

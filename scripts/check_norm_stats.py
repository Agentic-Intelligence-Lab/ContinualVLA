#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check whether the dataloader exposes actions, and sanity-check actions distribution vs. norm_stats (q01/q99).

This version FIXES your case:
  batch == (Observation, ArrayImpl)  where ArrayImpl is a JAX array (often actions/targets).

Key features:
  - tqdm progress bar with ETA
  - small-sample scan via --max_batches
  - robust batch handling: batch can be dict OR tuple/list
  - robust action extraction:
      (1) if batch is (Observation, array_like) -> treat 2nd element as actions
      (2) dict keys: "actions", "action", "target_actions", ...
      (3) pack from per-block keys like "action.left_arm" etc.
  - prints first-batch structure if no actions found
"""

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import openpi.training.data_loader as _data_loader
import openpi.training.config as _train_config


# -----------------------------
# Galaxea packed action format
# -----------------------------
ACTION_SEQUENCE_KEYS = [
    "action.left_arm",            # (H,6)
    "action.right_arm",           # (H,6)
    "action.torso.velocities",    # (H,6)
    "action.chassis.velocities",  # (H,3)
    "action.left_gripper",        # (H,1) or (H,)
    "action.right_gripper",       # (H,1) or (H,)
]
ACTION_DIMS = {
    "action.left_arm": 6,
    "action.right_arm": 6,
    "action.torso.velocities": 6,
    "action.chassis.velocities": 3,
    "action.left_gripper": 1,
    "action.right_gripper": 1,
}
EXPECTED_ACTION_DIM = 23


def _is_torch_tensor(x: Any) -> bool:
    return torch is not None and torch.is_tensor(x)


def _as_numeric_ndarray(x: Any) -> Optional[np.ndarray]:
    """
    Try convert to a numeric numpy array (works for torch / jax / numpy / list).
    Return None if conversion fails or results in object dtype.
    """
    try:
        a = np.asarray(x)
    except Exception:
        return None
    if a.dtype == object:
        return None
    return a


def _to_numpy(x: Any) -> np.ndarray:
    """Robust conversion: torch.Tensor / jax array / numpy / list -> numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if _is_torch_tensor(x):
        return x.detach().cpu().numpy()
    a = _as_numeric_ndarray(x)
    if a is None:
        # last resort
        return np.asarray(x)
    return a


def _summarize_obj(x: Any, max_keys: int = 40) -> str:
    """Human-readable summary for debugging batch structure."""
    if isinstance(x, dict):
        keys = list(x.keys())
        show = keys[:max_keys]
        more = "" if len(keys) <= max_keys else f", ... (+{len(keys)-max_keys} more)"
        return f"dict(keys={show}{more})"
    if isinstance(x, (tuple, list)):
        return f"{type(x).__name__}(len={len(x)})"
    if _is_torch_tensor(x):
        return f"torch.Tensor(shape={tuple(x.shape)}, dtype={x.dtype})"
    if isinstance(x, np.ndarray):
        return f"np.ndarray(shape={x.shape}, dtype={x.dtype})"

    # JAX ArrayImpl (or any array-like)
    a = _as_numeric_ndarray(x)
    if a is not None and a.ndim >= 1:
        return f"{type(x).__name__}(array_like shape={a.shape}, dtype={a.dtype})"

    return f"{type(x).__name__}"


def _batch_debug_report(batch: Any) -> str:
    """Detailed report for first batch if needed."""
    lines: List[str] = []
    lines.append(f"batch_type: {type(batch).__name__}")

    if isinstance(batch, dict):
        lines.append(f"dict_keys({len(batch)}): {list(batch.keys())[:120]}")
        return "\n".join(lines)

    if isinstance(batch, (tuple, list)):
        lines.append(f"tuple/list len = {len(batch)}")
        for i, elem in enumerate(batch):
            lines.append(f"  [{i}] { _summarize_obj(elem) }")
            if isinstance(elem, dict):
                lines.append(f"      keys({len(elem)}): {list(elem.keys())[:120]}")
        return "\n".join(lines)

    lines.append(f"summary: {_summarize_obj(batch)}")
    return "\n".join(lines)


def _infer_horizon_from_blocks(batch_dict: Dict[str, Any]) -> int:
    """Infer horizon H from any available action block."""
    for k in ACTION_SEQUENCE_KEYS:
        if k in batch_dict:
            a = _to_numpy(batch_dict[k])
            if a.ndim == 2:
                return int(a.shape[0])
    for k in ("action.left_gripper", "action.right_gripper"):
        if k in batch_dict:
            a = _to_numpy(batch_dict[k])
            if a.ndim == 1 and a.shape[0] > 1:
                return int(a.shape[0])
    return 1


def _ensure_horizon_and_dim(a: np.ndarray, *, H: int, expected_dim: int, key: str) -> np.ndarray:
    """
    Ensure a block becomes shape (H, expected_dim).

    Accepts:
      - (H, expected_dim)
      - (1, expected_dim) -> broadcast to (H, expected_dim)
      - (expected_dim,)   -> treat as (1, expected_dim) then broadcast
      - (H,) and expected_dim==1 -> reshape to (H,1)
    """
    if a.ndim == 1:
        if expected_dim == 1 and a.shape[0] == H:
            a = a.reshape(H, 1)
        elif a.shape[0] == expected_dim:
            a = a.reshape(1, expected_dim)
        else:
            raise ValueError(
                f"Action key '{key}' got 1D shape={a.shape}. "
                f"Cannot interpret as (H,1) with H={H} nor (1,{expected_dim})."
            )

    if a.ndim != 2:
        raise ValueError(f"Action key '{key}' must be 1D or 2D, got shape={a.shape}")

    if a.shape[1] != expected_dim:
        raise ValueError(f"Action key '{key}' dim mismatch: got shape={a.shape}, expected last_dim={expected_dim}")

    if a.shape[0] == H:
        return a
    if a.shape[0] == 1:
        return np.repeat(a, repeats=H, axis=0)

    raise ValueError(
        f"Action horizon mismatch for '{key}': got H={a.shape[0]}, expected H={H} (or 1 for broadcast)."
    )


def _pack_actions_from_blocks(batch_dict: Dict[str, Any]) -> Optional[np.ndarray]:
    """If per-block action keys exist, pack them into (H,23). Return None if keys missing."""
    if not all(k in batch_dict for k in ACTION_SEQUENCE_KEYS):
        return None

    H = _infer_horizon_from_blocks(batch_dict)
    blocks = []
    for k in ACTION_SEQUENCE_KEYS:
        a = _to_numpy(batch_dict[k]).astype(np.float32, copy=False)
        a = _ensure_horizon_and_dim(a, H=H, expected_dim=ACTION_DIMS[k], key=k)
        blocks.append(a)

    actions = np.concatenate(blocks, axis=-1)  # (H,23)
    if actions.shape[-1] != EXPECTED_ACTION_DIM:
        raise RuntimeError(f"Packed action dim mismatch: got {actions.shape[-1]}, expected {EXPECTED_ACTION_DIM}")
    return actions


def _extract_actions_from_dict(batch_dict: Dict[str, Any]) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Try multiple conventions to find actions in a dict batch."""
    for key in ("actions", "action", "target_actions", "policy_actions"):
        if key in batch_dict:
            return key, _to_numpy(batch_dict[key])

    # Heuristic: any key containing "actions"
    candidates = []
    for k, v in batch_dict.items():
        if isinstance(k, str) and "actions" in k.lower():
            a = _as_numeric_ndarray(v)
            if a is not None and a.ndim >= 2:
                candidates.append(k)
    if candidates:
        k = sorted(candidates, key=len)[0]
        return k, _to_numpy(batch_dict[k])

    packed = _pack_actions_from_blocks(batch_dict)
    if packed is not None:
        return "packed_from_action_blocks", packed

    return None, None


def _looks_like_actions_array(x: Any) -> bool:
    """
    Heuristic for action arrays.
    Accept torch / numpy / jax (anything convertible to numeric np array).
    """
    a = _as_numeric_ndarray(x)
    if a is None:
        return False
    if a.ndim not in (2, 3):
        return False
    if a.shape[-1] < 10:
        return False
    return True


def _extract_actions(batch: Any) -> Tuple[Optional[str], Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Extract actions from batch which may be dict OR tuple/list.
    Also returns a "merged_dict" if we can build one (for key printing).
    """
    if isinstance(batch, dict):
        k, a = _extract_actions_from_dict(batch)
        return k, a, batch

    if isinstance(batch, (tuple, list)):
        # Special-case: (Observation, ArrayImpl) -> treat second as actions if it looks like an action array
        if len(batch) == 2 and _looks_like_actions_array(batch[1]):
            return "tuple[1]_as_actions", _to_numpy(batch[1]), None

        # If any element is a dict, try extracting from merged dicts
        dict_elems = [e for e in batch if isinstance(e, dict)]
        merged: Dict[str, Any] = {}
        for d in dict_elems:
            merged.update(d)

        if merged:
            k, a = _extract_actions_from_dict(merged)
            if a is not None:
                return f"merged_dict::{k}", a, merged

        # Try each element directly as an array-like
        for i, e in enumerate(batch):
            if _looks_like_actions_array(e):
                return f"tuple[{i}]", _to_numpy(e), merged if merged else None

        return None, None, merged if merged else None

    return None, None, None


def _load_train_config_by_name(name: str):
    """Be robust to different OpenPI config APIs."""
    if hasattr(_train_config, "get_config"):
        return _train_config.get_config(name)
    if hasattr(_train_config, "load_config"):
        return _train_config.load_config(name)
    if hasattr(_train_config, "_CONFIGS") and name in _train_config._CONFIGS:
        return _train_config._CONFIGS[name]
    raise RuntimeError(f"Cannot load config '{name}'. Please check openpi.training.config APIs / _CONFIGS.")


def _create_loader(cfg, max_batches: int):
    """Create dataloader with best-effort kwargs matching current OpenPI version."""
    fn = _data_loader.create_data_loader
    sig = inspect.signature(fn)
    params = sig.parameters

    kwargs = {}
    if "skip_norm_stats" in params:
        kwargs["skip_norm_stats"] = True  # only for debug scanning
    if "num_batches" in params:
        kwargs["num_batches"] = max_batches
    if "is_training" in params:
        kwargs["is_training"] = True
    if "shuffle" in params:
        kwargs["shuffle"] = False
    if "num_workers" in params:
        kwargs["num_workers"] = 0

    return fn(cfg, **kwargs)


def _percentile(x: np.ndarray, p: float) -> float:
    return float(np.percentile(x, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_name", type=str, required=True, help="TrainConfig name, e.g. pi05_galaxea_r1lite")
    ap.add_argument("--max_batches", type=int, default=3, help="Number of batches to scan (small sample)")
    ap.add_argument("--norm_stats_path", type=str, default=None, help="Path to norm_stats.json (optional)")
    args = ap.parse_args()

    cfg = _load_train_config_by_name(args.config_name)

    # Load norm_stats if provided
    norm_stats = None
    if args.norm_stats_path is not None:
        p = Path(args.norm_stats_path)
        raw = json.loads(p.read_text())
        norm_stats = raw["norm_stats"] if "norm_stats" in raw else raw

    loader = _create_loader(cfg, args.max_batches)

    all_actions = []
    first_batch_report = None
    missing_actions_batches = 0

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, total=args.max_batches, desc="Scanning batches", dynamic_ncols=True)

    for bi, batch in enumerate(iterator):
        if bi >= args.max_batches:
            break

        if first_batch_report is None:
            first_batch_report = _batch_debug_report(batch)

        key, actions, _ = _extract_actions(batch)

        if actions is None:
            missing_actions_batches += 1
            if tqdm is not None:
                iterator.set_postfix(found_actions="NO")
            continue

        a = _to_numpy(actions)

        # Normalize shape:
        #   (B,H,A) -> (B*H,A)
        #   (H,A)   -> (H,A)
        #   (B,A)   -> (B,A)
        if a.ndim == 3:
            a2 = a.reshape(-1, a.shape[-1])
        elif a.ndim == 2:
            a2 = a
        else:
            raise RuntimeError(f"Unexpected action array shape from key='{key}': {a.shape}")

        all_actions.append(a2)

        if tqdm is not None:
            iterator.set_postfix(found_actions="YES", key=str(key), nsteps=sum(x.shape[0] for x in all_actions))

    if len(all_actions) == 0:
        print("\n[ERROR] No actions collected.")
        print("This means your dataloader batches do NOT expose actions at this stage.")
        print("\n[DEBUG] First batch structure:")
        print(first_batch_report)
        print("\n[DEBUG] Batches scanned:", args.max_batches, " | batches without actions:", missing_actions_batches)
        return

    A = np.concatenate(all_actions, axis=0)  # (N, action_dim)
    print(f"\nCollected actions: shape={A.shape} (flattened steps, action_dim)")

    action_dim = A.shape[-1]
    eps_zero = 1e-8

    print("\nPer-dimension sample stats (small-sample scan):")
    for j in range(action_dim):
        col = A[:, j]
        zr = float(np.mean(np.abs(col) < eps_zero))
        mn = float(np.min(col))
        mx = float(np.max(col))
        p01 = _percentile(col, 1.0)
        p99 = _percentile(col, 99.0)
        print(f"dim {j:02d}: zero_ratio={zr:7.4f}  min={mn: .6g}  max={mx: .6g}  p01={p01: .6g}  p99={p99: .6g}")

    # Compare with provided norm_stats (if any)
    if norm_stats is not None and "actions" in norm_stats:
        ns = norm_stats["actions"]
        q01 = np.asarray(ns.get("q01", []), dtype=np.float64)
        q99 = np.asarray(ns.get("q99", []), dtype=np.float64)
        std = np.asarray(ns.get("std", []), dtype=np.float64)

        print("\nNorm-stats sanity:")
        if std.size == action_dim:
            print(f"  std==0 dims: {np.where(std == 0)[0].tolist()}")
            print(f"  std<1e-6 dims: {np.where(std < 1e-6)[0].tolist()}")

        if q01.size == action_dim and q99.size == action_dim:
            gap = q99 - q01
            tiny = np.where(gap < 1e-4)[0].tolist()
            print(f"  q99-q01 min_gap={float(gap.min()):.6g}  median_gap={float(np.median(gap)):.6g}")
            print(f"  tiny_gap_dims(<1e-4): {tiny}")
            if tiny:
                print("\n[WARN] Extremely small q99-q01 dims (quantile-norm may amplify noise here):")
                for i in tiny[:80]:
                    print(f"  dim {i:02d}: q01={float(q01[i]):.6g}  q99={float(q99[i]):.6g}  gap={float(gap[i]):.6g}")
        else:
            print("\n[WARN] norm_stats['actions'] q01/q99 dim mismatch, cannot compare reliably.")
            print(f"  norm q01 dim={q01.size}, norm q99 dim={q99.size}, observed action_dim={action_dim}")


if __name__ == "__main__":
    main()

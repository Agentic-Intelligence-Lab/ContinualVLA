"""Compute normalization statistics using a fast parquet path when supported.

This script accelerates norm-stats computation for supported local LeRobot ALOHA-style
datasets by reading only the required parquet columns ("state" / "action") and replaying
the numeric transforms that affect those tensors. Unsupported configs can optionally fall
back to the legacy `compute_norm_stats.py --force-slow` path.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
import tyro

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import openpi.policies.aloha_policy as aloha_policy
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


class UnsupportedFastPathError(RuntimeError):
    """Raised when a config cannot be handled by the fast path."""


def _extract_columns(data_config: _config.DataConfig) -> tuple[str, str]:
    repack_inputs = data_config.repack_transforms.inputs
    if len(repack_inputs) != 1 or not isinstance(repack_inputs[0], _transforms.RepackTransform):
        raise UnsupportedFastPathError("Fast path requires a single RepackTransform")

    structure = repack_inputs[0].structure
    state_col = structure.get("state")
    action_col = structure.get("actions")
    if not isinstance(state_col, str) or not isinstance(action_col, str):
        raise UnsupportedFastPathError("Fast path requires direct string mappings for state/actions")
    return state_col, action_col


def _supports_fast_path(
    data_config: _config.DataConfig,
    *,
    max_frames: int | None,
) -> None:
    if data_config.rlds_data_dir is not None:
        raise UnsupportedFastPathError("RLDS datasets are not supported by the fast path")
    if not getattr(data_config, "local_roots", ()):
        raise UnsupportedFastPathError("Fast path currently supports only local LeRobot datasets")
    if max_frames is not None:
        raise UnsupportedFastPathError("Fast path does not support max_frames sampling")
    if tuple(data_config.action_sequence_keys) != ("action",):
        raise UnsupportedFastPathError("Fast path currently supports action_sequence_keys=('action',) only")

    for transform in data_config.data_transforms.inputs:
        if isinstance(transform, aloha_policy.AlohaInputs):
            if transform.adapt_to_pi:
                raise UnsupportedFastPathError("Fast path does not support adapt_to_pi=True yet")
            continue
        if isinstance(transform, _transforms.DeltaActions | _transforms.ClampToValues):
            continue
        raise UnsupportedFastPathError(f"Unsupported input transform for fast path: {type(transform).__name__}")


def _iter_local_parquet_files(local_roots: tuple[str, ...] | list[str]) -> list[Path]:
    parquet_files: list[Path] = []
    for root in local_roots:
        sanitized_root = _data_loader._sanitize_local_lerobot_root(root)  # noqa: SLF001
        if sanitized_root is None:
            continue
        parquet_files.extend(sorted((Path(sanitized_root) / "data").rglob("*.parquet")))
    return parquet_files


def _build_action_chunks(actions: np.ndarray, horizon: int) -> np.ndarray:
    if actions.ndim == 3:
        if actions.shape[1] != horizon:
            raise UnsupportedFastPathError(
                f"Pre-chunked actions have horizon {actions.shape[1]}, expected {horizon}"
            )
        return actions
    if actions.ndim != 2:
        raise UnsupportedFastPathError(f"Unsupported action array rank: {actions.ndim}")

    num_frames = actions.shape[0]
    if num_frames == 0:
        raise UnsupportedFastPathError("Encountered empty episode while building action chunks")
    query_indices = np.arange(num_frames)[:, None] + np.arange(horizon)[None, :]
    query_indices = np.clip(query_indices, 0, num_frames - 1)
    return actions[query_indices]


def _apply_fast_transforms(
    states: np.ndarray,
    action_chunks: np.ndarray,
    data_config: _config.DataConfig,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.array(states, copy=True)
    action_chunks = np.array(action_chunks, copy=True)

    for transform in data_config.data_transforms.inputs:
        if isinstance(transform, aloha_policy.AlohaInputs):
            # For adapt_to_pi=False, AlohaInputs leaves state/actions numerically unchanged.
            continue

        if isinstance(transform, _transforms.DeltaActions):
            if transform.mask is None:
                continue
            mask = np.asarray(transform.mask, dtype=bool)
            dims = mask.shape[-1]
            state_slice = states[:, :dims]
            action_chunks[:, :, :dims] -= np.expand_dims(np.where(mask, state_slice, 0), axis=1)
            continue

        if isinstance(transform, _transforms.ClampToValues):
            if transform.state_dims:
                dims = tuple(transform.state_dims)
                values = np.asarray(transform.state_values, dtype=states.dtype)
                states[:, dims] = values.astype(states.dtype, copy=False)
            if transform.action_dims:
                dims = tuple(transform.action_dims)
                values = np.asarray(transform.action_values, dtype=action_chunks.dtype)
                action_chunks[:, :, dims] = values.astype(action_chunks.dtype, copy=False)
            continue

        raise UnsupportedFastPathError(f"Unsupported input transform for fast path: {type(transform).__name__}")

    return states, action_chunks


def _episode_arrays(
    parquet_path: Path,
    state_col: str,
    action_col: str,
    action_horizon: int,
    data_config: _config.DataConfig,
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet_path, columns=[state_col, action_col])
    states = np.stack(df[state_col].to_numpy()).astype(np.float32)
    actions = np.stack(df[action_col].to_numpy()).astype(np.float32)
    action_chunks = _build_action_chunks(actions, action_horizon).astype(np.float32, copy=False)
    return _apply_fast_transforms(states, action_chunks, data_config)


def _yield_batched_arrays(
    parquet_files: list[Path],
    state_col: str,
    action_col: str,
    action_horizon: int,
    data_config: _config.DataConfig,
    batch_size: int,
):
    state_buffer: list[np.ndarray] = []
    action_buffer: list[np.ndarray] = []
    buffered = 0

    for parquet_path in parquet_files:
        states, actions = _episode_arrays(parquet_path, state_col, action_col, action_horizon, data_config)
        cursor = 0
        while cursor < len(states):
            take = min(batch_size - buffered, len(states) - cursor)
            state_buffer.append(states[cursor : cursor + take])
            action_buffer.append(actions[cursor : cursor + take])
            buffered += take
            cursor += take

            if buffered == batch_size:
                yield np.concatenate(state_buffer, axis=0), np.concatenate(action_buffer, axis=0)
                state_buffer.clear()
                action_buffer.clear()
                buffered = 0

    # Match TorchDataLoader(drop_last=True) used by the slow path.


def _fast_main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    _supports_fast_path(data_config, max_frames=max_frames)

    state_col, action_col = _extract_columns(data_config)
    parquet_files = _iter_local_parquet_files(list(data_config.local_roots))
    if not parquet_files:
        raise FileNotFoundError("No parquet files found under local_roots/data")

    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    batch_iter = _yield_batched_arrays(
        parquet_files,
        state_col,
        action_col,
        config.model.action_horizon,
        data_config,
        config.batch_size,
    )

    for state_batch, action_batch in tqdm.tqdm(batch_iter, desc="Computing stats (fast)", total=None):
        stats["state"].update(state_batch)
        stats["actions"].update(action_batch)

    norm_stats = {key: stat.get_statistics() for key, stat in stats.items()}

    asset_id = data_config.asset_id or data_config.repo_id
    if asset_id is None:
        raise ValueError("Data config must define either asset_id or repo_id to save norm stats.")

    output_path = config.assets_dirs / asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


def main(config_name: str, max_frames: int | None = None, fallback_to_slow: bool = True):
    try:
        _fast_main(config_name, max_frames=max_frames)
    except UnsupportedFastPathError as exc:
        if not fallback_to_slow:
            raise

        print(f"Fast path unsupported for this config: {exc}")
        print("Falling back to scripts/compute_norm_stats.py --force-slow ...")
        command = [
            sys.executable,
            str(Path(__file__).with_name("compute_norm_stats.py")),
            "--config-name",
            config_name,
            "--force-slow",
        ]
        if max_frames is not None:
            command.extend(["--max-frames", str(max_frames)])
        subprocess.run(command, check=True)


if __name__ == "__main__":
    tyro.cli(main)

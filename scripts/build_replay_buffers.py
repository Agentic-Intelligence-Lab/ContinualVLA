#!/usr/bin/env python3
"""CLI helper to build and downsample replay buffers for continual learning.

Called after each task finishes training. For the just-finished task, it creates
a new buffer. For all existing old-task buffers, it downsamples them (randomly
discards entries from the current buffer, does NOT re-sample from original data).

Usage:
    python scripts/build_replay_buffers.py \
        --config-name pi05_piper_stack_bowls_20260413 \
        --data-root /path/to/dataset \
        --buffer-dir ./replay_buffers \
        --buffer-size 0.2 \
        --num-total-tasks 2 \
        --mode episode \
        --seed 42
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on path
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Set JAX to CPU to avoid CUDA issues during data processing
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.training.replay_buffer import ReplayBufferDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Build/downsample replay buffers for continual learning")
    parser.add_argument("--config-name", type=str, required=True,
                        help="OpenPI config name for the just-finished task")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override data root directory (optional, uses config if not set)")
    parser.add_argument("--buffer-dir", type=str, required=True,
                        help="Directory to save/load buffer JSON files")
    parser.add_argument("--buffer-size", type=float, required=True,
                        help="Total buffer size fraction across all tasks (0-1)")
    parser.add_argument("--num-total-tasks", type=int, required=True,
                        help="Total number of tasks completed so far (including current)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="transition",
                        choices=["transition", "episode"],
                        help="Replay mode: 'transition' for individual steps, 'episode' for whole episodes")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name override for config resolution")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    num_tasks = args.num_total_tasks
    # All buffers (old and new) share the budget equally among all tasks
    per_task_fraction = args.buffer_size / num_tasks

    print(f"=== Replay Buffer Builder ===")
    print(f"  config_name: {args.config_name}")
    print(f"  buffer_size (total): {args.buffer_size}")
    print(f"  num_total_tasks: {num_tasks}")
    print(f"  per_task_fraction: {per_task_fraction:.6f}")
    print(f"  buffer_dir: {args.buffer_dir}")
    print(f"  mode: {args.mode}")

    os.makedirs(args.buffer_dir, exist_ok=True)

    # Load config
    config = _config.get_config(args.config_name)
    if args.exp_name:
        import dataclasses
        config = dataclasses.replace(config, exp_name=args.exp_name)

    # Create DataConfig from the config's data factory
    data_config = config.data.create(config.assets_dirs, config.model)

    # Step 1: Downsample all existing buffer files
    existing_buffers = sorted(glob.glob(os.path.join(args.buffer_dir, "task*_buffer.json")))
    print(f"\nFound {len(existing_buffers)} existing buffer(s)")

    for buf_path in existing_buffers:
        with open(buf_path, "r") as f:
            meta = json.load(f)

        buf_mode = meta.get("mode", "transition")

        if buf_mode == "episode":
            # Episode mode: downsample at episode level
            episode_indices = meta["episode_indices"]
            episode_to_steps = meta.get("episode_to_steps", {})
            original_num_episodes = meta.get("original_num_episodes", len(episode_indices))

            if isinstance(original_num_episodes, int):
                target_episode_count = max(1, int(original_num_episodes * per_task_fraction))
            else:
                target_episode_count = max(1, int(len(episode_indices) * per_task_fraction))

            old_step_count = len(meta.get("buffer_indices", []))
            new_episode_indices, new_buffer_indices, new_episode_to_steps = ReplayBufferDataset.downsample_episodes(
                episode_indices, target_episode_count, episode_to_steps, seed=args.seed
            )

            meta["episode_indices"] = new_episode_indices
            meta["buffer_indices"] = new_buffer_indices
            meta["episode_to_steps"] = {str(k): v for k, v in new_episode_to_steps.items()}
            meta["per_task_fraction"] = per_task_fraction
            meta["num_total_tasks"] = num_tasks

            with open(buf_path, "w") as f:
                json.dump(meta, f)

            print(f"  Downsampled {os.path.basename(buf_path)} (episode): "
                  f"{len(episode_indices)} -> {len(new_episode_indices)} episodes, "
                  f"{old_step_count} -> {len(new_buffer_indices)} steps "
                  f"(target_episodes={target_episode_count})")
        else:
            # Transition mode: downsample at step level
            old_indices = meta["buffer_indices"]
            original_size = meta.get("original_dataset_size", len(old_indices))

            if isinstance(original_size, int):
                target_count = max(1, int(original_size * per_task_fraction))
            else:
                target_count = max(1, int(len(old_indices) * per_task_fraction))

            new_indices = ReplayBufferDataset.downsample_to_count(old_indices, target_count, seed=args.seed)

            meta["buffer_indices"] = new_indices
            meta["per_task_fraction"] = per_task_fraction
            meta["num_total_tasks"] = num_tasks

            with open(buf_path, "w") as f:
                json.dump(meta, f)

            print(f"  Downsampled {os.path.basename(buf_path)}: {len(old_indices)} -> {len(new_indices)} steps "
                  f"(target={target_count}, original_size={original_size})")

    # Step 2: Build a new buffer for the just-finished task
    print(f"\nBuilding buffer for task: {args.config_name}")

    # Build the dataset using the same pipeline as training
    base_dataset = _data.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    transformed = _data.transform_dataset(base_dataset, data_config)

    # For episode mode, we need the underlying LeRobotDataset's episode_data_index
    episode_data_index = None
    if args.mode == "episode":
        # The base_dataset may be a TransformedDataset wrapping a LeRobotDataset
        # or a ConcatDataset. We need to find the underlying LeRobotDataset.
        ds = base_dataset
        # Unwrap TransformedDataset
        while hasattr(ds, "_dataset"):
            ds = ds._dataset
        # Now ds should be LeRobotDataset or ConcatDataset
        if hasattr(ds, "episode_data_index"):
            episode_data_index = ds.episode_data_index
        elif hasattr(ds, "datasets") and len(ds.datasets) > 0:
            # ConcatDataset: use first sub-dataset's episode_data_index
            episode_data_index = ds.datasets[0].episode_data_index
        else:
            print("WARNING: Cannot find episode_data_index for episode mode, falling back to transition mode")
            args.mode = "transition"

    # Determine dataset name from config
    dataset_name = args.config_name

    # Build the buffer
    buffer = ReplayBufferDataset.build_buffer(
        transformed,
        per_task_fraction,
        episode_data_index=episode_data_index,
        seed=args.seed,
        mode=args.mode,
        dataset_name=dataset_name,
    )

    # Save the buffer (pass original_num_episodes for correct downsampling later)
    save_kwargs = {}
    if args.mode == "episode" and episode_data_index is not None:
        save_kwargs["original_num_episodes"] = len(episode_data_index["from"])

    buf_path = os.path.join(args.buffer_dir, f"task{num_tasks}_buffer.json")
    buffer.save(buf_path, **save_kwargs)
    print(f"  Created task{num_tasks}_buffer.json: {len(buffer)} steps (mode={buffer.mode})")

    # Print summary
    all_buffers = sorted(glob.glob(os.path.join(args.buffer_dir, "task*_buffer.json")))
    total_steps = 0
    for bp in all_buffers:
        with open(bp, "r") as f:
            meta = json.load(f)
        total_steps += len(meta["buffer_indices"])
    print(f"\nTotal buffer: {total_steps} steps across {len(all_buffers)} file(s)")


if __name__ == "__main__":
    main()

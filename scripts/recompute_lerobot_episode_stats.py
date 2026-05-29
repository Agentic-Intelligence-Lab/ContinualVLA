#!/usr/bin/env python3
"""Recompute a standard LeRobot v2.1 `meta/episodes_stats.jsonl` from parquet + videos.

This is useful for datasets where `episodes_stats.jsonl` is missing or was synthesized
from global stats rather than computed per episode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import datasets
import numpy as np
from tqdm import tqdm

from lerobot.common.datasets.compute_stats import get_feature_stats, sample_indices
from lerobot.common.datasets.utils import serialize_dict
from lerobot.common.datasets.video_utils import decode_video_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root of the local LeRobot dataset")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to <dataset-root>/meta/episodes_stats.jsonl",
    )
    parser.add_argument("--episode-start", type=int, default=0, help="Inclusive start episode index")
    parser.add_argument(
        "--episode-stop",
        type=int,
        default=None,
        help="Exclusive stop episode index. Defaults to total_episodes",
    )
    parser.add_argument("--video-backend", type=str, default="pyav", help="Video backend for decoding")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output path if it already exists",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_episodes(path: Path) -> list[dict]:
    episodes = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def load_episode_table(parquet_path: Path) -> datasets.Dataset:
    return datasets.load_dataset("parquet", data_files=str(parquet_path), split="train")


def compute_video_stats(video_path: Path, num_frames: int, fps: int, backend: str) -> dict[str, np.ndarray]:
    sampled = sample_indices(num_frames)
    timestamps = [idx / fps for idx in sampled]
    frames = decode_video_frames(video_path, timestamps, tolerance_s=1.0 / fps + 1e-6, backend=backend)
    frames_np = frames.cpu().numpy()
    stats = get_feature_stats(frames_np, axis=(0, 2, 3), keepdims=True)
    return {key: value if key == "count" else np.squeeze(value, axis=0) for key, value in stats.items()}


def compute_tabular_stats(table: datasets.Dataset, key: str) -> dict[str, np.ndarray]:
    values = np.asarray(table[key])
    return get_feature_stats(values, axis=0, keepdims=values.ndim == 1)


def compute_episode_stats(root: Path, info: dict, episode_index: int, backend: str) -> dict[str, dict[str, np.ndarray]]:
    episode_chunk = episode_index // info["chunks_size"]
    data_relpath = info["data_path"].format(episode_chunk=episode_chunk, episode_index=episode_index)
    parquet_path = root / data_relpath
    table = load_episode_table(parquet_path)
    num_frames = len(table)

    ep_stats: dict[str, dict[str, np.ndarray]] = {}
    for key, feature in info["features"].items():
        dtype = feature["dtype"]
        if dtype == "string":
            continue
        if dtype in {"image", "video"}:
            video_relpath = info["video_path"].format(
                episode_chunk=episode_chunk,
                video_key=key,
                episode_index=episode_index,
            )
            ep_stats[key] = compute_video_stats(root / video_relpath, num_frames, info["fps"], backend)
        else:
            ep_stats[key] = compute_tabular_stats(table, key)

    return ep_stats


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    meta_dir = dataset_root / "meta"
    info = load_json(meta_dir / "info.json")
    episodes = load_episodes(meta_dir / "episodes.jsonl")

    start = args.episode_start
    stop = args.episode_stop if args.episode_stop is not None else info["total_episodes"]
    selected = [ep for ep in episodes if start <= ep["episode_index"] < stop]

    output_path = args.output_path or (meta_dir / "episodes_stats.jsonl")
    output_path = output_path.resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with tmp_path.open("w") as f:
        for episode in tqdm(selected, desc="Recomputing episode stats"):
            episode_index = episode["episode_index"]
            ep_stats = compute_episode_stats(dataset_root, info, episode_index, args.video_backend)
            record = {"episode_index": episode_index, "stats": serialize_dict(ep_stats)}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    tmp_path.replace(output_path)
    print(f"Wrote {len(selected)} episode stats to {output_path}")


if __name__ == "__main__":
    main()

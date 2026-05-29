#!/usr/bin/env python3
"""Convert RoboChallenge arrange_flowers raw data to local LeRobot v2.1 format."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import datasets
import numpy as np
from tqdm import tqdm

from lerobot.common.datasets.compute_stats import get_feature_stats, sample_indices
from lerobot.common.datasets.utils import serialize_dict
from lerobot.common.datasets.video_utils import decode_video_frames, get_video_info


FPS = 30
CHUNK_SIZE = 1000
VIDEO_MAP = {
    "global_realsense_rgb.mp4": "observation.images.cam_high",
    "arm_realsense_rgb.mp4": "observation.images.cam_left_wrist",
    "right_realsense_rgb.mp4": "observation.images.cam_right_wrist",
}
STATE_NAMES = [f"joint_{i}" for i in range(6)] + ["gripper_width"]


def parse_args() -> argparse.Namespace:
    _DATA_ROOT = os.environ.get("OPENPI_DATA_ROOT", "/data/datasets")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(_DATA_ROOT) / "RobochallengeDataset/arrange_flowers",
        help="Raw RoboChallenge dataset root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(_DATA_ROOT) / "RobochallengeDataset/lerobot_dataset/arrange_flowers",
        help="Output LeRobot dataset root",
    )
    parser.add_argument(
        "--episode-stop",
        type=int,
        default=None,
        help="Exclusive stop episode index for testing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the existing output directory before converting",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="pyav",
        help="Backend to use for decoding video stats",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def get_chunk_index(episode_index: int) -> int:
    return episode_index // CHUNK_SIZE


def make_video_stats(video_path: Path, num_frames: int, backend: str) -> dict[str, np.ndarray]:
    sampled_indices = sample_indices(num_frames)
    sampled_timestamps = [idx / FPS for idx in sampled_indices]
    frames = decode_video_frames(video_path, sampled_timestamps, tolerance_s=1.0 / FPS + 1e-6, backend=backend)
    frames_np = frames.cpu().numpy()
    stats = get_feature_stats(frames_np, axis=(0, 2, 3), keepdims=True)
    # Match LeRobot's expected video stat shape: (3, 1, 1) for non-count entries.
    return {key: value if key == "count" else np.squeeze(value, axis=0) for key, value in stats.items()}


def build_episode_arrays(state_rows: list[dict], episode_index: int, global_offset: int) -> dict[str, list]:
    num_frames = len(state_rows)
    states = np.asarray(
        [row["joint_positions"] + [row["gripper_width"]] for row in state_rows],
        dtype=np.float32,
    )
    frame_index = np.arange(num_frames, dtype=np.int64)
    timestamps = frame_index.astype(np.float32) / FPS
    task_index = np.zeros(num_frames, dtype=np.int64)
    episode_indices = np.full(num_frames, episode_index, dtype=np.int64)
    indices = np.arange(global_offset, global_offset + num_frames, dtype=np.int64)

    return {
        "observation.state": states.tolist(),
        "action": states.tolist(),
        "timestamp": timestamps.tolist(),
        "frame_index": frame_index.tolist(),
        "episode_index": episode_indices.tolist(),
        "index": indices.tolist(),
        "task_index": task_index.tolist(),
    }


def build_episode_stats(episode_arrays: dict[str, list], video_paths: dict[str, Path], backend: str) -> dict[str, dict]:
    num_frames = len(episode_arrays["frame_index"])
    stats = {}
    for key, values in episode_arrays.items():
        array = np.asarray(values)
        stats[key] = get_feature_stats(array, axis=0, keepdims=array.ndim == 1)
    for feature_key, video_path in video_paths.items():
        stats[feature_key] = make_video_stats(video_path, num_frames, backend)
    return stats


def episode_features() -> datasets.Features:
    return datasets.Features(
        {
            "observation.state": datasets.Sequence(datasets.Value("float32"), length=len(STATE_NAMES)),
            "action": datasets.Sequence(datasets.Value("float32"), length=len(STATE_NAMES)),
            "timestamp": datasets.Value("float32"),
            "frame_index": datasets.Value("int64"),
            "episode_index": datasets.Value("int64"),
            "index": datasets.Value("int64"),
            "task_index": datasets.Value("int64"),
        }
    )


def build_info(output_root: Path, total_episodes: int, total_frames: int, video_infos: dict[str, dict], max_episode_index: int) -> dict:
    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(STATE_NAMES)],
            "names": [STATE_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": [len(STATE_NAMES)],
            "names": [STATE_NAMES],
        },
    }
    for feature_key, video_info in video_infos.items():
        features[feature_key] = {
            "dtype": "video",
            "shape": [video_info["video.channels"], video_info["video.height"], video_info["video.width"]],
            "names": ["channels", "height", "width"],
            "info": video_info,
        }

    for key, dtype in {
        "timestamp": "float32",
        "frame_index": "int64",
        "episode_index": "int64",
        "index": "int64",
        "task_index": "int64",
    }.items():
        features[key] = {"dtype": dtype, "shape": [1], "names": None}

    total_chunks = get_chunk_index(max_episode_index) + 1 if total_episodes > 0 else 0
    return {
        "codebase_version": "v2.1",
        "robot_type": "arx5",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * len(VIDEO_MAP),
        "total_chunks": total_chunks,
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    task_desc = load_json(raw_root / "task_desc.json")
    prompt = task_desc["prompt"]
    raw_episodes = sorted(p for p in (raw_root / "data").iterdir() if p.is_dir())

    if args.episode_stop is not None:
        raw_episodes = [p for p in raw_episodes if int(p.name.split("_")[-1]) < args.episode_stop]

    feature_schema = episode_features()
    episode_rows = []
    episode_stats_rows = []
    global_offset = 0
    video_infos: dict[str, dict] = {}
    max_episode_index = -1

    for raw_episode in tqdm(raw_episodes, desc="Converting arrange_flowers"):
        episode_index = int(raw_episode.name.split("_")[-1])
        max_episode_index = max(max_episode_index, episode_index)
        state_rows = load_jsonl(raw_episode / "states" / "states.jsonl")
        episode_arrays = build_episode_arrays(state_rows, episode_index, global_offset)
        num_frames = len(state_rows)

        chunk_index = get_chunk_index(episode_index)
        parquet_path = output_root / f"data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        dataset = datasets.Dataset.from_dict(episode_arrays, features=feature_schema)
        dataset.to_parquet(str(parquet_path))

        linked_videos: dict[str, Path] = {}
        for raw_name, feature_key in VIDEO_MAP.items():
            src = raw_episode / "videos" / raw_name
            dst = output_root / f"videos/chunk-{chunk_index:03d}/{feature_key}/episode_{episode_index:06d}.mp4"
            ensure_link(src, dst)
            linked_videos[feature_key] = src
            if feature_key not in video_infos:
                video_infos[feature_key] = get_video_info(src)

        episode_stats = build_episode_stats(episode_arrays, linked_videos, args.video_backend)
        episode_stats_rows.append({"episode_index": episode_index, "stats": serialize_dict(episode_stats)})
        episode_rows.append({"episode_index": episode_index, "tasks": [prompt], "length": num_frames})
        global_offset += num_frames

    tasks_rows = [{"task_index": 0, "task": prompt}]
    info = build_info(output_root, len(episode_rows), global_offset, video_infos, max_episode_index)

    write_json(output_root / "meta" / "info.json", info)
    write_jsonl(output_root / "meta" / "tasks.jsonl", tasks_rows)
    write_jsonl(output_root / "meta" / "episodes.jsonl", episode_rows)
    write_jsonl(output_root / "meta" / "episodes_stats.jsonl", episode_stats_rows)

    print(f"Converted {len(episode_rows)} episodes to {output_root}")


if __name__ == "__main__":
    main()

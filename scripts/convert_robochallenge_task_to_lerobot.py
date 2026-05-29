#!/usr/bin/env python3
"""Convert a RoboChallenge raw task directory to local LeRobot v2.1 format."""

from __future__ import annotations

import argparse
import dataclasses
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
RAW_VIDEO_TO_FEATURE = {
    "cam_high_rgb.mp4": "observation.images.cam_high",
    "cam_wrist_left_rgb.mp4": "observation.images.cam_left_wrist",
    "cam_wrist_right_rgb.mp4": "observation.images.cam_right_wrist",
    "global_realsense_rgb.mp4": "observation.images.cam_high",
    "arm_realsense_rgb.mp4": "observation.images.cam_left_wrist",
    "right_realsense_rgb.mp4": "observation.images.cam_right_wrist",
    "handeye_realsense_rgb.mp4": "observation.images.cam_left_wrist",
    "main_realsense_rgb.mp4": "observation.images.cam_high",
    "side_realsense_rgb.mp4": "observation.images.cam_right_wrist",
}


@dataclasses.dataclass(frozen=True)
class TaskSchema:
    mode: str
    robot_type: str
    state_names: list[str]
    video_map: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="Raw RoboChallenge task root")
    parser.add_argument("--output-root", type=Path, required=True, help="Output LeRobot task root")
    parser.add_argument("--episode-stop", type=int, default=None, help="Exclusive stop episode index for testing")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--overwrite", action="store_true", help="Remove existing output directory first")
    mode_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partially converted task by reusing completed episodes",
    )
    parser.add_argument("--video-backend", type=str, default="pyav", help="Video backend for decode_video_frames")
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


def load_first_jsonl_row(path: Path) -> dict:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No rows found in {path}")


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


def scalarize(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        if len(value) == 0:
            return None
        if len(value) == 1:
            return float(value[0])
        raise ValueError(f"Expected scalar-like list, got {value}")
    raise TypeError(f"Unsupported scalar value type: {type(value)}")


def extract_single_arm_state(row: dict) -> list[float]:
    joints = [float(x) for x in row["joint_positions"]]
    gripper_value = None
    if "gripper_width" in row:
        gripper_value = scalarize(row["gripper_width"])
    if gripper_value is None and "gripper" in row:
        try:
            gripper_value = scalarize(row["gripper"])
        except ValueError:
            gripper_value = None
    if gripper_value is not None:
        joints.append(gripper_value)
    return joints


def extract_dual_arm_state(left_row: dict, right_row: dict) -> list[float]:
    return (
        [float(x) for x in left_row["joint_positions"]]
        + [float(left_row["gripper"])]
        + [float(x) for x in right_row["joint_positions"]]
        + [float(right_row["gripper"])]
    )


def detect_video_map(video_dir: Path) -> dict[str, str]:
    video_map = {}
    for raw_path in sorted(video_dir.glob("*.mp4")):
        raw_name = raw_path.name
        if raw_name not in RAW_VIDEO_TO_FEATURE:
            raise ValueError(f"Unsupported video filename: {raw_name}")
        video_map[raw_name] = RAW_VIDEO_TO_FEATURE[raw_name]
    if not video_map:
        raise ValueError(f"No mp4 files found in {video_dir}")
    return video_map


def detect_schema(raw_root: Path, raw_episode: Path) -> TaskSchema:
    video_map = detect_video_map(raw_episode / "videos")

    left_states = raw_episode / "states" / "left_states.jsonl"
    right_states = raw_episode / "states" / "right_states.jsonl"
    if left_states.exists() and right_states.exists():
        state_names = [f"left_joint_{i}" for i in range(6)] + ["left_gripper"]
        state_names += [f"right_joint_{i}" for i in range(6)] + ["right_gripper"]
        return TaskSchema(mode="dual_arm", robot_type="aloha_5", state_names=state_names, video_map=video_map)

    state_path = raw_episode / "states" / "states.jsonl"
    first_row = load_first_jsonl_row(state_path)
    sample_state = extract_single_arm_state(first_row)
    joint_dim = len(first_row["joint_positions"])
    if joint_dim == 6 and len(sample_state) == 7:
        robot_type = "arx5"
    elif joint_dim == 7 and len(sample_state) == 8:
        robot_type = "single_arm_7dof"
    else:
        robot_type = "single_arm"
    state_names = [f"joint_{i}" for i in range(joint_dim)]
    if len(sample_state) == joint_dim + 1:
        state_names.append("gripper")
    return TaskSchema(mode="single_arm", robot_type=robot_type, state_names=state_names, video_map=video_map)


def load_task_prompt(raw_root: Path) -> str:
    task_desc_path = raw_root / "task_desc.json"
    if task_desc_path.exists():
        task_desc = load_json(task_desc_path)
        prompt = task_desc.get("prompt") or task_desc.get("task_name")
        if prompt:
            return prompt
    return raw_root.name.replace("_", " ")


def make_video_stats(video_path: Path, num_frames: int, backend: str) -> dict[str, np.ndarray]:
    sampled_indices = sample_indices(num_frames)
    sampled_timestamps = [idx / FPS for idx in sampled_indices]
    frames = decode_video_frames(video_path, sampled_timestamps, tolerance_s=1.0 / FPS + 1e-6, backend=backend)
    frames_np = frames.cpu().numpy()
    stats = get_feature_stats(frames_np, axis=(0, 2, 3), keepdims=True)
    return {key: value if key == "count" else np.squeeze(value, axis=0) for key, value in stats.items()}


def build_episode_arrays(schema: TaskSchema, raw_episode: Path, episode_index: int, global_offset: int) -> dict[str, list]:
    if schema.mode == "dual_arm":
        left_rows = load_jsonl(raw_episode / "states" / "left_states.jsonl")
        right_rows = load_jsonl(raw_episode / "states" / "right_states.jsonl")
        if len(left_rows) != len(right_rows):
            raise ValueError(f"Mismatched dual-arm row counts in {raw_episode}")
        states = np.asarray(
            [extract_dual_arm_state(left_row, right_row) for left_row, right_row in zip(left_rows, right_rows, strict=True)],
            dtype=np.float32,
        )
    else:
        state_rows = load_jsonl(raw_episode / "states" / "states.jsonl")
        states = np.asarray([extract_single_arm_state(row) for row in state_rows], dtype=np.float32)

    num_frames = len(states)
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


def build_episode_stats_from_table(table: datasets.Dataset, video_paths: dict[str, Path], backend: str) -> dict[str, dict]:
    num_frames = len(table)
    stats = {}
    for key in table.column_names:
        array = np.asarray(table[key])
        stats[key] = get_feature_stats(array, axis=0, keepdims=array.ndim == 1)
    for feature_key, video_path in video_paths.items():
        stats[feature_key] = make_video_stats(video_path, num_frames, backend)
    return stats


def load_episode_table(parquet_path: Path) -> datasets.Dataset:
    return datasets.load_dataset("parquet", data_files=str(parquet_path), split="train")


def load_existing_serialized_stats(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    return {int(row["episode_index"]): row["stats"] for row in rows}


def output_video_paths(output_root: Path, schema: TaskSchema, episode_index: int) -> dict[str, Path]:
    chunk_index = get_chunk_index(episode_index)
    return {
        feature_key: output_root / f"videos/chunk-{chunk_index:03d}/{feature_key}/episode_{episode_index:06d}.mp4"
        for feature_key in schema.video_map.values()
    }


def episode_features(state_dim: int) -> datasets.Features:
    return datasets.Features(
        {
            "observation.state": datasets.Sequence(datasets.Value("float32"), length=state_dim),
            "action": datasets.Sequence(datasets.Value("float32"), length=state_dim),
            "timestamp": datasets.Value("float32"),
            "frame_index": datasets.Value("int64"),
            "episode_index": datasets.Value("int64"),
            "index": datasets.Value("int64"),
            "task_index": datasets.Value("int64"),
        }
    )


def build_info(
    schema: TaskSchema,
    total_episodes: int,
    total_frames: int,
    total_videos: int,
    video_infos: dict[str, dict],
    max_episode_index: int,
) -> dict:
    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(schema.state_names)],
            "names": [schema.state_names],
        },
        "action": {
            "dtype": "float32",
            "shape": [len(schema.state_names)],
            "names": [schema.state_names],
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
        "robot_type": schema.robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_videos,
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

    data_dir = raw_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")
    if not os.access(data_dir, os.R_OK | os.X_OK):
        raise PermissionError(f"Cannot access task data directory: {data_dir}")

    if output_root.exists():
        if not args.overwrite and not args.resume:
            raise FileExistsError(f"{output_root} already exists. Pass --overwrite to replace it.")
        if args.overwrite:
            shutil.rmtree(output_root)

    prompt = load_task_prompt(raw_root)
    raw_episodes = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if args.episode_stop is not None:
        raw_episodes = [p for p in raw_episodes if int(p.name.split("_")[-1]) < args.episode_stop]
    if not raw_episodes:
        raise ValueError(f"No episodes found under {data_dir}")

    schema = detect_schema(raw_root, raw_episodes[0])
    feature_schema = episode_features(len(schema.state_names))
    existing_stats = load_existing_serialized_stats(output_root / "meta" / "episodes_stats.jsonl") if args.resume else {}

    episode_rows = []
    episode_stats_rows = []
    global_offset = 0
    total_videos = 0
    video_infos: dict[str, dict] = {}
    max_episode_index = -1
    reused_episodes = 0
    rewritten_episodes = 0

    for raw_episode in tqdm(raw_episodes, desc=f"Converting {raw_root.name}"):
        episode_index = int(raw_episode.name.split("_")[-1])
        max_episode_index = max(max_episode_index, episode_index)
        chunk_index = get_chunk_index(episode_index)
        parquet_path = output_root / f"data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet"
        raw_video_paths = {
            feature_key: raw_episode / "videos" / raw_name for raw_name, feature_key in schema.video_map.items()
        }
        linked_video_paths = output_video_paths(output_root, schema, episode_index)

        episode_complete = parquet_path.exists() and all(path.exists() for path in linked_video_paths.values())
        if args.resume and episode_complete:
            dataset = load_episode_table(parquet_path)
            num_frames = len(dataset)
            if num_frames == 0:
                raise ValueError(f"Existing parquet has no rows: {parquet_path}")
            if int(dataset[0]["episode_index"]) != episode_index:
                raise ValueError(f"Existing parquet has mismatched episode index: {parquet_path}")
            if int(dataset[0]["index"]) != global_offset:
                raise ValueError(
                    f"Existing parquet has unexpected global index start for episode {episode_index}: "
                    f"{dataset[0]['index']} != {global_offset}"
                )

            for feature_key, video_path in linked_video_paths.items():
                if feature_key not in video_infos:
                    video_infos[feature_key] = get_video_info(video_path)
            total_videos += len(linked_video_paths)

            serialized_stats = existing_stats.get(episode_index)
            if serialized_stats is None:
                episode_stats = build_episode_stats_from_table(dataset, linked_video_paths, args.video_backend)
                serialized_stats = serialize_dict(episode_stats)

            episode_stats_rows.append({"episode_index": episode_index, "stats": serialized_stats})
            episode_rows.append({"episode_index": episode_index, "tasks": [prompt], "length": num_frames})
            global_offset += num_frames
            reused_episodes += 1
            continue

        episode_arrays = build_episode_arrays(schema, raw_episode, episode_index, global_offset)
        num_frames = len(episode_arrays["frame_index"])

        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        dataset = datasets.Dataset.from_dict(episode_arrays, features=feature_schema)
        dataset.to_parquet(str(parquet_path))

        linked_videos_for_stats: dict[str, Path] = {}
        for raw_name, feature_key in schema.video_map.items():
            src = raw_episode / "videos" / raw_name
            dst = linked_video_paths[feature_key]
            ensure_link(src, dst)
            linked_videos_for_stats[feature_key] = src
            if feature_key not in video_infos:
                video_infos[feature_key] = get_video_info(src)
        total_videos += len(linked_video_paths)

        episode_stats = build_episode_stats(episode_arrays, linked_videos_for_stats, args.video_backend)
        episode_stats_rows.append({"episode_index": episode_index, "stats": serialize_dict(episode_stats)})
        episode_rows.append({"episode_index": episode_index, "tasks": [prompt], "length": num_frames})
        global_offset += num_frames
        rewritten_episodes += 1

    tasks_rows = [{"task_index": 0, "task": prompt}]
    info = build_info(schema, len(episode_rows), global_offset, total_videos, video_infos, max_episode_index)

    write_json(output_root / "meta" / "info.json", info)
    write_jsonl(output_root / "meta" / "tasks.jsonl", tasks_rows)
    write_jsonl(output_root / "meta" / "episodes.jsonl", episode_rows)
    write_jsonl(output_root / "meta" / "episodes_stats.jsonl", episode_stats_rows)

    if args.resume:
        print(
            f"Resumed {raw_root.name}: reused {reused_episodes} episode(s), "
            f"wrote {rewritten_episodes} episode(s) to {output_root}"
        )
    else:
        print(f"Converted {len(episode_rows)} episodes from {raw_root.name} to {output_root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Batch-convert RoboChallenge raw tasks to LeRobot format."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_DATA_ROOT = os.environ.get("OPENPI_DATA_ROOT", "/data/datasets")
EXCLUDED_DIRS = {".cache", "lerobot_dataset"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(_DATA_ROOT) / "RobochallengeDataset",
        help="Root directory containing raw RoboChallenge task folders",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(_DATA_ROOT) / "RobochallengeDataset/lerobot_dataset",
        help="Root directory for converted LeRobot datasets",
    )
    parser.add_argument("--task", action="append", default=[], help="Specific task(s) to convert")
    parser.add_argument("--episode-stop", type=int, default=None, help="Exclusive episode stop for testing")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing task outputs")
    parser.add_argument("--skip-existing", action="store_true", help="Skip tasks already present in output root")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume partially converted task outputs and skip tasks that are already complete",
    )
    parser.add_argument("--continue-on-error", action="store_true", help="Continue converting after a task failure")
    parser.add_argument("--video-backend", type=str, default="pyav", help="Video backend for decoding")
    return parser.parse_args()


def iter_tasks(dataset_root: Path, requested_tasks: list[str]) -> list[Path]:
    if requested_tasks:
        return [dataset_root / task for task in requested_tasks]
    return sorted(
        path for path in dataset_root.iterdir() if path.is_dir() and path.name not in EXCLUDED_DIRS
    )


def count_raw_episodes(task_root: Path, episode_stop: int | None) -> int:
    data_dir = task_root / "data"
    episodes = [p for p in data_dir.iterdir() if p.is_dir()]
    if episode_stop is not None:
        episodes = [p for p in episodes if int(p.name.split("_")[-1]) < episode_stop]
    return len(episodes)


def count_jsonl_rows(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def is_task_complete(task_root: Path, output_task_root: Path, episode_stop: int | None) -> bool:
    meta_dir = output_task_root / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    stats_path = meta_dir / "episodes_stats.jsonl"
    if not info_path.exists() or not episodes_path.exists() or not stats_path.exists():
        return False

    expected_episodes = count_raw_episodes(task_root, episode_stop)
    parquet_count = sum(1 for _ in output_task_root.glob("data/chunk-*/episode_*.parquet"))
    if parquet_count != expected_episodes:
        return False

    try:
        with info_path.open() as f:
            info = json.load(f)
    except json.JSONDecodeError:
        return False

    if info.get("total_episodes") != expected_episodes:
        return False
    if count_jsonl_rows(episodes_path) != expected_episodes:
        return False
    if count_jsonl_rows(stats_path) != expected_episodes:
        return False
    return True


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    converter = Path(__file__).resolve().with_name("convert_robochallenge_task_to_lerobot.py")

    failures: list[tuple[str, int]] = []
    tasks = iter_tasks(dataset_root, args.task)
    for task_root in tasks:
        if not task_root.exists():
            failures.append((task_root.name, 1))
            if not args.continue_on_error:
                break
            continue

        output_task_root = output_root / task_root.name
        resume_task = False
        if output_task_root.exists():
            if args.overwrite:
                resume_task = False
            elif args.resume:
                if is_task_complete(task_root, output_task_root, args.episode_stop):
                    print(f"[skip-existing] {task_root.name}")
                    continue
                print(f"[resume] {task_root.name}")
                resume_task = True
            elif args.skip_existing:
                print(f"[skip-existing] {task_root.name}")
                continue

        cmd = [
            sys.executable,
            str(converter),
            "--raw-root",
            str(task_root),
            "--output-root",
            str(output_task_root),
            "--video-backend",
            args.video_backend,
        ]
        if args.episode_stop is not None:
            cmd.extend(["--episode-stop", str(args.episode_stop)])
        if args.overwrite:
            cmd.append("--overwrite")
        if resume_task:
            cmd.append("--resume")

        if not resume_task:
            print(f"[convert] {task_root.name}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures.append((task_root.name, result.returncode))
            if not args.continue_on_error:
                break

    if failures:
        print("Failed tasks:")
        for task_name, code in failures:
            print(f"  - {task_name}: exit code {code}")
        raise SystemExit(1)

    print("All requested tasks converted successfully.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Trim the retraction phase from the press_button dataset.

For each episode, find the frame where joint_1 (the main extension joint that drives
the arm forward) reaches its maximum, then keep frames up to that point plus a small
buffer. This removes the arm-retracting phase so the model only learns the
extending-to-press-button behavior.

Safety rules:
- No movement (joint_1 range < MIN_J1_RANGE): drop the episode entirely.
- Peak before 40% of the episode: keep at least 40% of total frames (floor), because
  the arm hasn't retracted much yet and cutting too early would discard useful data.
- Peak at frame 0: drop the episode (pure retraction, no extension phase).

Usage:
    python scripts/trim_press_button_retraction.py

After running, you must:
1. Recompute norm stats for the trimmed dataset
2. Update the train config's local_roots to point to the new dataset path
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ORIGINAL_ROOT = Path("realworld_piper/press_button_20260414")
NEW_ROOT = Path("realworld_piper/press_button_20260414_trimmed")

# ---------------------------------------------------------------------------
# Trimming parameters
# ---------------------------------------------------------------------------
BUFFER_FRAMES = 10       # frames to keep after the peak extension (includes the press)
MIN_J1_RANGE = 0.01      # skip trimming entirely if joint_1 moves less than this
MIN_KEEP_FRACTION = 0.40 # floor: always keep at least this fraction of each episode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# Sentinel: episode should be dropped entirely.
_DROP = -1


def find_cut_frame(df: pd.DataFrame) -> int:
    """Return the last frame index to KEEP (inclusive), or _DROP to skip the episode.

    The episode will be trimmed to frames [0, cut_frame].
    """
    state = np.array(df["observation.state"].tolist())
    j1 = state[:, 0]
    total_frames = len(j1)

    # ---- Rule 1: no movement → drop ----
    if j1.max() - j1.min() < MIN_J1_RANGE:
        return _DROP

    peak_frame = int(np.argmax(j1))

    # ---- Rule 2: peak at frame 0 → pure retraction, drop ----
    if peak_frame == 0:
        return _DROP

    # ---- Normal case: cut at peak + buffer ----
    cut = peak_frame + BUFFER_FRAMES

    # ---- Safety floor: never keep less than MIN_KEEP_FRACTION ----
    floor = int(total_frames * MIN_KEEP_FRACTION)
    if cut < floor:
        cut = floor

    return min(cut, total_frames - 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    info = load_json(ORIGINAL_ROOT / "meta" / "info.json")
    episodes = load_jsonl(ORIGINAL_ROOT / "meta" / "episodes.jsonl")
    total_episodes = info["total_episodes"]
    chunks_size = info["chunks_size"]

    assert total_episodes == len(episodes), (
        f"info.json total_episodes={total_episodes} != episodes.jsonl len={len(episodes)}"
    )

    # Create output directory structure
    for subdir in ["data/chunk-000", "meta"]:
        (NEW_ROOT / subdir).mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Symlink videos (no modification needed - parquet frame_index
    #              still correctly maps to the original video frames) ----
    print("Creating video symlinks...")
    videos_src = (ORIGINAL_ROOT / "videos").resolve()
    videos_dst = NEW_ROOT / "videos"
    if videos_dst.exists() or videos_dst.is_symlink():
        if videos_dst.is_symlink():
            videos_dst.unlink()
        else:
            shutil.rmtree(videos_dst)
    videos_dst.symlink_to(videos_src)
    print(f"  {videos_dst} -> {videos_src}")

    # ---- Step 2: Trim each episode ----
    print(f"Trimming {total_episodes} episodes "
          f"(buffer={BUFFER_FRAMES}, min_keep_frac={MIN_KEEP_FRACTION:.0%})...")

    new_total_frames = 0
    new_episodes = []
    dropped_episodes = []
    rules_used = {"dropped_no_movement": 0, "dropped_peak_zero": 0, "floored": 0, "normal": 0}

    for ep in tqdm(episodes, desc="Trimming episodes"):
        ep_idx = ep["episode_index"]
        chunk = ep_idx // chunks_size

        src_path = ORIGINAL_ROOT / info["data_path"].format(
            episode_chunk=chunk, episode_index=ep_idx
        )
        df = pd.read_parquet(src_path)
        original_len = len(df)

        state = np.array(df["observation.state"].tolist())
        j1 = state[:, 0]
        peak_frame = int(np.argmax(j1))

        cut_frame = find_cut_frame(df)

        if cut_frame == _DROP:
            if j1.max() - j1.min() < MIN_J1_RANGE:
                rules_used["dropped_no_movement"] += 1
                rule = "dropped_no_movement"
            else:
                rules_used["dropped_peak_zero"] += 1
                rule = "dropped_peak_zero"
            dropped_episodes.append(ep_idx)
            tqdm.write(f"  ep {ep_idx:03d}: DROPPED ({rule}), {original_len} frames")
            continue

        # Determine which keep-rule applied
        naive_cut = peak_frame + BUFFER_FRAMES
        floor = int(original_len * MIN_KEEP_FRACTION)
        if naive_cut < floor:
            rules_used["floored"] += 1
            rule = "floored"
        else:
            rules_used["normal"] += 1
            rule = "normal"

        df_trimmed = df.iloc[: cut_frame + 1].copy()
        new_len = len(df_trimmed)

        # Write trimmed parquet
        dst_path = NEW_ROOT / info["data_path"].format(
            episode_chunk=0, episode_index=ep_idx
        )
        df_trimmed.to_parquet(dst_path, index=False)

        new_total_frames += new_len
        new_episodes.append({**ep, "length": new_len})

        # Verbose logging for first few and every 100th
        if ep_idx < 3 or ep_idx % 100 == 0:
            peak_pct = peak_frame / original_len * 100
            kept_pct = new_len / original_len * 100
            tqdm.write(
                f"  ep {ep_idx:03d}: {original_len} -> {new_len} frames "
                f"(peak at {peak_pct:.0f}%, kept {kept_pct:.0f}%, rule={rule})"
            )

    # ---- Step 3: Update metadata ----
    print("\nUpdating metadata...")

    new_episode_count = len(new_episodes)
    num_dropped = len(dropped_episodes)

    # info.json
    new_info = {**info}
    new_info["total_episodes"] = new_episode_count
    new_info["total_frames"] = new_total_frames
    new_info["total_videos"] = new_episode_count * 4  # 4 camera views × kept episodes
    save_json(NEW_ROOT / "meta" / "info.json", new_info)
    print(f"  info.json: total_episodes {info['total_episodes']} -> {new_episode_count}")
    print(f"  info.json: total_frames {info['total_frames']} -> {new_total_frames}")
    print(f"  info.json: total_videos {info['total_videos']} -> {new_info['total_videos']}")

    # episodes.jsonl
    save_jsonl(NEW_ROOT / "meta" / "episodes.jsonl", new_episodes)
    print(f"  episodes.jsonl: updated {len(new_episodes)} episodes")

    # Copy unchanged metadata files
    for fname in ["tasks.jsonl", "modality.json"]:
        src = ORIGINAL_ROOT / "meta" / fname
        if src.exists():
            shutil.copy2(src, NEW_ROOT / "meta" / fname)
            print(f"  {fname}: copied")

    # ---- Summary ----
    reduction = (1 - new_total_frames / info["total_frames"]) * 100
    print(f"\n{'='*60}")
    print(f"Done! New dataset at: {NEW_ROOT}")
    print(f"  Original episodes: {total_episodes}")
    print(f"  Kept episodes:     {new_episode_count}")
    print(f"  Dropped episodes:  {num_dropped}")
    if dropped_episodes:
        print(f"  Dropped indices:   {dropped_episodes}")
    print(f"  Original frames:   {info['total_frames']}")
    print(f"  Trimmed frames:    {new_total_frames}")
    print(f"  Reduction:         {reduction:.1f}%")
    print(f"  Rules applied:")
    for rule, count in rules_used.items():
        if count > 0:
            print(f"    {rule}: {count} episodes")
    print(f"\nNext steps:")
    print(f"  1. Recompute norm stats:")
    print(f"     python scripts/compute_norm_stats.py --config-name pi05_piper_press_button_20260414_4cam_hold_dim7_13")
    print(f"  2. Recompute episode stats (optional, for video-based stats):")
    print(f"     python scripts/recompute_lerobot_episode_stats.py --dataset-root {NEW_ROOT} --overwrite")
    print(f"  3. Update the train config to point to the new dataset:")
    print(f"     Edit src/openpi/training/config.py, change local_roots to '{NEW_ROOT}'")


if __name__ == "__main__":
    main()

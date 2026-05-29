#!/usr/bin/env python3
"""
R1Lite Offline Evaluation (LeRobot -> OpenPI policy server)

Goal:
- Use real-robot LeRobot dataset as input (state + RGB images).
- Query the OpenPI WebSocket policy server to get predicted actions.
- Compare predicted actions vs. ground-truth actions from dataset.
- Plot 14D arm actions (L_arm 6 + L_grip 1 + R_arm 6 + R_grip 1) in ONE figure (14 subplots).

Key features:
- Supports selecting episodes by:
    * --episode-ids "65" or "0,6,65" (preferred)
    * or --episodes N (first N episodes if episode-ids not given)
- Supports max number of evaluated frames via --max-steps
- Saves:
    * pred_gt_14d.npz      (pred14, gt14 arrays)
    * metrics.parquet      (per-frame MAE/RMSE + inference time)
    * plots/actions_14d_grid.png  (one image with 14 subplots)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from openpi_client import websocket_client_policy
from openpi_client import image_tools


# -----------------------------
# Video Reading Backends
# -----------------------------
class VideoReaderBase:
    """Abstract video reader interface."""

    def get_rgb(self, frame_index: int) -> np.ndarray:
        """Return RGB frame as uint8 HxWx3."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class PyAVVideoReader(VideoReaderBase):
    """
    PyAV-based reader (software decoding).
    This typically avoids the 'hardware accelerated AV1 decoding' warnings.

    Note:
    - We assume frames are accessed in non-decreasing frame_index order.
    - If the dataset jumps around, we fall back to a reset+scan (slower but safe).
    """

    def __init__(self, mp4_path: Path, out_h: int = 224, out_w: int = 224):
        import av  # lazy import

        # Reduce ffmpeg/av logging noise
        try:
            av.logging.set_level(av.logging.ERROR)
        except Exception:
            pass

        self.mp4_path = mp4_path
        self.out_h = out_h
        self.out_w = out_w

        self.container = av.open(str(mp4_path))
        self.stream = self.container.streams.video[0]
        self._decoded_iter = self.container.decode(self.stream)

        self._cur_frame_idx = -1
        self._cur_rgb: Optional[np.ndarray] = None

    def _reset(self) -> None:
        """Reset decode iterator to the beginning."""
        self.container.close()
        import av
        self.container = av.open(str(self.mp4_path))
        self.stream = self.container.streams.video[0]
        self._decoded_iter = self.container.decode(self.stream)
        self._cur_frame_idx = -1
        self._cur_rgb = None

    def get_rgb(self, frame_index: int) -> np.ndarray:
        if frame_index < 0:
            return np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)

        # If requesting earlier frame, reset and re-scan.
        if frame_index <= self._cur_frame_idx:
            self._reset()

        # Decode sequentially until reaching desired frame_index.
        try:
            while self._cur_frame_idx < frame_index:
                frame = next(self._decoded_iter)
                self._cur_frame_idx += 1
                rgb = frame.to_rgb().to_ndarray()  # HxWx3, uint8
                rgb = image_tools.resize_with_pad(rgb, self.out_h, self.out_w)
                rgb = image_tools.convert_to_uint8(rgb)
                self._cur_rgb = rgb
        except StopIteration:
            # Out of frames
            return np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)
        except Exception:
            # Any decode failure -> return zeros
            return np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)

        return self._cur_rgb if self._cur_rgb is not None else np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)

    def close(self) -> None:
        try:
            self.container.close()
        except Exception:
            pass


class OpenCVVideoReader(VideoReaderBase):
    """OpenCV (cv2.VideoCapture) backend."""

    def __init__(self, mp4_path: Path, out_h: int = 224, out_w: int = 224):
        import cv2  # lazy import

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(str(mp4_path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {mp4_path}")
        self.out_h = out_h
        self.out_w = out_w

    def get_rgb(self, frame_index: int) -> np.ndarray:
        self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            return np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)

        rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        rgb = image_tools.resize_with_pad(rgb, self.out_h, self.out_w)
        rgb = image_tools.convert_to_uint8(rgb)
        return rgb

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


def make_video_reader(backend: str, mp4_path: Path, out_h: int = 224, out_w: int = 224) -> VideoReaderBase:
    """Factory for video readers."""
    backend = backend.lower()
    if backend == "pyav":
        return PyAVVideoReader(mp4_path, out_h=out_h, out_w=out_w)
    if backend == "opencv":
        return OpenCVVideoReader(mp4_path, out_h=out_h, out_w=out_w)
    raise ValueError(f"Unknown video backend: {backend}")


# -----------------------------
# CLI Arguments
# -----------------------------
@dataclass
class Config:
    dataset_root: Path
    host: str
    port: int
    prompt: str
    episodes: int
    episode_ids: str
    max_steps: int
    out_dir: Path
    video_backend: str


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="R1Lite offline evaluation from LeRobot dataset.")
    p.add_argument("--dataset-root", type=Path, required=True, help="LeRobot dataset root (contains data/meta/videos).")
    p.add_argument("--host", type=str, default="127.0.0.1", help="OpenPI policy server host.")
    p.add_argument("--port", type=int, default=8000, help="OpenPI policy server port.")
    p.add_argument("--prompt", type=str, default="Fold the green T-shirt.", help="Prompt sent to the policy.")
    p.add_argument("--episodes", type=int, default=1, help="If --episode-ids not set, run first N episodes.")
    p.add_argument(
        "--episode-ids",
        type=str,
        default="",
        help='Comma-separated episode indices to run, e.g. "65" or "0,6,65". Overrides --episodes.',
    )
    p.add_argument("--max-steps", type=int, default=600, help="Max number of frames to evaluate in total.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("examples/r1lite/eval_out"),
        help="Output directory for plots and metrics.",
    )
    p.add_argument(
        "--video-backend",
        type=str,
        default="pyav",
        choices=["pyav", "opencv"],
        help="Video decode backend. Use pyav to reduce AV1 hw decode warnings.",
    )
    a = p.parse_args()
    return Config(
        dataset_root=a.dataset_root,
        host=a.host,
        port=a.port,
        prompt=a.prompt,
        episodes=a.episodes,
        episode_ids=a.episode_ids,
        max_steps=a.max_steps,
        out_dir=a.out_dir,
        video_backend=a.video_backend,
    )


# -----------------------------
# Data helpers (obs/action)
# -----------------------------
CAM_KEYS = [
    "observation.images.head_rgb",
    "observation.images.left_wrist_rgb",
    "observation.images.right_wrist_rgb",
]


def build_obs(row: dict, frames: Dict[str, np.ndarray], prompt: str) -> dict:
    """
    Build an OpenPI observation dict that matches galaxea_policy expected keys.
    - State keys are already dotted in parquet.
    - Grippers are scalar in parquet; we convert to shape (1,) float32 arrays.
    - Images are HxWx3 uint8.
    """
    obs = {}

    list_keys = [
        "observation.state.left_arm",
        "observation.state.left_arm.velocities",
        "observation.state.right_arm",
        "observation.state.right_arm.velocities",
        "observation.state.chassis.imu",
        "observation.state.chassis",
        "observation.state.chassis.velocities",
        "observation.state.torso",
        "observation.state.torso.velocities",
        "observation.state.left_ee_pose",
        "observation.state.right_ee_pose",
    ]
    for k in list_keys:
        obs[k] = np.asarray(row[k], dtype=np.float32)

    obs["observation.state.left_gripper"] = np.asarray([row["observation.state.left_gripper"]], dtype=np.float32)
    obs["observation.state.right_gripper"] = np.asarray([row["observation.state.right_gripper"]], dtype=np.float32)

    for cam_key in CAM_KEYS:
        obs[cam_key] = frames.get(cam_key, np.zeros((224, 224, 3), dtype=np.uint8))

    obs["prompt"] = prompt
    return obs


def extract_gt14(row: dict) -> np.ndarray:
    """
    Ground truth 14D action from parquet:
    [L_arm(6), L_grip(1), R_arm(6), R_grip(1)]
    """
    la = np.asarray(row["action.left_arm"], dtype=np.float32).reshape(-1)[:6]
    ra = np.asarray(row["action.right_arm"], dtype=np.float32).reshape(-1)[:6]
    lg = np.asarray([row["action.left_gripper"]], dtype=np.float32)
    rg = np.asarray([row["action.right_gripper"]], dtype=np.float32)
    gt14 = np.concatenate([la, lg, ra, rg], axis=0)
    if gt14.shape[0] != 14:
        raise ValueError(f"GT14 dim mismatch: {gt14.shape}")
    return gt14


def extract_pred14(result: dict) -> np.ndarray:
    """
    Predicted 14D action from OpenPI inference result.
    Many OpenPI policies output a chunk of horizon H; we take the FIRST step (t=0).
    """
    la = np.asarray(result["action.left_arm"], dtype=np.float32)   # (H,6)
    ra = np.asarray(result["action.right_arm"], dtype=np.float32)  # (H,6)

    lg = np.asarray(result["action.left_gripper"], dtype=np.float32)   # (H,) or (H,1)
    rg = np.asarray(result["action.right_gripper"], dtype=np.float32)  # (H,) or (H,1)

    la0 = la[0].reshape(-1)[:6]
    ra0 = ra[0].reshape(-1)[:6]

    lg0 = float(lg[0]) if lg.ndim == 1 else float(lg[0].reshape(-1)[0])
    rg0 = float(rg[0]) if rg.ndim == 1 else float(rg[0].reshape(-1)[0])

    pred14 = np.concatenate(
        [la0, np.asarray([lg0], np.float32), ra0, np.asarray([rg0], np.float32)],
        axis=0,
    )
    if pred14.shape[0] != 14:
        raise ValueError(f"Pred14 dim mismatch: {pred14.shape}")
    return pred14


# -----------------------------
# Plotting
# -----------------------------
def plot_14d_grid(pred14_all: np.ndarray, gt14_all: np.ndarray, out_png: Path) -> None:
    """
    Plot 14 action dimensions in ONE figure with 14 subplots (7 rows x 2 cols).
    Each subplot overlays Real(GT) vs Model(Pred) curves with clear legends.
    Also add a global legend for the whole figure.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    names = [
        "L_arm_0", "L_arm_1", "L_arm_2", "L_arm_3", "L_arm_4", "L_arm_5",
        "L_gripper",
        "R_arm_0", "R_arm_1", "R_arm_2", "R_arm_3", "R_arm_4", "R_arm_5",
        "R_gripper",
    ]

    T = pred14_all.shape[0]
    x = np.arange(T)

    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(18, 20), sharex=True)
    axes = axes.reshape(-1)

    # We'll capture handles once for a global legend
    global_handles = None
    global_labels = None

    for d in range(14):
        ax = axes[d]

        # Use explicit labels for clarity
        h1 = ax.plot(x, gt14_all[:, d], label="Real")
        h2 = ax.plot(x, pred14_all[:, d], label="Pred")

        ax.set_title(f"{d:02d} - {names[d]}")
        ax.grid(True, alpha=0.3)

        # Put legend in each subplot (small & readable)
        ax.legend(loc="upper right", fontsize=8, frameon=True)

        # Save handles for global legend (use first subplot)
        if global_handles is None:
            global_handles = [h1[0], h2[0]]
            global_labels = ["Real", "Pred"]

    fig.suptitle("R1Lite Action (14D): Real vs Pred", fontsize=16)

    # Add a global legend on top center (in addition to per-axes legends)
    fig.legend(
        global_handles,
        global_labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.99),
        fontsize=11,
    )

    # Leave space for the global legend + title
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# -----------------------------
# Episode selection
# -----------------------------
def select_episode_parquets(data_dir: Path, episodes: int, episode_ids: str) -> List[Path]:
    """
    Select parquet files (episode_XXXXXX.parquet) according to CLI options.
    - If episode_ids provided: select those exact ids.
    - Else: take first `episodes` files sorted by name.
    """
    parquets = sorted(data_dir.glob("episode_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No episode parquet found under: {data_dir}")

    if episode_ids.strip():
        want = [int(x) for x in episode_ids.split(",") if x.strip() != ""]
        want_set = set(want)
        selected = []
        for pq in parquets:
            eid = int(pq.stem.split("_")[-1])
            if eid in want_set:
                selected.append(pq)
        if not selected:
            raise ValueError(f"--episode-ids {want} not found under {data_dir}")
        return selected

    return parquets[: max(1, episodes)]


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    cfg = parse_args()

    dataset_root = cfg.dataset_root
    data_dir = dataset_root / "data" / "chunk-000"
    video_dir = dataset_root / "videos" / "chunk-000"

    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_plot = out_dir / "plots" / "actions_predict_and_real_action_chunk30.png"

    # Connect to OpenPI policy server (must be started separately, e.g., scripts/serve_policy.sh)
    client = websocket_client_policy.WebsocketClientPolicy(host=cfg.host, port=cfg.port)
    print("Server metadata:", client.get_server_metadata())

    # Select episodes to run
    parquets = select_episode_parquets(data_dir, cfg.episodes, cfg.episode_ids)
    print(f"Selected {len(parquets)} episode parquet(s).")

    pred_list: List[np.ndarray] = []
    gt_list: List[np.ndarray] = []
    recs: List[dict] = []

    steps = 0

    for pq in parquets:
        episode_id = int(pq.stem.split("_")[-1])
        df = pl.read_parquet(pq)

        # Open one reader per camera for this episode
        readers: Dict[str, VideoReaderBase] = {}
        try:
            for cam_key in CAM_KEYS:
                mp4 = video_dir / cam_key / f"episode_{episode_id:06d}.mp4"
                if mp4.exists():
                    readers[cam_key] = make_video_reader(cfg.video_backend, mp4, out_h=224, out_w=224)
                else:
                    readers[cam_key] = None  # type: ignore

            # Iterate frames in this episode
            for row in df.iter_rows(named=True):
                frame_index = int(row["frame_index"])

                # Read frames for each camera (missing -> zeros)
                frames = {}
                for cam_key in CAM_KEYS:
                    r = readers.get(cam_key, None)
                    if r is None:
                        frames[cam_key] = np.zeros((224, 224, 3), dtype=np.uint8)
                    else:
                        frames[cam_key] = r.get_rgb(frame_index)

                obs = build_obs(row, frames, cfg.prompt)

                # Policy inference
                t0 = time.time()
                result = client.infer(obs)
                infer_ms = (time.time() - t0) * 1000.0
                
                #print(f"the length of result:", len(result))
                #print(f"the keys of result:", result.keys())
                #print(f"results:", result)
                gt14 = extract_gt14(row)

                #print("gt14:", gt14)

                pred14 = extract_pred14(result)

                #print("pred14:", pred14)

                pred_list.append(pred14)
                gt_list.append(gt14)

                err = pred14 - gt14
                recs.append(
                    {
                        "episode": episode_id,
                        "frame_index": frame_index,
                        "infer_ms": float(infer_ms),
                        "mae14": float(np.mean(np.abs(err))),
                        "rmse14": float(np.sqrt(np.mean(err * err))),
                    }
                )

                if steps % 50 == 0:
                    print(
                        f"[step {steps}] ep={episode_id} frame={frame_index} "
                        f"infer={infer_ms:.1f}ms mae14={recs[-1]['mae14']:.4f}"
                    )

                steps += 1
                if steps >= cfg.max_steps:
                    break

        finally:
            for r in readers.values():
                try:
                    if r is not None:
                        r.close()
                except Exception:
                    pass

        if steps >= cfg.max_steps:
            break

    if not pred_list:
        raise RuntimeError("No frames evaluated. Check dataset paths / episode selection.")

    pred_all = np.stack(pred_list, axis=0)  # (T,14)
    gt_all = np.stack(gt_list, axis=0)      # (T,14)

    # Save numeric results
    np.savez_compressed(out_dir / "pred_gt_14d.npz", pred14=pred_all, gt14=gt_all)
    pl.DataFrame(recs).write_parquet(out_dir / "metrics.parquet")

    # Plot one figure with 14 subplots
    plot_14d_grid(pred_all, gt_all, out_plot)

    # Print summary
    err = pred_all - gt_all
    mae_per_dim = np.mean(np.abs(err), axis=0)
    rmse_per_dim = np.sqrt(np.mean(err * err, axis=0))

    print("\n=== Summary (over evaluated frames) ===")
    print("MAE per dim:", np.array2string(mae_per_dim, precision=4))
    print("RMSE per dim:", np.array2string(rmse_per_dim, precision=4))
    print(f"\nWrote outputs to:\n  {out_dir}")
    print(f"Plot saved to:\n  {out_plot}")


if __name__ == "__main__":
    main()


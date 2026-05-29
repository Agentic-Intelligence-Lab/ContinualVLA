
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np
import torch

import openpi.transforms as _transforms
import openpi.models.model as _model

# =============================================================================
# Galaxea Open World (R1Lite) LeRobot dataset policy transforms
#
# IMPORTANT:
#   - Your images are dtype="video" in meta/info.json -> pixels are stored in mp4, not parquet.
#     If you train with images, decoding is REQUIRED (pyav/torchcodec backend).
#   - Your packed dims (from your check_dataset.py):
#       state_dim = 64
#       action_dim = 23
# =============================================================================

# ----------------------------
# State blocks (packed -> 64)
# ----------------------------
# STATE_KEYS = [
#     "observation.state.left_arm",              # (6,)
#     "observation.state.left_arm.velocities",   # (6,)
#     "observation.state.right_arm",             # (6,)
#     "observation.state.right_arm.velocities",  # (6,)
#     "observation.state.chassis.imu",           # (10,)
#     "observation.state.chassis",               # (6,)
#     "observation.state.torso",                 # (4,)
#     "observation.state.torso.velocities",      # (4,)
#     "observation.state.left_gripper",          # (1,)
#     "observation.state.right_gripper",         # (1,)
#     "observation.state.left_ee_pose",          # (7,)
#     "observation.state.right_ee_pose",         # (7,)
# ]
# STATE_KEYS = [
#     "observation.state.left_arm",              # (6,)
#     "observation.state.right_arm",             # (6,)
#     "observation.state.left_arm.velocities",   # (6,)
#     "observation.state.right_arm.velocities",  # (6,)
#     "observation.state.chassis.imu",           # (10,)
#     "observation.state.chassis",               # (6,)
#     "observation.state.torso",                 # (4,)
#     "observation.state.torso.velocities",      # (4,)
#     "observation.state.left_gripper",          # (1,)
#     "observation.state.right_gripper",         # (1,)
#     "observation.state.left_ee_pose",          # (7,)
#     "observation.state.right_ee_pose",         # (7,)
# ]
STATE_KEYS = [
    "observation.state.left_arm",              # 6
    "observation.state.right_arm",             # 6
    "observation.state.left_arm.velocities",   # 6
    "observation.state.right_arm.velocities",  # 6
    "observation.state.chassis.imu",           # 10
    "observation.state.chassis",               # 3
    "observation.state.chassis.velocities",    # 3
    "observation.state.torso",                 # 4
    "observation.state.torso.velocities",      # 4
    "observation.state.left_gripper",          # 1
    "observation.state.right_gripper",         # 1
    "observation.state.left_ee_pose",          # 7
    "observation.state.right_ee_pose",         # 7
]
EXPECTED_STATE_DIM = 64

# ----------------------------
# Action blocks (packed -> 23)
# Order MUST match Outputs splitting.
# ----------------------------
ACTION_SEQUENCE_KEYS = [
    "action.left_arm",            # (H,6)
    "action.right_arm",           # (H,6)
    "action.torso.velocities",    # (H,6)
    "action.chassis.velocities",  # (H,6)
    "action.left_gripper",        # (H,1)  <-- often returned as (H,) !!! (the bug you hit)
    "action.right_gripper",       # (H,1)  <-- often returned as (H,) !!! (the bug you hit)
]
# ACTION_DIMS = {
#     "action.left_arm": 6,
#     "action.right_arm": 6,
#     "action.torso.velocities": 6,
#     "action.chassis.velocities": 3,
#     "action.left_gripper": 1,
#     "action.right_gripper": 1,
# }
ACTION_DIMS = {
    "action.left_arm": 6,
    "action.right_arm": 6,
    "action.torso.velocities": 6,
    "action.chassis.velocities": 6,   #
    "action.left_gripper": 1,
    "action.right_gripper": 1,
}
EXPECTED_ACTION_DIM = 26

# ----------------------------
# Image key mapping
# OpenPI common naming convention:
#   base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
# ----------------------------
IMAGE_KEY_MAP_3VIEW = {
    "observation.images.head_rgb": "base_0_rgb",
    "observation.images.left_wrist_rgb": "left_wrist_0_rgb",
    "observation.images.right_wrist_rgb": "right_wrist_0_rgb",
}
IMAGE_KEY_MAP_4VIEW = {
    **IMAGE_KEY_MAP_3VIEW,
    "observation.images.head_right_rgb": "head_right_0_rgb",
}


# =============================================================================
# Helper functions
# =============================================================================
def _to_numpy(x: Any) -> np.ndarray:
    """Robust conversion: torch.Tensor / numpy / list -> numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _pack_1d_float32(x: Any) -> np.ndarray:
    """Flatten any numeric block to 1D float32."""
    a = _to_numpy(x).reshape(-1)
    return a.astype(np.float32, copy=False)


def _parse_image_to_uint8_hwc(img: Any) -> np.ndarray:
    """
    Convert decoded frames to uint8 HWC.

    LeRobot may return:
      - torch float32 [C,H,W] in [0,1]
      - torch uint8  [C,H,W]
      - numpy float/uint8 either CHW or HWC
    """
    if torch.is_tensor(img):
        x = img
        if x.dtype.is_floating_point:
            x = (x.clamp(0, 1) * 255.0).to(torch.uint8)
        x = x.permute(1, 2, 0).contiguous()  # CHW -> HWC
        return x.cpu().numpy()

    x = _to_numpy(img)

    # CHW -> HWC
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[2] not in (1, 3):
        x = np.transpose(x, (1, 2, 0))

    # float -> uint8
    if np.issubdtype(x.dtype, np.floating):
        x = np.clip(x, 0.0, 1.0)
        x = (x * 255.0).astype(np.uint8)
    elif x.dtype != np.uint8:
        x = x.astype(np.uint8)

    return x


def _infer_horizon(data: Dict[str, Any]) -> int:
    """
    Infer action horizon H from available action blocks.
    Prefer 2D blocks (H,dim). If only 1D blocks exist, fall back carefully.
    """
    # Prefer any 2D (H,dim)
    for k in ACTION_SEQUENCE_KEYS:
        if k in data:
            a = _to_numpy(data[k])
            if a.ndim == 2:
                return int(a.shape[0])

    # If no 2D found, try scalar sequences like grippers: they may come as (H,)
    for k in ("action.left_gripper", "action.right_gripper"):
        if k in data:
            a = _to_numpy(data[k])
            if a.ndim == 1 and a.shape[0] > 1:
                return int(a.shape[0])

    # Otherwise treat as single-step
    return 1


def _ensure_horizon_and_dim(a: np.ndarray, *, H: int, expected_dim: int, key: str) -> np.ndarray:
    """
    Ensure block has shape (H, expected_dim).

    Acceptable input patterns:
      - (H, expected_dim) -> ok
      - (1, expected_dim) -> broadcast to (H, expected_dim)
      - (expected_dim,)   -> single-step, broadcast to (H, expected_dim)
      - (H,) when expected_dim==1 -> reshape to (H,1)  

    Anything else -> raise with key and shape.
    """
    # Case: 1D input
    if a.ndim == 1:
        # If it's a scalar time-series (H,) and expected_dim == 1, interpret as (H,1)
        if expected_dim == 1 and a.shape[0] == H:
            a = a.reshape(H, 1)
        # If it's a single-step vector (expected_dim,), interpret as (1,expected_dim)
        elif a.shape[0] == expected_dim:
            a = a.reshape(1, expected_dim)
        else:
            raise ValueError(
                f"Action key '{key}' got 1D shape={a.shape}, cannot interpret as "
                f"(H,1) with H={H} nor (1,{expected_dim})."
            )

    # Case: 2D input
    if a.ndim != 2:
        raise ValueError(f"Action key '{key}' must be 1D or 2D, got shape={a.shape}")

    if a.shape[1] != expected_dim:
        raise ValueError(f"Action key '{key}' dim mismatch: got shape={a.shape}, expected last_dim={expected_dim}")

    if a.shape[0] == H:
        return a
    if a.shape[0] == 1:
        return np.repeat(a, repeats=H, axis=0)

    raise ValueError(
        f"Action horizon mismatch for key '{key}': got H={a.shape[0]}, expected H={H} (or 1 for broadcast). "
        f"Usually means your DataConfig.action_sequence_keys / delta_timestamps is not correctly set."
    )


# =============================================================================
# Transforms
# =============================================================================
@dataclass(frozen=True)
class GalaxeaR1LiteInputs(_transforms.DataTransformFn):
    """
    Convert a Galaxea (R1Lite) LeRobot sample into OpenPI model input format.

    Output dict keys:
      - state: (64,) float32
      - image: dict[str, HWC uint8]     (keys always present; missing filled with zeros)
      - image_mask: dict[str, bool]
      - actions: (H, 23) float32        (training only)
      - prompt: str                     (prompt if exists else task else "")
    """

    model_type: _model.ModelType
    use_4th_view: bool = False

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # ----------------------------
        # [1] Pack state -> (64,)
        # ----------------------------
        missing_state = [k for k in STATE_KEYS if k not in data]
        if missing_state:
            raise KeyError(f"Missing state keys in sample: {missing_state}")

        state = np.concatenate([_pack_1d_float32(data[k]) for k in STATE_KEYS], axis=0)
        if state.shape[0] != EXPECTED_STATE_DIM:
            raise ValueError(f"Packed state dim mismatch: got {state.shape[0]}, expected {EXPECTED_STATE_DIM}")

        # ----------------------------
        # [2] Pack images (always provide keys; missing -> black image)
        # ----------------------------
        key_map: Mapping[str, str] = IMAGE_KEY_MAP_4VIEW if self.use_4th_view else IMAGE_KEY_MAP_3VIEW
        image_dict: Dict[str, np.ndarray] = {}
        mask_dict: Dict[str, np.bool_] = {}

        ref_img = None
        # First pass: decode what exists
        for src_key, dst_key in key_map.items():
            if src_key in data and data[src_key] is not None:
                im = _parse_image_to_uint8_hwc(data[src_key])
                image_dict[dst_key] = im
                mask_dict[dst_key] = np.True_
                if ref_img is None:
                    ref_img = im

        # Second pass: fill missing
        if ref_img is None:
            ref_img = np.zeros((224, 224, 3), dtype=np.uint8)
        for _, dst_key in key_map.items():
            if dst_key not in image_dict:
                image_dict[dst_key] = np.zeros_like(ref_img)
                mask_dict[dst_key] = np.False_

        # ----------------------------
        # [3] Prompt
        # ----------------------------
        prompt = data.get("prompt", None)
        if not prompt:
            prompt = data.get("task", "")
        if prompt is None:
            prompt = ""

        out: Dict[str, Any] = {
            "state": state,
            "image": image_dict,
            "image_mask": mask_dict,
            "prompt": prompt,
        }

        # ----------------------------
        # [4] Pack actions -> (H,23) (training only)
        # ----------------------------
        if all(k in data for k in ACTION_SEQUENCE_KEYS):
            H = _infer_horizon(data)

            blocks = []
            for k in ACTION_SEQUENCE_KEYS:
                a = _to_numpy(data[k]).astype(np.float32, copy=False)
                a = _ensure_horizon_and_dim(a, H=H, expected_dim=ACTION_DIMS[k], key=k)
                blocks.append(a)

            actions = np.concatenate(blocks, axis=-1)  # (H,23)
            if actions.shape[-1] != EXPECTED_ACTION_DIM:
                raise ValueError(
                    f"Packed action dim mismatch: got {actions.shape[-1]}, expected {EXPECTED_ACTION_DIM}. "
                    f"Per-key shapes: " + ", ".join([f"{k}={b.shape}" for k, b in zip(ACTION_SEQUENCE_KEYS, blocks)])
                )

            out["actions"] = actions

        return out


@dataclass(frozen=True)
class GalaxeaR1LiteOutputs(_transforms.DataTransformFn):
    """
    Convert model outputs back to Galaxea-like action blocks (mainly for inference).

    - pi05_base typically outputs/pads to 32 dims.
    - We slice the first 23 dims and split into named blocks.
    """

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "actions" not in data:
            return data

        a = _to_numpy(data["actions"])
        if a.shape[-1] < EXPECTED_ACTION_DIM:
            raise ValueError(f"Model output action dim too small: got {a.shape[-1]}, need >= {EXPECTED_ACTION_DIM}")

        # a23 = a[..., :EXPECTED_ACTION_DIM]
        out = dict(data)
        # out["actions"] = a23
        # out["action.left_arm"] = a23[..., 0:6]
        # out["action.right_arm"] = a23[..., 6:12]
        # out["action.torso.velocities"] = a23[..., 12:18]
        # out["action.chassis.velocities"] = a23[..., 18:21]
        # out["action.left_gripper"] = a23[..., 21:22]
        # out["action.right_gripper"] = a23[..., 22:23]
        
        a26 = a[..., :EXPECTED_ACTION_DIM]   # 26
        out["actions"] = a26
        out["action.left_arm"] = a26[..., 0:6]
        out["action.right_arm"] = a26[..., 6:12]
        out["action.torso.velocities"] = a26[..., 12:18]
        out["action.chassis.velocities"] = a26[..., 18:24]   # 6 dims
        out["action.left_gripper"] = a26[..., 24:25]
        out["action.right_gripper"] = a26[..., 25:26]
        return out


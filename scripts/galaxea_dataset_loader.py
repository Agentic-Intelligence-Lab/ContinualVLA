#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File: smoke_galaxea_loader.py

#
# Purpose (Why this script exists):
#   This is a *smoke test* for your Galaxea (R1Lite) LeRobot dataset integration
#   inside OpenPI, specifically for the config: "pi05_galaxea_r1lite".
#
#   It answers the question:
#     "Can OpenPI successfully build the dataloader, run all transforms, decode images
#      if needed (video->frames), and output tensors/arrays with the expected shapes?"
#
#   This script is NOT training.
#   It only loads a couple of batches and prints:
#     - observation keys
#     - state shape/dtype
#     - image keys + each image shape/dtype
#     - action chunk shape/dtype
#
# When to use it:
#   1) After you modify galaxea_policy.py / transforms / config.py
#   2) After you change repo_id / local dataset root settings
#   3) Before running compute_norm_stats.py or training
#
# Typical failure modes it helps catch early:
#   - Wrong key mapping (e.g., missing "observation.images.head_rgb")
#   - Action packing dimension mismatch (e.g., 23 vs 32)
#   - Image decoding backend issues (torchcodec/ffmpeg/pyav)
#   - Transforms producing wrong shapes (e.g., HWC vs CHW)
#   - Prompt/task injection issues (missing prompt tokens)
#
# How it works (High level):
#   - cfg.get_config("pi05_galaxea_r1lite") loads the TrainConfig from openpi/training/config.py
#   - create_data_loader(...) builds:
#       LeRobotDataset -> repack transforms -> dataset transforms -> normalization(optional)
#       -> model transforms (resize/tokenize/pad) -> batched loader
#   - We iterate the loader once and print shapes.
#
# Notes about key arguments used here:
#   - framework="jax"
#       OpenPI's training pipeline is JAX-first: it expects batches as numpy/JAX arrays and
#       will shard to devices for JAX training. Even though LeRobotDataset is implemented
#       with PyTorch DataLoader under the hood, OpenPI converts the batch to JAX arrays
#       when framework="jax".
#       (So: dataset format == LeRobot; training backend == JAX. These are different concepts.)
#
#   - skip_norm_stats=True
#       Skips normalization stats requirement so this smoke test can run even before you compute
#       norm stats. If you set this False, it will require config.data.norm_stats to exist
#       (computed by scripts/compute_norm_stats.py).
#
#   - num_batches=2
#       Only produce 2 batches and then stop. This prevents the script from running forever.
#
#   - shuffle=True
#       Randomizes sampling order. If debugging deterministic issues, set shuffle=False.
# =============================================================================

import openpi.training.config as cfg
from openpi.training.data_loader import create_data_loader


def main():
    # -------------------------------------------------------------------------
    # [1] Load the training config by name
    # -------------------------------------------------------------------------
    # This config must exist in openpi/training/config.py inside _CONFIGS.
    # It typically contains:
    #   - model config (pi0/pi05 action_horizon/action_dim/etc.)
    #   - dataset repo_id / local dataset settings
    #   - transforms (repack/data/model transforms)
    #   - batch_size, num_workers, and other training hyperparameters
    c = cfg.get_config("pi05_galaxea_r1lite")

    # -------------------------------------------------------------------------
    # [2] Build the data loader
    # -------------------------------------------------------------------------
    # create_data_loader returns an iterator yielding:
    #   (Observation, Actions)
    #
    # Here:
    #   shuffle=True: shuffle dataset samples
    #   num_batches=2: stop after 2 batches (smoke test)
    #   framework="jax": convert output batch to JAX arrays (for JAX training loop)
    #   skip_norm_stats=True: don't require normalization stats file
    loader = create_data_loader(
        c,
        shuffle=True,
        num_batches=2,
        framework="jax",
        skip_norm_stats=True,
    )

    # -------------------------------------------------------------------------
    # [3] Fetch one batch and print key shapes
    # -------------------------------------------------------------------------
    it = iter(loader)
    obs, acts = next(it)

    # Observation is an OpenPI Observation dataclass-like wrapper.
    # Convert to dict for easy inspection.
    print("obs keys:", obs.to_dict().keys())
    d = obs.to_dict()

    # "state" should be [B, state_dim], where state_dim for your Galaxea packing is 64.
    print("state:", d["state"].shape, d["state"].dtype)

    # "image" should be a dict of camera views.
    # Each view should be [B, H, W, C] after ResizeImages (e.g., 224x224) and float32 in most cases.
    print("image keys:", d["image"].keys())
    for k, v in d["image"].items():
        print("image", k, v.shape, v.dtype)

    # "acts" is the action chunk output:
    #   [B, action_horizon, model_action_dim]
    # For pi05_base: model_action_dim is usually 32, so your 23-dim action is padded to 32.
    print("actions:", acts.shape, acts.dtype)


if __name__ == "__main__":
    main()
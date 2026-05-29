"""Replay buffer dataset for continual learning via data replay.

Wraps a transformed dataset with a subset of step indices, enabling
sampling from old task data during training on new tasks. The buffer
stores only indices (not raw data), making it memory-efficient.

Supports two modes:
- "transition": samples individual steps uniformly at random
- "episode": samples whole episodes (trajectories), keeping all steps
  from selected episodes
"""

import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ReplayBufferDataset(Dataset):
    """Wraps a dataset, retaining a random subset of its indices.

    The buffer stores indices into the underlying dataset. __getitem__
    delegates to the wrapped dataset, which should already have transforms
    (normalization, repacking, etc.) applied so that returned data is
    consistent with the main training dataloader.
    """

    def __init__(
        self,
        dataset: Dataset,
        buffer_indices: list[int],
        dataset_name: str = "",
        mode: str = "transition",
        episode_indices: list | None = None,
        episode_to_steps: dict | None = None,
    ):
        self.dataset = dataset
        self.buffer_indices = buffer_indices
        self.dataset_name = dataset_name
        self.mode = mode
        self.episode_indices = episode_indices
        self.episode_to_steps = episode_to_steps

    def __len__(self) -> int:
        return len(self.buffer_indices)

    def __getitem__(self, index: int):
        step_idx = self.buffer_indices[index]
        return self.dataset[step_idx]

    def save(self, path: str, *, original_num_episodes: int | None = None) -> None:
        """Save buffer metadata to a JSON file.

        Args:
            path: Path to save the JSON file.
            original_num_episodes: Total number of episodes in the original dataset.
                Required for episode mode to enable correct downsampling later.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        meta = {
            "dataset_name": self.dataset_name,
            "mode": self.mode,
            "buffer_indices": self.buffer_indices,
            "original_dataset_size": len(self.dataset),
        }
        if self.mode == "episode" and self.episode_indices is not None:
            meta["episode_indices"] = self.episode_indices
            meta["original_num_episodes"] = original_num_episodes if original_num_episodes is not None else len(self.episode_indices)
            if self.episode_to_steps is not None:
                meta["episode_to_steps"] = {str(k): v for k, v in self.episode_to_steps.items()}
        with open(path, "w") as f:
            json.dump(meta, f)
        logger.info(f"Saved replay buffer ({len(self.buffer_indices)} steps, mode={self.mode}) to {path}")

    @staticmethod
    def load(path: str, dataset: Dataset) -> "ReplayBufferDataset":
        """Load buffer indices from a JSON file and wrap the given dataset."""
        with open(path, "r") as f:
            meta = json.load(f)
        mode = meta.get("mode", "transition")
        logger.info(f"Loaded replay buffer ({len(meta['buffer_indices'])} steps, mode={mode}) from {path}")
        return ReplayBufferDataset(
            dataset=dataset,
            buffer_indices=meta["buffer_indices"],
            dataset_name=meta.get("dataset_name", ""),
            mode=mode,
            episode_indices=meta.get("episode_indices"),
            episode_to_steps=meta.get("episode_to_steps"),
        )

    @staticmethod
    def build_buffer(
        dataset: Dataset,
        fraction: float,
        *,
        episode_data_index: dict | None = None,
        seed: int = 42,
        mode: str = "transition",
        dataset_name: str = "",
    ) -> "ReplayBufferDataset":
        """Sample from the dataset to create a new buffer.

        Args:
            dataset: The underlying dataset (should be already transformed).
            fraction: Fraction to retain. In transition mode, fraction of steps.
                      In episode mode, fraction of episodes.
            episode_data_index: Dict with "from" and "to" keys mapping episode
                indices to frame ranges. Required for episode mode.
            seed: Random seed.
            mode: "transition" for individual step sampling, "episode" for
                whole-episode sampling.
            dataset_name: Name of the dataset for metadata.
        """
        rng = np.random.default_rng(seed)

        if mode == "episode":
            if episode_data_index is None:
                raise ValueError("episode_data_index is required for episode mode")

            num_episodes = len(episode_data_index["from"])
            original_num_episodes = num_episodes
            k = max(1, int(num_episodes * fraction))
            selected_positions = sorted(rng.choice(num_episodes, size=k, replace=False).tolist())

            # Build episode_to_steps mapping and collect buffer_indices
            episode_to_steps = {}
            buffer_indices = []
            for ep_pos in selected_positions:
                ep_start = int(episode_data_index["from"][ep_pos])
                ep_end = int(episode_data_index["to"][ep_pos])
                steps = list(range(ep_start, ep_end))
                buffer_indices.extend(steps)
                episode_to_steps[str(ep_pos)] = steps

            logger.info(
                f"Built episode replay buffer: {k}/{num_episodes} episodes, "
                f"{len(buffer_indices)} steps (fraction={fraction:.4f})"
            )
            return ReplayBufferDataset(
                dataset,
                buffer_indices,
                dataset_name,
                mode="episode",
                episode_indices=[str(p) for p in selected_positions],
                episode_to_steps=episode_to_steps,
            )
        else:
            # Transition mode
            n = len(dataset)
            k = max(1, int(n * fraction))
            indices = sorted(rng.choice(n, size=k, replace=False).tolist())
            logger.info(f"Built replay buffer: {k}/{n} steps (fraction={fraction:.4f})")
            return ReplayBufferDataset(dataset, indices, dataset_name, mode="transition")

    @staticmethod
    def downsample_to_count(
        existing_indices: list[int], target_count: int, seed: int = 42
    ) -> list[int]:
        """Randomly discard entries from existing buffer down to target_count.

        Does NOT re-sample from the original dataset. Keeps a random subset
        of the already-buffered indices.
        """
        target_count = max(1, min(target_count, len(existing_indices)))
        if target_count >= len(existing_indices):
            return list(existing_indices)
        rng = np.random.default_rng(seed)
        keep_positions = sorted(rng.choice(len(existing_indices), size=target_count, replace=False).tolist())
        return [existing_indices[pos] for pos in keep_positions]

    @staticmethod
    def downsample_episodes(
        episode_indices: list,
        target_episode_count: int,
        episode_to_steps: dict,
        seed: int = 42,
    ) -> tuple[list, list[int], dict]:
        """Randomly discard whole episodes down to target_episode_count.

        Returns:
            (new_episode_indices, new_buffer_indices, new_episode_to_steps) tuple.
        """
        target_episode_count = max(1, min(target_episode_count, len(episode_indices)))
        if target_episode_count >= len(episode_indices):
            all_steps = []
            for steps in episode_to_steps.values():
                all_steps.extend(steps)
            return list(episode_indices), sorted(all_steps), dict(episode_to_steps)

        rng = np.random.default_rng(seed)
        keep_positions = sorted(rng.choice(len(episode_indices), size=target_episode_count, replace=False).tolist())
        new_episode_indices = [episode_indices[pos] for pos in keep_positions]

        # Re-expand from mapping
        buffer_indices = []
        new_episode_to_steps = {}
        for traj_id in new_episode_indices:
            key = str(traj_id)
            if key in episode_to_steps:
                buffer_indices.extend(episode_to_steps[key])
                new_episode_to_steps[key] = episode_to_steps[key]

        return new_episode_indices, sorted(buffer_indices), new_episode_to_steps

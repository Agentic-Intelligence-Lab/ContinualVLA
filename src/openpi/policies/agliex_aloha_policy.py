import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_agliex_aloha_example() -> dict:
    """Creates a random input example for the Agliex/Aloha-style policy."""
    return {
        "observation.state": np.ones((14,), dtype=np.float32),
        "observation.images.cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "observation.images.cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "observation.images.cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "prompt": "stack the bowls",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AgliexAlohaInputs(transforms.DataTransformFn):
    """Convert RoboChallenge Aloha-style LeRobot samples into OpenPI model inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation.images.cam_high"])
        left_wrist_image = _parse_image(data["observation.images.cam_left_wrist"])
        right_wrist_image = _parse_image(data["observation.images.cam_right_wrist"])

        inputs = {
            "state": np.asarray(data["observation.state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "action" in data:
            inputs["actions"] = np.asarray(data["action"], dtype=np.float32)
        elif "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        prompt = data.get("prompt")
        if prompt is None:
            prompt = data.get("task", "")
        if prompt is not None:
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class AgliexAlohaOutputs(transforms.DataTransformFn):
    """Convert model outputs back to the dataset action space."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14], dtype=np.float32)}

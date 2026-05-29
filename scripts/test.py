# import inspect
# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# print(inspect.signature(LeRobotDataset.__init__))

import os
from pathlib import Path
import traceback

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

_DATA_ROOT = os.environ.get("OPENPI_DATA_ROOT", "/data/datasets")
DATASET_ROOT = Path(_DATA_ROOT) / "galaxea_open_world_datasets/lerobot/Dry_Clothes_In_A_Dryer"

# Use a valid-looking repo_id to avoid HF repo id validation issues in some versions.
# root points to your local extracted dataset folder containing meta/, data/, videos/.
REPO_ID = "local/galaxea_dry_clothes"

# Try a few backends. One of them may work without system FFmpeg.
BACKENDS = ["opencv", "pyav", None]  # None = default (likely torchcodec)

def try_backend(backend):
    print(f"\n=== Trying video_backend={backend} ===")
    ds = LeRobotDataset(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
        episodes=[0],              # limit scope (faster)
        download_videos=False,      # IMPORTANT: do not try to fetch from hub
        video_backend=backend,      # switch decoding backend
    )
    x = ds[0]  # will trigger decoding of image frames
    # Print only keys and one image shape to verify we got frames.
    print("Top-level keys:", list(x.keys()))
    # Depending on LeRobot version, images may be nested or flattened keys.
    # Try both common patterns:
    if "observation" in x and isinstance(x["observation"], dict) and "images" in x["observation"]:
        imgs = x["observation"]["images"]
        k0 = next(iter(imgs.keys()))
        print("Example image key:", k0, "shape:", getattr(imgs[k0], "shape", None), "dtype:", getattr(imgs[k0], "dtype", None))
    else:
        # flattened form: "observation.images.head_rgb" etc.
        img_keys = [k for k in x.keys() if "observation.images" in k]
        print("Found flattened image keys:", img_keys[:5])
        if img_keys:
            v = x[img_keys[0]]
            print("Example image key:", img_keys[0], "shape:", getattr(v, "shape", None), "dtype:", getattr(v, "dtype", None))
    print("SUCCESS ✅")

for b in BACKENDS:
    try:
        try_backend(b)
        break
    except Exception as e:
        print("FAILED ❌", repr(e))
        traceback.print_exc(limit=2)

#!/usr/bin/env bash
set -e
uv run serve_policy.py policy:checkpoint \
  --policy.config=pi05_galaxea_r1lite_Fold_Green_Tshirt \
  --policy.dir=openpi/pi05_galaxea_r1lite_Fold_Green_Tshirt/pi05_galaxea_fold_green_Tshirt_100_chunk30/10000

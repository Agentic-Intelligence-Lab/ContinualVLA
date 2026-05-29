# ContinualVLA

**Can VLA Models Learn from Real-World Data Continually without Forgetting?**

[Jiarun Zhu](https://github.com/Agentic-Intelligence-Lab), [Yijun Hong](https://github.com/Agentic-Intelligence-Lab), [Xiaoquan Sun](https://github.com/Agentic-Intelligence-Lab), [Zetian Xu](https://github.com/Agentic-Intelligence-Lab), [Mingqi Yuan](https://github.com/Agentic-Intelligence-Lab), [Zhiyong Wang](https://github.com/Agentic-Intelligence-Lab), [Wenjun Zeng](https://github.com/Agentic-Intelligence-Lab), [Jiayu Chen](https://github.com/Agentic-Intelligence-Lab)

[![arXiv](https://img.shields.io/badge/arXiv-2605.26820-b31b1b.svg)](https://arxiv.org/abs/2605.26820)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://agentic-intelligence-lab.org/Never/pages/continualvla)

Official implementation of the paper "Can VLA Models Learn from Real-World Data Continually without Forgetting?". This codebase is built on [Physical Intelligence's openpi](https://github.com/Physical-Intelligence/openpi).

## Overview

Vision-language-action (VLA) models provide a promising foundation for general-purpose robotics. However, deploying them in the real world requires the ability to continually acquire new skills while retaining previously learned behaviors — a challenge largely unexplored under realistic conditions.

This work provides the **first empirical study of real-world continual VLA learning** and offers practical guidance for deploying long-lived robot policies.

### Key Contributions

- **Real-world continual learning dataset**: Four sequential manipulation tasks spanning rigid-object pick-and-place, contact-rich pressing, and deformable-object folding.
- **Empirical findings**: VLA models suffer significant catastrophic forgetting when continually learning from heterogeneous real-world demonstrations.
- **Systematic evaluation of experience replay**: Key implementation factors that govern the success of replay-based continual learning are identified and analyzed.

## Installation

### 1. Clone and set up the environment

```bash
git clone https://github.com/Agentic-Intelligence-Lab/ContinualVLA.git
cd ContinualVLA

# Create a clean Python 3.11+ virtual environment
python3.11 -m venv .venv && source .venv/bin/activate

# Install PyTorch with CUDA 12 support (adjust for your CUDA version: https://pytorch.org)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# Install all remaining dependencies
pip install -r requirements.txt

# Install the local workspace packages
pip install -e packages/openpi-client
pip install -e .
```

### Requirements

- Python 3.11+
- PyTorch 2.7+ with CUDA 12 support (see [pytorch.org](https://pytorch.org) for other CUDA versions)
- 8 GPUs recommended for training (tested on NVIDIA A100/H100)

### 2. Obtain pretrained Pi0.5 weights

Training fine-tunes from the Pi0.5 base model. Download the weights from HuggingFace:

```bash
bash scripts/download_pi05base.sh
```

This downloads the JAX-format checkpoint to `~/.cache/openpi/pi05_base`. Convert it to PyTorch format:

```bash
python examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir ~/.cache/openpi/pi05_base \
  --output_path ~/.cache/openpi/pi05_base_pytorch
```

Set the environment variable to point to the converted weights:

```bash
export OPENPI_PYTORCH_WEIGHT_PATH="$HOME/.cache/openpi/pi05_base_pytorch"
```

You can also [download from Google Cloud Storage](https://console.cloud.google.com/storage/browser/openpi-assets/checkpoints/pi05_base) if you have GCS access (`gs://openpi-assets/checkpoints/pi05_base/params`).

## Dataset

The real-world continual learning dataset consists of four sequential manipulation tasks collected on a Piper robotic arm:

| Task | Description | Type |
|------|-------------|------|
| Stack Bowls | Pick and stack bowls | Rigid-object pick-and-place |
| Hang Cup | Hang a cup on a rack | Articulated-object manipulation |
| Fold Towel | Fold a deformable towel | Deformable-object manipulation |
| Press Button | Press a button with precise contact | Contact-rich manipulation |

### Data directory structure

Datasets are in [LeRobot](https://github.com/huggingface/lerobot) format. Set `OPENPI_DATA_ROOT` to point to your dataset storage (defaults to `/data/datasets`). The expected directory layout:

```
$OPENPI_DATA_ROOT/
├── realworld_piper/
│   ├── stack_bowls_20260413/
│   ├── hang_cup_20260413/
│   ├── fold_towel_20260417/
│   └── press_button_20260414_trimmed/
└── ...
```

You can override the path for individual tasks by editing the `local_roots` entries in `src/openpi/training/config.py`.

## Training

Before training, set the required environment variables:

```bash
export WANDB_API_KEY="your_key"                # Weights & Biases logging
export OPENPI_DATA_ROOT="/path/to/datasets"    # Dataset root directory
export OPENPI_PYTORCH_WEIGHT_PATH="$HOME/.cache/openpi/pi05_base_pytorch"  # Pretrained weights
```

### Single-Task Training

```bash
export OPENPI_NORM_STATS_DIR="./assets/pi05_piper_stack_bowls_20260413_4cam/piper_stack_bowls_20260413_4cam"

bash scripts/train_pytorch.sh \
  --config-name pi05_piper_stack_bowls_20260413_4cam_hold_dim7_13 \
  --exp-name my_experiment \
  --gpus 0,1,2,3,4,5,6,7 \
  --nproc 8
```

### Joint (Multi-Task) Training

Train on all tasks simultaneously as a baseline:

```bash
bash scripts/train_joint.sh --steps 20000
```

### Continual Learning via Data Replay

Train sequentially across tasks with experience replay to mitigate forgetting:

```bash
bash scripts/train_cl.sh
```

Key hyperparameters (set as environment variables):
- `DATA_BUFFER_SIZE`: Fraction of data retained from old tasks (default: 0.2)
- `DATA_REPLAY_RATIO`: Per-step probability of sampling from replay buffer (default: 0.2)
- `REPLAY_MODE`: Buffer sampling mode — `"episode"` or `"transition"`

Resume from a checkpoint:

```bash
bash scripts/train_cl_resume.sh
```

### Configuration

Training configurations are defined in `src/openpi/training/config.py`. Key config names:

- `pi05_piper_stack_bowls_20260413_4cam_hold_dim7_13`
- `pi05_piper_hang_cup_20260413_4cam_hold_dim7_13`
- `pi05_piper_fold_towel_20260417_4cam_hold_dim7_13`
- `pi05_piper_press_button_20260414_4cam_hold_dim7_13`
- `pi05_piper_4tasks_joint` (joint training baseline)

### Computing Normalization Statistics

Norm stats are pre-computed and stored in `assets/`. To recompute:

```bash
python scripts/compute_norm_stats.py --config-name <config_name>
```

You can also use a shared norm stats file across tasks by setting `OPENPI_NORM_STATS_DIR`.

## Project Structure

```
├── scripts/              # Training and utility scripts
│   ├── train_pytorch.sh  # Main PyTorch training launcher
│   ├── train_cl.sh       # Continual learning loop
│   ├── train_joint.sh    # Joint multi-task baseline
│   └── train_cl_resume.sh
├── src/openpi/
│   ├── models/           # Model implementations (Pi0, Pi0-FAST)
│   ├── models_pytorch/   # PyTorch model implementations
│   ├── policies/         # Robot-specific policy transforms
│   ├── training/         # Training loop, config, data loading, replay buffer
│   ├── serving/          # Policy server for deployment
│   └── shared/           # Shared utilities
├── assets/               # Normalization statistics
└── third_party/          # External dependencies (ALOHA, LIBERO)
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhu2026continualvla,
  title={Can VLA Models Learn from Real-World Data Continually without Forgetting?},
  author={Zhu, Jiarun and Hong, Yijun and Sun, Xiaoquan and Xu, Zetian and
          Yuan, Mingqi and Wang, Zhiyong and Zeng, Wenjun and Chen, Jiayu},
  journal={arXiv preprint arXiv:2605.26820},
  year={2026}
}
```

## License

This project is released under the [MIT License](LICENSE).

The codebase builds on [openpi](https://github.com/Physical-Intelligence/openpi) by Physical Intelligence, which is also MIT-licensed. The Gemma model weights are subject to their own [license terms](LICENSE_GEMMA.txt).

## Acknowledgements

This work is supported by the Agentic Intelligence Lab (AIL) at The University of Hong Kong. We thank the Physical Intelligence team for open-sourcing the openpi codebase.

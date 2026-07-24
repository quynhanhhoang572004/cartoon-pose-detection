# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository adapts **PoseAnything** — a few-shot Category-Agnostic Pose Estimation (CAPE) model — for carton/object pose detection. The model uses a Graph Transformer Decoder to localize keypoints on query images given only a support image with annotated keypoints (1-shot or 5-shot).

All commands below are run from inside the `PoseAnything/` subdirectory.

## Setup

```bash
# Install the local models package in editable mode (required before any training/testing)
cd PoseAnything
python setup.py develop
```

**Dependencies:** Python 3.8, PyTorch 2.0.1, CUDA 12.1, `mmcv-full==1.6.2`, `mmpose==0.29.0`. See `docker/Dockerfile` for the exact install sequence.

**Docker (recommended):**
```bash
docker pull orhir/pose_anything
docker run --name pose_anything -v {DATA_DIR}:/workspace/PoseAnything/PoseAnything/data/mp100 -it orhir/pose_anything /bin/bash
```

## Common Commands

**Train:**
```bash
python train.py --config configs/1shots/graph_split1_config.py --work-dir work_dirs/split1
```

**Evaluate:**
```bash
python test.py configs/1shots/graph_split1_config.py path/to/checkpoint.pth
```
Results are also appended to `work_dirs/testing_log.txt`.

**Terminal demo** (interactive keypoint annotation via OpenCV):
```bash
python demo.py --support examples/dog1.png --query examples/dog2.png \
  --config configs/demo_b.py --checkpoint path/to/checkpoint.pth
```

**Gradio demo:**
```bash
pip install gradio==3.44.0
python app.py --checkpoint path/to/checkpoint.pth
```

**Fix CarFusion dataset filename errors:**
```bash
python tools/fix_carfusion.py path/to/CarFusion path/to/mp100_annotations
```

## Architecture

### Model Flow
The model (`PoseAnythingModel` in `models/models/detectors/pam.py`) is registered with MMPose's `POSENETS` registry. It receives:
- **Support images** (`img_s`): 1 or 5 reference images with known keypoints and skeleton
- **Query image** (`img_q`): target image to localize keypoints on

Forward pass: `extract_features` → `PoseHead` (proposal generation + Graph Transformer) → coordinate predictions.

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `PoseAnythingModel` | `models/models/detectors/pam.py` | Top-level model; handles backbone selection and the support/query split |
| `PoseHead` | `models/models/keypoint_heads/head.py` | Graph Transformer Decoder; generates keypoint proposals from support tokens, refines via cross-attention |
| `EncoderDecoder` | `models/models/utils/encoder_decoder.py` | Transformer backbone with `ProposalGenerator` for initial keypoint coordinate proposals |
| `TransformerPoseDataset` | `models/datasets/datasets/mp100/transformer_dataset.py` | Train/val dataset; dynamically pairs support+query images per episode |
| `TestPoseDataset` | `models/datasets/datasets/mp100/test_dataset.py` | Fixed-episode test dataset for reproducible evaluation |

### Backbone Options
Controlled by the `pretrained` field in config:
- `pretrained/swinv2_*.pth` → Swin Transformer V2 (default, best performance)
- `"dino"` / `"dinov2"` → loads from `torch.hub` (facebookresearch repos)
- `"resnet"` → ResNet-50 via torchvision

### Config Structure
Configs in `configs/` follow the naming pattern `{N}shot{-swin}/graph_split{1-5}_config.py`:
- `1shots/` — 1-shot with Tiny Swin backbone
- `1shot-swin/` — 1-shot with Small Swin backbone (stronger)
- `5shots/`, `5shot-swin/` — 5-shot equivalents
- Each config fully defines model architecture, optimizer, data pipeline, and dataset paths

The data root is hardcoded as `data/mp100` relative to the `PoseAnything/` directory. Annotation files are at `data/mp100/annotations/mp100_split{N}_{train|val|test}.json`.

### MMPose Integration
The `models/` package extends MMPose via its registry system. Custom classes (`PoseAnythingModel`, `PoseHead`, `TransformerPoseDataset`, etc.) are registered and resolved by MMPose's `build_posenet`/`build_dataset` when configs specify `type=`. The wildcard import `from models import *` in `train.py`/`test.py` triggers all registrations.

### Skeleton Representation
Keypoint connectivity is passed as a list of index tuples (e.g., `[(0,1), (1,2)]`) in `img_metas` under `sample_skeleton` and `query_skeleton`. The Graph Transformer uses this to structure cross-keypoint attention in `PoseHead`.
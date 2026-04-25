# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ReconDrive** is a feed-forward 4D Gaussian Splatting framework for autonomous driving scene reconstruction. It extends the [VGGT](https://github.com/facebookresearch/vggt) vision foundation model to generate 4D Gaussians from multi-camera sequences (6 nuScenes cameras), enabling novel-view synthesis at arbitrary camera positions and times.

## Environment Setup

```bash
conda create -n recondrive python=3.10
conda activate recondrive
pip install -r requirements.txt

# Manual installs required:
# PyTorch3D 0.7.8 (CUDA 12.1):
wget https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py310_cu121_pyt231.tar.bz2
conda install ./pytorch3d-0.7.8-py310_cu121_pyt231.tar.bz2

# Gaussian Splatting submodules:
git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive
cd gaussian-splatting && pip install submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim

# SAM2:
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 && pip install -e .
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

## Common Commands

**Training (Stage 2, multi-GPU):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train.sh 8
```
This requires `./checkpoints/recondrive_stage1.ckpt` (Stage 1 pretrained checkpoint from HuggingFace: `TuojingAI/ReconDrive`).

**Evaluation / Inference (single GPU):**
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/inference.sh
```
This requires `./checkpoints/recondrive_stage2.ckpt`. Outputs to `./work_dirs/recondrive_stage2_eval_output/`.

**Direct script invocation:**
```bash
# Training
python -m scripts.trainer --cfg_path=./configs/nuscenes/recondrive.yaml --train_4d --devices=8 --pretrained_ckpt=./checkpoints/recondrive_stage1.ckpt

# Inference
python -m scripts.inference --cfg_path=./configs/nuscenes/recondrive.yaml --restore_ckpt=./checkpoints/recondrive_stage2.ckpt --output_dir=./work_dirs/output --device=0 --novel_distances=1.0,2.0,3.0 --eval_resolution=280x518
```

## Architecture

### Training Pipeline

```
scripts/trainer.py
  └─ ReconDrive_LITModelModule  (PyTorch Lightning Module)
      ├─ ReconDriveModel  (torch.nn.Module)
      │   ├─ VGGT foundation model  (models/vggt/)
      │   │   ├─ Aggregator  (feature extraction with LoRA fine-tuning)
      │   │   └─ Heads: camera, depth (DPT), Gaussian (GS-DPT), track
      │   └─ Separate depth head + Gaussian splatting head
      ├─ gsplat differentiable renderer
      ├─ SAM2 vehicle segmentation (optional, for motion modeling)
      └─ LPIPS perceptual loss
```

### Key Files

| File | Role |
|------|------|
| `models/recondrive_model.py` | `ReconDriveModel` and `ReconDrive_LITModelModule` — core model and Lightning wrapper |
| `models/vggt/models/vggt.py` | VGGT foundation model |
| `models/vggt/heads/gs_dpt_head.py` | DPT-based Gaussian parameter prediction head |
| `models/gaussian_util.py` | Gaussian rendering utilities (`render`, `depth2pc`, `focal2fov`) |
| `models/loss_util.py` | Loss functions (photometric, SSIM, edge smoothness) |
| `dataset/vggt4dgs_dataset.py` | nuScenes 4D dataset loader (`NuScenesdataset4D`) |
| `dataset/vggt4dgs_data_module.py` | PyTorch Lightning DataModule |
| `configs/nuscenes/recondrive.yaml` | Primary configuration for all training/eval hyperparameters |

### Model Forward Pass

`ReconDriveModel.forward(images)` takes `[B, V, 3, H, W]` (batch × 6 views × RGB × height × width) and outputs: depth maps, Gaussian rotations, scales, opacity, spherical harmonics coefficients (SH degree 4), and flow/velocity fields.

### Loss Components

- **Photometric loss**: `0.85 * SSIM + 0.15 * L1` between rendered and target images
- **Projection loss** (`lambda_project=1.0`): multi-scale photometric loss
- **Edge smoothness loss** (`lambda_edge=0.1`): edge-aware depth regularization
- **Depth loss** (`lambda_depth=0.001`): depth supervision
- **Gaussian property losses** (`lambda_gaussian=2`, `lambda_scale=0.01`, `lambda_opacity=0.01`): MSE on Gaussian parameters

### Training Strategy

- Two-stage training: Stage 1 pre-trains the full model; Stage 2 fine-tunes with two-frame inputs
- VGGT weights are frozen; only depth/Gaussian heads and LoRA-adapted aggregator are trained (`freeze_parameters_except_heads()`)
- Multi-GPU DDP with `ddp_find_unused_parameters_true` strategy
- Gradient accumulation: 8 steps, batch size 2 per GPU

## Data

- Dataset: nuScenes with 12Hz interpolated annotations (`interp_12Hz_trainval` version from UniScene)
- Expected path: `./data/nuscenes/` (symlink supported)
- 6 cameras: `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`, `CAM_BACK`
- Input resolution: 280×518; context span: 6 frames

## Checkpoints

Pretrained checkpoints on HuggingFace (`TuojingAI/ReconDrive`):
- Stage 1: `recondrive_stage1.ckpt` (MD5: `a429e3a3ea03d0bab1579d099cfff2c8`)
- Stage 2: `recondrive_stage2.ckpt` (MD5: `fd6ed379f136b3c17d0e27a5aec8c0b7`)

Place or symlink checkpoints to `./checkpoints/`. VGGT base checkpoint goes to `./checkpoints/vggt.pt`.

## Training Outputs

Training artifacts are written to `./work_dirs/recondrive_training/`:
- `ckpt/` — model checkpoints
- `log/` — TensorBoard logs
- `code/` — code snapshot for reproducibility
- `cfg.yaml` — config copy

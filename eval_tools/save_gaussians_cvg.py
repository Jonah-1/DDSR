#!/usr/bin/env python3
#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

"""
逐帧保存 CVG 场景的 3D Gaussian PLY 文件（标准 3DGS 格式）。

输出目录结构：
  <output_dir>/<scene_name>/sample_<idx>/
      gaussians.ply   — 标准 3DGS PLY（SuperSplat / SIBR / gsplat 可读）

用法示例：
  # 保存 scene_000 所有帧，仅 cam-0
  python scripts/save_gaussians_cvg.py \
      --cfg_path configs/cvg/recondrive_cvg_1cam.yaml \
      --restore_ckpt checkpoints/recondrive_stage2.ckpt \
      --scene scene_000

  # 保存 scene_001，全部相机，每6帧
  python eval_tools/save_gaussians_cvg.py \
      --cfg_path configs/cvg/recondrive_cvg_1cam.yaml \
      --restore_ckpt /home/yuchi.zuo/FeedforwardGS-RD/ReconDrive/work_dirs/cvg_cam_far_training/ckpt/best_module.ckpt \
      --scene scene_000 \
      --cam_ids all --frame_skip 6

  # 保存指定帧列表
  python eval_tools/save_gaussians_cvg.py \
      --cfg_path configs/cvg/recondrive_cvg_1cam.yaml \
      --restore_ckpt checkpoints/recondrive_stage2.ckpt \
      --scene scene_000 \
      --frames 0,6,12,18,24

注意：CVG 数据集启用 use_relative_pose=True，所有帧高斯均已统一到
      场景 frame-0 ego 坐标系，无需额外对齐。
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "models"))

from dataset.cvg_scene_data_module import CVGSceneSequence
from dataset.cvg_data_module import cvg_collate_fn
from models.recondrive_model import ReconDrive_LITModelModule


def transform_xyz_to_global(xyz: torch.Tensor, ego_pose: torch.Tensor) -> torch.Tensor:
    """将 xyz (N,3) 从当前 sample 的自车坐标系变换到全局第0帧坐标系。"""
    ones = torch.ones(xyz.shape[0], 1, dtype=xyz.dtype, device=xyz.device)
    xyz_h = torch.cat([xyz, ones], dim=-1)
    return (ego_pose.to(xyz) @ xyz_h.T).T[:, :3]


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class SceneSampleDataset(Dataset):
    def __init__(self, indices, dataset, scene_idx):
        self.sample_indices = indices
        self.dataset = dataset
        self.scene_idx = scene_idx

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(self.sample_indices[idx], self.scene_idx)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, model_cfg, device):
    print(f"Loading model: {checkpoint_path}")
    model_cfg.setdefault('batch_size', 1)
    model = ReconDrive_LITModelModule(cfg=model_cfg, save_dir='./temp_log', logger=None)
    model.load_pretrained_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    return model


def to_device(data, device):
    if isinstance(data, dict):
        return {k: (v if k == 'vehicle_annotations' else to_device(v, device))
                for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device) for x in data)
    elif torch.is_tensor(data):
        return data.to(device)
    return data


# ---------------------------------------------------------------------------
# PLY save（标准 3DGS 格式）
# ---------------------------------------------------------------------------

def save_gaussians_ply(gaussians: dict, save_path: str):
    """保存标准 3DGS PLY：scale=log(s), opacity=logit(o), f_dc/f_rest channel-major。"""
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        print("  [ERROR] plyfile not installed: pip install plyfile")
        return

    C0 = 0.28209479177387814

    xyz     = gaussians['xyz'].float()
    rot     = gaussians['rot'].float().numpy()
    scale   = gaussians['scale'].float().numpy()
    opacity = gaussians['opacity'].float().numpy()
    sh      = gaussians['sh'].float()
    N, K, _ = sh.shape

    f_dc   = sh[:, 0, :].numpy()
    f_rest = sh[:, 1:, :].permute(0, 2, 1).reshape(N, -1).numpy()

    xyz_np  = xyz.numpy()
    normals = np.zeros_like(xyz_np)

    scale_raw   = np.log(np.clip(scale, 1e-9, None))
    opacity_raw = np.log(np.clip(opacity, 1e-9, 1.0 - 1e-9) /
                         (1.0 - np.clip(opacity, 1e-9, 1.0 - 1e-9)))

    attr_names  = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    attr_names += [f'f_dc_{i}'   for i in range(3)]
    attr_names += [f'f_rest_{i}' for i in range(f_rest.shape[1])]
    attr_names += ['opacity']
    attr_names += [f'scale_{i}'  for i in range(3)]
    attr_names += [f'rot_{i}'    for i in range(4)]

    dtype    = [(a, 'f4') for a in attr_names]
    attrs    = np.concatenate([xyz_np, normals, f_dc, f_rest, opacity_raw, scale_raw, rot], axis=1)
    elements = np.empty(N, dtype=dtype)
    elements[:] = list(map(tuple, attrs))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    PlyData([PlyElement.describe(elements, 'vertex')]).write(save_path)
    print(f"      → PLY: {N:,} Gaussians  {save_path}")


def save_sky_gaussians_ply(model, sky_color_sum: torch.Tensor,
                           sky_color_count: torch.Tensor, save_path: str):
    """保存天空高斯球为标准 3DGS PLY，颜色为所有帧 MLP 预测的平均值，坐标在自车系（原点）。"""
    if not getattr(model, 'use_sky_model', False) or model.sky_model is None:
        print("  [SKIP] sky model not enabled")
        return

    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        print("  [ERROR] plyfile not installed: pip install plyfile")
        return

    C0  = 0.28209479177387814
    sky = model.sky_model

    # 只保留至少被观测一次的点
    seen = sky_color_count > 0
    if seen.sum() == 0:
        print("  [SKIP] sky: no points were projected in any frame")
        return

    avg_rgb = (sky_color_sum[seen] / sky_color_count[seen].unsqueeze(-1)).clamp(0, 1)  # (M, 3)

    xyz_np      = sky.bg_pcd.float().cpu()[seen].numpy()
    scale_raw   = sky.bg_scales.float().cpu()[seen].numpy()
    opacity_raw = sky.bg_opacity.float().cpu()[seen].clamp(1e-9, 1 - 1e-9)
    opacity_raw = torch.log(opacity_raw / (1 - opacity_raw)).numpy()
    rot         = sky.bg_quat.float().cpu()[seen].numpy()

    N       = xyz_np.shape[0]
    normals = np.zeros_like(xyz_np)
    f_dc    = ((avg_rgb - 0.5) / C0).numpy()
    n_rest  = (model.sh_degree + 1) ** 2 - 1
    f_rest  = np.zeros((N, 3 * n_rest), dtype=np.float32)

    attr_names  = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    attr_names += [f'f_dc_{i}'   for i in range(3)]
    attr_names += [f'f_rest_{i}' for i in range(f_rest.shape[1])]
    attr_names += ['opacity']
    attr_names += [f'scale_{i}'  for i in range(3)]
    attr_names += [f'rot_{i}'    for i in range(4)]

    dtype    = [(a, 'f4') for a in attr_names]
    attrs    = np.concatenate([xyz_np, normals, f_dc, f_rest, opacity_raw, scale_raw, rot], axis=1)
    elements = np.empty(N, dtype=dtype)
    elements[:] = list(map(tuple, attrs))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    PlyData([PlyElement.describe(elements, 'vertex')]).write(save_path)
    print(f"  → Sky PLY: {N:,} Gaussians  {save_path}")


# ---------------------------------------------------------------------------
# Gaussian extraction（含 per-camera eps2d scale boost）
# ---------------------------------------------------------------------------

def extract_gaussians(recontrast_data: dict, h: int, w: int,
                      num_cams: int = 1, render_data: dict = None,
                      cam_ids: list = None, eps2d: float = 0.3) -> dict:
    """提取指定相机的高斯，并做 eps2d scale boost。

    Args:
        cam_ids : 要提取的相机列表（默认 [0]）
    """
    if cam_ids is None:
        cam_ids = [0]

    N_total        = recontrast_data['xyz'].shape[1]
    mid_point      = N_total // 2
    points_per_cam = mid_point // num_cams

    all_xyz, all_rot, all_scale, all_opacity, all_sh = [], [], [], [], []
    cam0_pos = None

    for cam_id in cam_ids:
        s0 = cam_id * points_per_cam
        e0 = s0 + points_per_cam
        s1 = mid_point + s0
        e1 = s1 + points_per_cam

        def _cat(key, _s0=s0, _e0=e0, _s1=s1, _e1=e1):
            a = recontrast_data[key][0, _s0:_e0]
            b = recontrast_data[key][0, _s1:_e1]
            return torch.cat([a, b], dim=0).cpu()

        xyz   = _cat('xyz_transformed')
        scale = _cat('scale_maps')
        rot_data     = _cat('rot_maps')
        opacity_data = _cat('opacity_maps')
        sh_data      = _cat('sh_maps')

        # 过滤天空和车头区域的高斯点
        n = xyz.shape[0]
        valid = torch.ones(n, dtype=torch.bool)
        if 'sky_mask_flat' in recontrast_data:
            sky_mask = _cat('sky_mask_flat')
            valid &= (sky_mask < 0.5)
        if 'hoodline_mask_flat' in recontrast_data:
            hood_mask = _cat('hoodline_mask_flat')
            valid &= (hood_mask < 0.5)
        if not valid.all():
            xyz          = xyz[valid]
            scale        = scale[valid]
            rot_data     = rot_data[valid]
            opacity_data = opacity_data[valid]
            sh_data      = sh_data[valid]

        if render_data is not None:
            try:
                e2c = render_data[('e2c_extr', 0, cam_id)][0].cpu().float()
                K   = render_data[('K',        0, cam_id)][0].cpu().float()
                fx  = K[0, 0].item()

                c2e     = torch.linalg.inv(e2c)
                cam_pos = c2e[:3, 3]
                if cam_id == 0:
                    cam0_pos = cam_pos

                ones    = torch.ones(xyz.shape[0], 1)
                xyz_h   = torch.cat([xyz, ones], dim=-1)
                xyz_cam = (e2c[:3] @ xyz_h.T).T
                depth   = xyz_cam[:, 2].clamp(min=0.01)
                min_sc  = (eps2d * depth / fx).unsqueeze(-1).expand_as(scale)
                scale   = torch.max(scale, min_sc)
            except Exception as exc:
                print(f"      [WARN] cam {cam_id} eps2d boost skipped: {exc}")

        all_xyz.append(xyz)
        all_rot.append(rot_data)
        all_scale.append(scale)
        all_opacity.append(opacity_data)
        all_sh.append(sh_data)

    return {
        'xyz':     torch.cat(all_xyz,     dim=0),
        'rot':     torch.cat(all_rot,     dim=0),
        'scale':   torch.cat(all_scale,   dim=0),
        'opacity': torch.cat(all_opacity, dim=0),
        'sh':      torch.cat(all_sh,      dim=0),
        'cam_pos': cam0_pos,
    }


# ---------------------------------------------------------------------------
# 场景推理与高斯保存
# ---------------------------------------------------------------------------

def _inference_collate_fn(batch):
    """CVG 推理用的 collate：保留 all_dict / context_frames 嵌套结构，模型需要它们。"""
    if not batch:
        return {}
    return cvg_collate_fn(batch)


def process_scene(model, scene_path, device, output_dir,
                  cam_ids=None, frame_skip=1, frame_indices=None,
                  max_dist=80.0, save_merged=False):
    """对单个 CVG 场景运行推理并保存 Gaussian PLY。"""
    scene_start = time.time()
    scene_name  = os.path.basename(scene_path)

    h         = getattr(model, 'height',   280)
    w         = getattr(model, 'width',    518)
    num_cams  = getattr(model, 'num_cams',   1)
    sh_degree = getattr(model, 'sh_degree',  4)

    # ----- 建立数据集 -----
    print(f"  Loading scene: {scene_path}")
    scene_seq = CVGSceneSequence(
        scene_path=scene_path,
        image_source='undistorted',
        context_span=model.context_span if hasattr(model, 'context_span') else 6,
        height=h, width=w,
        with_mask=True, use_relative_pose=True,
    )
    scene_length = len(scene_seq)

    # ----- 确定要处理的帧 -----
    all_indices = list(range(scene_length))
    if frame_indices is not None:
        keep = [i for i in frame_indices if i < scene_length]
    elif frame_skip > 1:
        keep = list(range(0, scene_length, frame_skip))
    else:
        keep = all_indices

    cam_label = ','.join(str(c) for c in cam_ids) if cam_ids else '0'
    print(f"  {scene_name}: {len(keep)}/{scene_length} frames  "
          f"cams=[{cam_label}]  max_dist={max_dist}m")

    scene_dataset = SceneSampleDataset(keep, scene_seq, scene_idx=0)
    scene_loader  = DataLoader(
        scene_dataset, batch_size=1, shuffle=False,
        num_workers=0, drop_last=False,
        collate_fn=_inference_collate_fn,
    )

    saved = 0
    # 用于合并：按属性分别累积，避免一次性占用大量内存
    merged_attrs = {k: [] for k in ('xyz', 'rot', 'scale', 'opacity', 'sh')}
    merged_cam_pos = None

    # 天空颜色累积（所有帧平均）
    use_sky = getattr(model, 'use_sky_model', False) and model.sky_model is not None
    if use_sky:
        N_sky = model.sky_model.bg_pcd.shape[0]
        sky_color_sum   = torch.zeros(N_sky, 3, dtype=torch.float32)
        sky_color_count = torch.zeros(N_sky,    dtype=torch.float32)
    for batch_count, batch_data in enumerate(scene_loader):
        actual_idx = keep[batch_count]

        batch_data = to_device(batch_data, device)

        with torch.no_grad():
            output = model.predict_step(batch_data, batch_idx=0)

        if not isinstance(output, tuple):
            print(f"    sample {actual_idx:04d}: unexpected output, skip")
            continue

        recontrast_data, render_data, _splat = output

        # 提取指定相机的高斯
        gaussians = extract_gaussians(
            recontrast_data, h, w,
            num_cams=num_cams,
            render_data=render_data,
            cam_ids=cam_ids,
        )

        # 变换到全局第0帧自车坐标系
        ego_pose = batch_data['all_dict']['ego_pose'][0, 0].float().cpu()  # (4, 4)
        gaussians['xyz'] = transform_xyz_to_global(gaussians['xyz'].float().cpu(), ego_pose)
        cam_pos = gaussians.get('cam_pos')
        if cam_pos is not None:
            gaussians['cam_pos'] = transform_xyz_to_global(cam_pos.unsqueeze(0), ego_pose).squeeze(0)
        cam_pos = gaussians.get('cam_pos')

        # 距离过滤
        if cam_pos is not None and max_dist > 0:
            xyz  = gaussians['xyz'].float()
            dist = torch.norm(xyz - cam_pos.float().to(xyz), dim=-1)
            mask = dist < max_dist
            n_before = xyz.shape[0]
            gaussians = {
                k: (v[mask] if (torch.is_tensor(v) and v.shape[0] == n_before) else v)
                for k, v in gaussians.items()
            }
            print(f"    sample {actual_idx:04d}: {mask.sum():,}/{n_before:,} pts within {max_dist}m")

        # 保存 PLY
        sample_dir = os.path.join(output_dir, scene_name, f'sample_{actual_idx:04d}')
        save_gaussians_ply(
            gaussians,
            os.path.join(sample_dir, 'gaussians.ply'),
        )
        saved += 1

        # 累积用于合并
        if save_merged:
            for k in ('xyz', 'rot', 'scale', 'opacity', 'sh'):
                merged_attrs[k].append(gaussians[k].float().cpu())
            if merged_cam_pos is None and cam_pos is not None:
                merged_cam_pos = cam_pos.cpu()

        # 累积天空颜色
        if use_sky:
            try:
                sky    = model.sky_model
                sdev   = next(sky.parameters()).device
                cf     = batch_data.get('context_frames', {})
                ad     = batch_data.get('all_dict', {})
                si     = cf.get(('color_aug', 0))
                se     = ad.get('c2e_extr')
                sk     = ad.get('K')
                sky_mask_raw = ad.get('sky_masks')
                if sky_mask_raw is None:
                    sky_mask_raw = cf.get('sky_masks')

                if si is not None and se is not None and sk is not None:
                    n_cam_avail = se.shape[1]
                    # 遍历所有 cam_ids，每个摄像头各自投影积累颜色
                    for cid in cam_ids:
                        if cid >= n_cam_avail:
                            continue
                        try:
                            e2c = torch.linalg.inv(se[:, cid]).to(sdev)  # (B,4,4) ego→cam
                            with torch.no_grad():
                                rgb, proj_mask, _ = sky._get_background_color(
                                    si[:, cid].to(sdev), e2c, sk[:, cid].to(sdev)
                                )
                            rgb_01 = ((rgb + 1.0) * 0.5).clamp(0.0, 1.0).float().cpu()

                            # sky_mask 过滤：只保留投影到天空像素的高斯点
                            sky_filter = None
                            if sky_mask_raw is not None:
                                try:
                                    # sky_mask_raw: (B, n_cam, 1, H, W)
                                    sm = sky_mask_raw.float().cpu()[0, cid, 0:1]  # (1, H, W)
                                    H_sm, W_sm = sm.shape[-2], sm.shape[-1]
                                    e2c_cpu = e2c[0].cpu().float()              # (4, 4)
                                    K_cpu   = sk[0, cid].cpu().float()
                                    bg      = sky.bg_pcd.float().cpu()
                                    N_bg    = bg.shape[0]
                                    xyz_h   = torch.cat([bg, torch.ones(N_bg, 1)], dim=1)
                                    xyz_cam = (e2c_cpu @ xyz_h.T).T
                                    valid_z = xyz_cam[:, 2] > 0
                                    uv      = (K_cpu @ xyz_cam[:, :3].T).T
                                    uv      = uv[:, :2] / (uv[:, 2:3] + 1e-6)
                                    u_norm  = (2 * uv[:, 0] / (W_sm - 1)) - 1
                                    v_norm  = (2 * uv[:, 1] / (H_sm - 1)) - 1
                                    in_frame = ((u_norm >= -1) & (u_norm <= 1) &
                                                (v_norm >= -1) & (v_norm <= 1))
                                    grid    = torch.stack([u_norm, v_norm], dim=1).view(1, 1, N_bg, 2)
                                    sampled = torch.nn.functional.grid_sample(
                                        sm.unsqueeze(0), grid,
                                        align_corners=False, mode='bilinear', padding_mode='zeros'
                                    ).reshape(N_bg)
                                    sky_filter = valid_z & in_frame & (sampled > 0.5)
                                except Exception as ex:
                                    print(f"    [WARN] cam{cid} sky_mask filter failed: {ex}")

                            if sky_filter is not None:
                                pm_cpu = proj_mask.cpu()
                                within_proj = sky_filter[pm_cpu]
                                sky_indices = torch.where(pm_cpu)[0][within_proj]
                                sky_color_sum[sky_indices]   += rgb_01[within_proj]
                                sky_color_count[sky_indices] += 1.0
                            else:
                                sky_color_sum[proj_mask.cpu()]   += rgb_01
                                sky_color_count[proj_mask.cpu()] += 1.0
                        except Exception as ec:
                            print(f"    [WARN] cam{cid} sky accumulation failed: {ec}")
            except Exception as e:
                print(f"    [WARN] sky color accumulation failed: {e}")

        del recontrast_data, render_data, _splat, output, gaussians
        torch.cuda.empty_cache()

    # ----- 保存合并高斯 -----
    if save_merged and len(merged_attrs['xyz']) > 0:
        merged = {k: torch.cat(v, dim=0) for k, v in merged_attrs.items()}
        merged['cam_pos'] = merged_cam_pos
        merged_path = os.path.join(output_dir, scene_name, 'merged_gaussians.ply')
        total_pts = merged['xyz'].shape[0]
        print(f"  Saving merged gaussians: {total_pts:,} pts → {merged_path}")
        save_gaussians_ply(merged, merged_path)
        del merged

    # ----- 保存天空高斯球（所有帧平均颜色，第一帧自车系）-----
    if use_sky:
        sky_path = os.path.join(output_dir, scene_name, 'sky_gaussians.ply')
        save_sky_gaussians_ply(model, sky_color_sum, sky_color_count, sky_path)

    elapsed = time.time() - scene_start
    print(f"  {scene_name}: saved {saved} frames  ({elapsed:.1f}s)")
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='CVG 场景高斯保存：逐帧保存 3D Gaussian PLY'
    )
    # 必选
    parser.add_argument('--cfg_path',     required=True,  help='YAML 配置文件路径')
    parser.add_argument('--restore_ckpt', required=True,  help='模型 checkpoint 路径')
    parser.add_argument('--scene',        required=True,
                        help='场景名或编号，如 "scene_000" 或 "0"；'
                             '传 "all" 处理所有场景')

    # 可选
    parser.add_argument('--output_dir',  default=None,
                        help='输出根目录（默认 <save_dir>/gaussians）')
    parser.add_argument('--device',      default=None,
                        help='CUDA 设备，如 "0" 或 "cuda:0"')

    # 相机选项
    parser.add_argument('--cam_ids',  type=str, default='0',
                        help='相机索引，逗号分隔（默认 "0"；传 "all" = 全部）')

    # 帧选项
    parser.add_argument('--frame_skip', type=int,  default=1,
                        help='每隔 N 帧取一帧（默认 1 = 全部帧）')
    parser.add_argument('--frames',     type=str,  default=None,
                        help='指定帧索引，逗号分隔，优先级高于 --frame_skip，如 "0,6,12,18"')

    # 过滤
    parser.add_argument('--max_dist',    type=float, default=80.0,
                        help='过滤距相机超过此距离（米）的高斯（默认 80；0 = 不过滤）')

    # 合并
    parser.add_argument('--save_merged', action='store_true',
                        help='额外保存场景级合并高斯 merged_gaussians.ply')

    args = parser.parse_args()

    # ----- 配置 -----
    print(f"Config: {args.cfg_path}")
    with open(args.cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['model_cfg']['batch_size'] = 1
    config['data_cfg']['batch_size']  = 1
    if 'context_span' in config['data_cfg']:
        config['model_cfg']['context_span'] = config['data_cfg']['context_span']

    # ----- 设备 -----
    if args.device:
        device = args.device if args.device.startswith('cuda:') else f"cuda:{args.device}"
    elif config.get('devices'):
        device = f"cuda:{config['devices'][0]}"
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ----- 输出目录 -----
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(config.get('save_dir', './work_dirs/default')), '3d_output', 'gaussian'
        )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output: {args.output_dir}")

    # ----- 解析相机列表 -----
    if args.cam_ids.strip().lower() == 'all':
        num_cams = config['model_cfg'].get('num_cams', 1)
        cam_ids  = list(range(num_cams))
        print(f"Cameras: all ({num_cams})")
    else:
        cam_ids = [int(x.strip()) for x in args.cam_ids.split(',')]
        print(f"Cameras: {cam_ids}")

    # ----- 解析帧列表 -----
    frame_indices = None
    if args.frames:
        frame_indices = [int(x.strip()) for x in args.frames.split(',')]
        print(f"Frames (explicit): {frame_indices}")
    elif args.frame_skip > 1:
        print(f"Frame skip: every {args.frame_skip} frames")

    # ----- 确定要处理的场景路径 -----
    data_path = config['data_cfg']['data_path']

    def _resolve_scene_dir(scene_name: str) -> str:
        """将场景名解析为绝对路径。

        兼容两种 config 写法：
          1. data_path = .../data/scene_000   （直接指向场景目录）
          2. data_path = .../data             （指向多场景根目录）
        """
        # 纯数字 → 补全为 "scene_XXX"
        if scene_name.isdigit():
            scene_name = f"scene_{int(scene_name):03d}"

        # 已经是有效路径（绝对路径 or 存在的相对路径）
        if os.path.isdir(scene_name):
            return os.path.abspath(scene_name)

        # 尝试 data_path/scene_name
        candidate = os.path.join(data_path, scene_name)
        if os.path.isdir(candidate):
            return candidate

        # data_path 本身就是该场景目录（basename 匹配）
        if os.path.basename(data_path) == scene_name and os.path.isdir(data_path):
            return data_path

        raise FileNotFoundError(
            f"Scene '{scene_name}' not found.\n"
            f"  Tried: {candidate}\n"
            f"  data_path: {data_path}"
        )

    if args.scene.strip().lower() == 'all':
        # 自动发现：data_path 下所有 scene_* 子目录
        if os.path.isdir(data_path) and any(
            d.startswith('scene_') for d in os.listdir(data_path)
        ):
            scene_dirs = sorted([
                os.path.join(data_path, d)
                for d in os.listdir(data_path)
                if d.startswith('scene_') and os.path.isdir(os.path.join(data_path, d))
            ])
        else:
            # data_path 本身是单个场景
            scene_dirs = [data_path]
        print(f"Scenes: {len(scene_dirs)} found")
    else:
        scene_dirs = [_resolve_scene_dir(args.scene.strip())]
        print(f"Scene: {scene_dirs[0]}")

    # ----- 加载模型 -----
    model = load_model(args.restore_ckpt, config['model_cfg'], device)

    # ----- 逐场景处理 -----
    t0 = time.time()
    summary = []

    for scene_path in scene_dirs:
        scene_name = os.path.basename(scene_path)
        print(f"\n{'='*50}")
        print(f"Scene: {scene_name}")

        saved = process_scene(
            model, scene_path, device,
            output_dir=args.output_dir,
            cam_ids=cam_ids,
            frame_skip=args.frame_skip,
            frame_indices=frame_indices,
            max_dist=args.max_dist,
            save_merged=args.save_merged,
        )
        summary.append({'scene': scene_name, 'saved': saved})

    # ----- 汇总 -----
    elapsed = time.time() - t0
    total_saved = sum(r['saved'] for r in summary)

    summary_data = {
        'total_scenes':  len(summary),
        'total_samples': total_saved,
        'total_time_s':  elapsed,
        'cam_ids':       args.cam_ids,
        'frame_skip':    args.frame_skip,
        'frames':        args.frames,
        'max_dist':      args.max_dist,
        'scenes':        summary,
    }
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done: {len(summary)} scenes, {total_saved} frames ({elapsed:.1f}s)")
    print(f"Summary → {summary_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

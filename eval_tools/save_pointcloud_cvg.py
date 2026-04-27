#!/usr/bin/env python3
#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

"""
逐帧保存 CVG 场景的彩色点云 PLY（标准 RGB 格式，CloudCompare / MeshLab 可读）。

点云颜色：优先用 gsplat.spherical_harmonics 按 cam-0 视向计算，
          fallback 为 DC SH 分量近似色。

输出目录结构：
  <output_dir>/<scene_name>/sample_<idx>/
      pointcloud.ply   — 标准彩色点云 PLY（xyz + rgb）

用法示例：
  # 保存 scene_000 指定帧，cam-0
  python eval_tools/save_pointcloud_cvg.py \
      --cfg_path configs/cvg/recondrive_cvg_1cam.yaml \
      --restore_ckpt checkpoints/recondrive_stage2.ckpt \
      --scene scene_000 \
      --frames 0,6,12,18,24

  # 每6帧取一帧，同时保存合并点云
  python eval_tools/save_pointcloud_cvg.py \
      --cfg_path configs/cvg/recondrive_cvg_1cam.yaml \
      --restore_ckpt checkpoints/recondrive_stage2.ckpt \
      --scene scene_000 \
      --frame_skip 6 \
      --save_merged
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
# 点云保存（标准 RGB PLY）
# ---------------------------------------------------------------------------

def save_pointcloud_ply(xyz: torch.Tensor, rgb: torch.Tensor, save_path: str):
    """保存标准彩色点云 PLY（xyz + uint8 rgb），CloudCompare / MeshLab 可直接读取。

    Args:
        xyz : [N, 3] float tensor，点坐标
        rgb : [N, 3] float tensor，颜色值范围 [0, 1]
    """
    xyz_np  = xyz.float().cpu().numpy()
    rgb_u8  = (rgb.float().cpu().clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
    N = len(xyz_np)

    try:
        from plyfile import PlyData, PlyElement
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        verts = np.empty(N, dtype=dtype)
        verts['x'], verts['y'], verts['z']         = xyz_np[:, 0], xyz_np[:, 1], xyz_np[:, 2]
        verts['red'], verts['green'], verts['blue'] = rgb_u8[:, 0], rgb_u8[:, 1], rgb_u8[:, 2]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        PlyData([PlyElement.describe(verts, 'vertex')]).write(save_path)
        print(f"      → PLY: {N:,} pts  {save_path}")
    except ImportError:
        # fallback: open3d
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_np)
            pcd.colors = o3d.utility.Vector3dVector(rgb.float().cpu().clamp(0, 1).numpy())
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            o3d.io.write_point_cloud(save_path, pcd)
            print(f"      → PLY: {N:,} pts  {save_path}")
        except ImportError:
            print("  [ERROR] plyfile 和 open3d 均未安装，无法保存点云。"
                  "  pip install plyfile  或  pip install open3d")


# ---------------------------------------------------------------------------
# Gaussian extraction + 颜色计算
# ---------------------------------------------------------------------------

def extract_gaussians(recontrast_data: dict, h: int, w: int,
                      num_cams: int = 1, render_data: dict = None,
                      cam_ids: list = None, eps2d: float = 0.3) -> dict:
    """提取指定相机的高斯，做 eps2d scale boost，返回 xyz/sh/cam_pos。"""
    if cam_ids is None:
        cam_ids = [0]

    N_total        = recontrast_data['xyz'].shape[1]
    mid_point      = N_total // 2
    points_per_cam = mid_point // num_cams

    all_xyz, all_scale, all_sh = [], [], []
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
        sh_data = _cat('sh_maps')

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
            xyz     = xyz[valid]
            scale   = scale[valid]
            sh_data = sh_data[valid]

        if render_data is not None:
            try:
                e2c = render_data[('e2c_extr', 0, cam_id)][0].cpu().float()
                K   = render_data[('K',        0, cam_id)][0].cpu().float()
                fx  = K[0, 0].item()
                c2e = torch.linalg.inv(e2c)
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
        all_scale.append(scale)
        all_sh.append(sh_data)

    return {
        'xyz':     torch.cat(all_xyz,   dim=0),
        'scale':   torch.cat(all_scale, dim=0),
        'sh':      torch.cat(all_sh,    dim=0),
        'cam_pos': cam0_pos,
    }


def gaussians_to_rgb(xyz: torch.Tensor, sh: torch.Tensor,
                     cam_pos: torch.Tensor = None, sh_degree: int = 4) -> torch.Tensor:
    """从 SH 系数计算点云颜色 [N, 3]，范围 [0, 1]。"""
    C0 = 0.28209479177387814
    if cam_pos is not None:
        try:
            from gsplat import spherical_harmonics
            cuda = torch.device('cuda') if torch.cuda.is_available() else xyz.device
            xyz_cu = xyz.to(cuda)
            sh_cu  = sh.to(cuda)
            dirs = torch.nn.functional.normalize(
                cam_pos.to(xyz_cu) - xyz_cu, dim=-1)
            rgb = (spherical_harmonics(sh_degree, dirs, sh_cu) + 0.5).clamp(0.0, 1.0).cpu()
            return rgb
        except Exception as e:
            print(f"      [WARN] SH eval failed ({e}), using DC component")
    return (0.5 + C0 * sh[:, 0, :]).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# 场景推理与点云保存
# ---------------------------------------------------------------------------

def _inference_collate_fn(batch):
    if not batch:
        return {}
    return cvg_collate_fn(batch)


def transform_xyz_to_global(xyz: torch.Tensor, ego_pose: torch.Tensor) -> torch.Tensor:
    """将 xyz (N,3) 从当前 sample 的自车坐标系变换到全局第0帧坐标系。

    ego_pose: (4,4) float tensor，即 ego_i → ego_0 的变换矩阵
    """
    ones = torch.ones(xyz.shape[0], 1, dtype=xyz.dtype, device=xyz.device)
    xyz_h = torch.cat([xyz, ones], dim=-1)          # (N, 4)
    return (ego_pose.to(xyz) @ xyz_h.T).T[:, :3]   # (N, 3)


def process_scene(model, scene_path, device, output_dir,
                  cam_ids=None, frame_skip=1, frame_indices=None,
                  max_dist=80.0, save_merged=False):
    """对单个 CVG 场景运行推理并保存彩色点云 PLY。"""
    scene_start = time.time()
    scene_name  = os.path.basename(scene_path)

    h         = getattr(model, 'height',   280)
    w         = getattr(model, 'width',    518)
    num_cams  = getattr(model, 'num_cams',   1)
    sh_degree = getattr(model, 'sh_degree',  4)

    print(f"  Loading scene: {scene_path}")
    scene_seq = CVGSceneSequence(
        scene_path=scene_path,
        image_source='undistorted',
        context_span=model.context_span if hasattr(model, 'context_span') else 6,
        height=h, width=w,
        with_mask=True, use_relative_pose=True,
    )
    scene_length = len(scene_seq)

    # 确定要处理的帧
    if frame_indices is not None:
        keep = [i for i in frame_indices if i < scene_length]
    elif frame_skip > 1:
        keep = list(range(0, scene_length, frame_skip))
    else:
        keep = list(range(scene_length))

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
    merged_xyz_list = []
    merged_rgb_list = []

    for batch_count, batch_data in enumerate(scene_loader):
        actual_idx = keep[batch_count]

        batch_data = to_device(batch_data, device)
        with torch.no_grad():
            output = model.predict_step(batch_data, batch_idx=0)

        if not isinstance(output, tuple):
            print(f"    sample {actual_idx:04d}: unexpected output, skip")
            continue

        recontrast_data, render_data, _splat = output

        # 提取高斯（只需 xyz / sh / cam_pos）
        gauss = extract_gaussians(
            recontrast_data, h, w,
            num_cams=num_cams,
            render_data=render_data,
            cam_ids=cam_ids,
        )

        xyz     = gauss['xyz'].float().cpu()
        sh      = gauss['sh'].float().cpu()
        cam_pos = gauss['cam_pos']

        # 变换到全局第0帧自车坐标系
        ego_pose = batch_data['all_dict']['ego_pose'][0, 0].float().cpu()  # (4, 4)
        xyz = transform_xyz_to_global(xyz, ego_pose)
        if cam_pos is not None:
            cam_pos = transform_xyz_to_global(cam_pos.unsqueeze(0), ego_pose).squeeze(0)

        # 距离过滤
        if cam_pos is not None and max_dist > 0:
            dist     = torch.norm(xyz - cam_pos.float().to(xyz), dim=-1)
            mask     = dist < max_dist
            n_before = xyz.shape[0]
            xyz      = xyz[mask]
            sh       = sh[mask]
            print(f"    sample {actual_idx:04d}: {mask.sum():,}/{n_before:,} pts within {max_dist}m")

        # 计算颜色
        rgb = gaussians_to_rgb(xyz, sh, cam_pos, sh_degree=sh_degree)

        # 保存逐帧点云
        sample_dir = os.path.join(output_dir, scene_name, f'sample_{actual_idx:04d}')
        save_pointcloud_ply(xyz, rgb, os.path.join(sample_dir, 'pointcloud.ply'))
        saved += 1

        # 累积用于合并
        if save_merged:
            merged_xyz_list.append(xyz.cpu())
            merged_rgb_list.append(rgb.cpu())

        del recontrast_data, render_data, _splat, output, gauss
        torch.cuda.empty_cache()

    # 保存合并点云
    if save_merged and len(merged_xyz_list) > 0:
        merged_xyz = torch.cat(merged_xyz_list, dim=0)
        merged_rgb = torch.cat(merged_rgb_list, dim=0)
        merged_path = os.path.join(output_dir, scene_name, 'merged_pointcloud.ply')
        print(f"  Saving merged pointcloud: {merged_xyz.shape[0]:,} pts → {merged_path}")
        save_pointcloud_ply(merged_xyz, merged_rgb, merged_path)

    elapsed = time.time() - scene_start
    print(f"  {scene_name}: saved {saved} frames  ({elapsed:.1f}s)")
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='CVG 场景点云保存：逐帧保存标准彩色点云 PLY'
    )
    # 必选
    parser.add_argument('--cfg_path',     required=True,  help='YAML 配置文件路径')
    parser.add_argument('--restore_ckpt', required=True,  help='模型 checkpoint 路径')
    parser.add_argument('--scene',        required=True,
                        help='场景名（scene_000）、纯数字（0）或 all')

    # 可选
    parser.add_argument('--output_dir',  default=None,
                        help='输出根目录（默认 <save_dir>/pointclouds）')
    parser.add_argument('--device',      default=None,
                        help='CUDA 设备，如 "0" 或 "cuda:0"')

    # 相机
    parser.add_argument('--cam_ids',    type=str, default='0',
                        help='相机索引，逗号分隔（默认 "0"；"all" = 全部）')

    # 帧
    parser.add_argument('--frame_skip', type=int, default=1,
                        help='每隔 N 帧取一帧（默认 1 = 全部）')
    parser.add_argument('--frames',     type=str, default=None,
                        help='指定帧索引，逗号分隔，优先级高于 --frame_skip')

    # 过滤
    parser.add_argument('--max_dist',    type=float, default=80.0,
                        help='过滤距相机超过此距离（米）的点（默认 80；0 = 不过滤）')

    # 合并
    parser.add_argument('--save_merged', action='store_true', default=True,
                        help='额外保存场景合并点云 merged_pointcloud.ply（默认开启）')

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
            os.path.dirname(config.get('save_dir', './work_dirs/default')), '3d_output', 'pointcloud'
        )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output: {args.output_dir}")

    # ----- 相机列表 -----
    if args.cam_ids.strip().lower() == 'all':
        num_cams = config['model_cfg'].get('num_cams', 1)
        cam_ids  = list(range(num_cams))
        print(f"Cameras: all ({num_cams})")
    else:
        cam_ids = [int(x.strip()) for x in args.cam_ids.split(',')]
        print(f"Cameras: {cam_ids}")

    # ----- 帧列表 -----
    frame_indices = None
    if args.frames:
        frame_indices = [int(x.strip()) for x in args.frames.split(',')]
        print(f"Frames (explicit): {frame_indices}")
    elif args.frame_skip > 1:
        print(f"Frame skip: every {args.frame_skip} frames")

    # ----- 场景路径解析 -----
    data_path = config['data_cfg']['data_path']

    def _resolve_scene_dir(scene_name: str) -> str:
        if scene_name.isdigit():
            scene_name = f"scene_{int(scene_name):03d}"
        if os.path.isdir(scene_name):
            return os.path.abspath(scene_name)
        candidate = os.path.join(data_path, scene_name)
        if os.path.isdir(candidate):
            return candidate
        if os.path.basename(data_path) == scene_name and os.path.isdir(data_path):
            return data_path
        raise FileNotFoundError(
            f"Scene '{scene_name}' not found.\n"
            f"  Tried: {candidate}\n"
            f"  data_path: {data_path}"
        )

    if args.scene.strip().lower() == 'all':
        if os.path.isdir(data_path) and any(
            d.startswith('scene_') for d in os.listdir(data_path)
        ):
            scene_dirs = sorted([
                os.path.join(data_path, d)
                for d in os.listdir(data_path)
                if d.startswith('scene_') and os.path.isdir(os.path.join(data_path, d))
            ])
        else:
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

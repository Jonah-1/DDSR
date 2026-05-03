#!/usr/bin/env python3
#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

"""
Scene-based inference script for ReconDrive
This script demonstrates inference using scene-by-scene iteration
"""

import yaml
import argparse
import os
import sys
import torch
import json
from pathlib import Path
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn.functional as F
from gsplat.rendering import rasterization
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "models"))  # Add models directory for vggt imports

from dataset.vggt4dgs_data_module import VGGT4DGS_LITDataModule
from dataset.vggt3dgs_scene_data_module import VGGT3DGS_SceneDataModule
from dataset.cvg_scene_data_module import CVGSceneSequence
from dataset.vggt4dgs_scene_dataset import custom_collate_fn
from models.recondrive_model import ReconDrive_LITModelModule


class SceneSampleDataset(Dataset):
    """Dataset wrapper that supports both pre-loaded samples and lazy loading"""
    
    def __init__(self, samples_or_indices, dataset=None, scene_idx=None):
        if dataset is not None and scene_idx is not None:
            # Lazy loading mode: samples_or_indices is list of sample indices
            self.lazy_mode = True
            self.sample_indices = samples_or_indices
            self.dataset = dataset
            self.scene_idx = scene_idx
        else:
            # Pre-loaded mode: samples_or_indices is list of actual samples
            self.lazy_mode = False
            self.samples = samples_or_indices
    
    def __len__(self):
        if self.lazy_mode:
            return len(self.sample_indices) 
        else:
            return len(self.samples)
    
    def __getitem__(self, idx):
        if self.lazy_mode:
            # Load sample on-demand
            sample_idx = self.sample_indices[idx] 
            return self.dataset.__getitem__(sample_idx,self.scene_idx)
        else:
            return self.samples[idx]



def load_model_from_checkpoint(checkpoint_path, model_cfg, device):
    """Load model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Ensure batch_size is in model config
    if 'batch_size' not in model_cfg:
        model_cfg['batch_size'] = 1  # Set default batch_size for inference
    
    # Initialize model
    model = ReconDrive_LITModelModule(
        cfg=model_cfg,
        save_dir='./temp_log',
        logger=None
    )

    model.load_pretrained_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    return model


def run_inference(model_cfg=None, model=None, checkpoint_path=None,
                  scene_dataloader=None, device='cuda:0',
                  save_results=True, output_dir=None, novel_distances=[1.0, 2.0],
                  eval_resolution='280x518', datamodule_type='vggt4dgs'):
    """
    Scene-based inference function for single GPU

    Args:
        model_cfg: Model configuration (when model=None)
        model: Pre-loaded model (optional)
        checkpoint_path: Path to model checkpoint
        scene_dataloader: Scene data loader
        device: Device string (e.g., 'cuda:0')
        save_results: Whether to save results
        output_dir: Output directory
        novel_distances: List of distances for novel view generation
        eval_resolution: Resolution mode - 'original' or 'upsampled'
    """
    print(f"\nStarting single-GPU scene-based inference on {device}")
    print(f"Number of scenes to process: {len(scene_dataloader)}")

    # Load model if not provided
    if model is None:
        if model_cfg is None or checkpoint_path is None:
            raise ValueError("model_cfg and checkpoint_path required when model is None")
        model = load_model_from_checkpoint(checkpoint_path, model_cfg, device)

    return _run_single_gpu_inference(model, scene_dataloader, device, save_results, output_dir, novel_distances, eval_resolution, datamodule_type)


def save_rendered_image(tensor_img, save_path, upsample_to=None):
    """Save a tensor image to file with optional upsampling"""
    if tensor_img.dim() == 4:
        tensor_img = tensor_img.squeeze(0)

    if upsample_to is not None:
        target_height, target_width = upsample_to
        device = tensor_img.device
        if tensor_img.device.type == 'cpu':
            tensor_img = tensor_img.cuda()
        
        tensor_img = tensor_img.unsqueeze(0)
        tensor_img = F.interpolate(tensor_img, size=(target_height, target_width), 
                                 mode='bilinear', align_corners=False)
        tensor_img = tensor_img.squeeze(0)
        
        if device.type == 'cpu':
            tensor_img = tensor_img.cpu()
    
    img_np = tensor_img.detach().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    Image.fromarray(img_np).save(save_path)


def create_lateral_translation_matrices(translation_distances=[1.0, 2.0]):
    """Create transformation matrices for lateral (left/right) ego vehicle translation"""
    transforms = {}
    
    for dist in translation_distances:
        left_transform = torch.eye(4)
        left_transform[1, 3] = dist  # Negative Y for left
        transforms[f'left_{dist}m'] = left_transform
        
        right_transform = torch.eye(4)
        right_transform[1, 3] = -dist   # Positive Y for right
        transforms[f'right_{dist}m'] = right_transform
    
    return transforms


def render_novel_views(model, recontrast_data, render_data, device, scene_name, sample_idx, save_dir, actual_sample_idx, translation_distances=[1.0, 2.0], eval_resolution='280x518', novel_render_frames=[]):
    """Render novel views with lateral ego translation"""
    # Get transformation matrices for translation
    translation_transforms = create_lateral_translation_matrices(translation_distances)
    
    saved_paths = []
    
    xyz_i = recontrast_data['xyz'][sample_idx:sample_idx+1]  # Keep batch dimension
    rot_i = recontrast_data['rot_maps'][sample_idx:sample_idx+1]
    scale_i = recontrast_data['scale_maps'][sample_idx:sample_idx+1]
    opacity_i = recontrast_data['opacity_maps'][sample_idx:sample_idx+1]
    sh_i = recontrast_data['sh_maps'][sample_idx:sample_idx+1]
    
    # Get camera parameters
    frame_id = 0  # Use current frame

    num_cams = getattr(model, 'num_cams', 6)
    model_width = getattr(model, 'width', 518)
    model_height = getattr(model, 'height', 280)
    for transform_name, transform_matrix in translation_transforms.items():
        transform_matrix = transform_matrix.to(device)
        for frame_id in novel_render_frames:
            for cam_id in range(num_cams):
                # Get original camera extrinsics and intrinsics
                original_e2c_extr = render_data[('e2c_extr', frame_id, cam_id)][sample_idx:sample_idx+1]
                K_i = render_data[('K', frame_id, cam_id)][sample_idx:sample_idx+1, :3, :3]
                
                # Apply lateral translation to camera pose
                # Transform ego to camera: new_e2c = e2c @ inv(transform)
                novel_e2c_extr = torch.matmul(original_e2c_extr, torch.linalg.inv(transform_matrix.unsqueeze(0)))
                
                # Render with new camera pose
                render_colors_i, render_alphas_i, meta_i = rasterization(
                    xyz_i.squeeze(0),      # [N, 3]
                    rot_i.squeeze(0),      # [N, 4]
                    scale_i.squeeze(0),    # [N, 3]
                    opacity_i.squeeze(0).squeeze(-1),  # [N]
                    sh_i.squeeze(0),       # [N, K, 3]
                    novel_e2c_extr,        # [1, 4, 4]
                    K_i,                   # [1, 3, 3]
                    model_width,
                    model_height,
                    sh_degree=getattr(model, 'sh_degree', 3),
                    render_mode="RGB",
                )
                
                # Extract RGB and convert to proper format
                render_rgb = render_colors_i[..., :3].permute(0, 3, 1, 2)[0]  # [C, H, W]
                
                # Save the novel view
                global_sample_idx =  actual_sample_idx + frame_id
                save_path = os.path.join(save_dir, scene_name, 
                                        f'sample_{global_sample_idx:04d}', transform_name, f'{eval_resolution}_cam_{cam_id}.png')
            
                resize_height, resize_width = eval_resolution.split('x')
                resize_height = int(resize_height)
                resize_width = int(resize_width)
                if resize_height !=model_height or resize_width != model_width:
                    save_rendered_image(render_rgb, save_path, upsample_to=(resize_height, resize_width))
                else:  # eval_resolution == 'original'
                    save_rendered_image(render_rgb, save_path)
        
    return saved_paths

def _do_rasterize(gs_xyz, gs_rot, gs_scale, gs_opacity, gs_sh, e2c_extr, K_i,
                  model_width, model_height, sh_degree):
    render_colors, _, _ = rasterization(
        gs_xyz.squeeze(0),
        gs_rot.squeeze(0),
        gs_scale.squeeze(0),
        gs_opacity.squeeze(0).squeeze(-1),
        gs_sh.squeeze(0),
        e2c_extr,
        K_i,
        model_width,
        model_height,
        sh_degree=sh_degree,
        render_mode="RGB",
    )
    return render_colors[..., :3].permute(0, 3, 1, 2)[0]


def render_split_frames(model, recontrast_data, render_data, device, scene_name, sample_idx,
                        save_dir, actual_sample_idx, eval_resolution='280x518', render_frames=[0]):
    """Render frame 0 / frame 1 Gaussians separately, plus cross-frame renders."""
    num_cams = getattr(model, 'num_cams', 6)
    model_width = getattr(model, 'width', 518)
    model_height = getattr(model, 'height', 280)
    sh_degree = getattr(model, 'sh_degree', 3)

    xyz_key = 'xyz_transformed' if 'xyz_transformed' in recontrast_data else 'xyz'
    rot_key = 'rot_maps_transformed' if 'rot_maps_transformed' in recontrast_data else 'rot_maps'

    xyz_i     = recontrast_data[xyz_key][sample_idx:sample_idx+1]
    rot_i     = recontrast_data[rot_key][sample_idx:sample_idx+1]
    scale_i   = recontrast_data['scale_maps'][sample_idx:sample_idx+1]
    opacity_i = recontrast_data['opacity_maps'][sample_idx:sample_idx+1]
    sh_i      = recontrast_data['sh_maps'][sample_idx:sample_idx+1]

    mid = xyz_i.shape[1] // 2

    gs = {
        'frame0': (xyz_i[:, :mid], rot_i[:, :mid], scale_i[:, :mid], opacity_i[:, :mid], sh_i[:, :mid]),
        'frame1': (xyz_i[:, mid:], rot_i[:, mid:], scale_i[:, mid:], opacity_i[:, mid:], sh_i[:, mid:]),
    }

    resize_h, resize_w = (int(v) for v in eval_resolution.split('x')) if 'x' in eval_resolution else (model_height, model_width)
    upsample = (resize_h, resize_w) if (resize_h != model_height or resize_w != model_width) else None

    save_base = os.path.join(save_dir, scene_name, f'sample_{actual_sample_idx:04d}', 'split_frames')

    input_all = render_data.get('input_all', {})
    # Second context frame is frame context_span (e.g. frame 6 when context_span=6)
    context_span = getattr(model, 'context_span', 6)

    for cam_id in range(num_cams):
        # 第0帧相机（重建参考帧）
        e2c_f0 = render_data[('e2c_extr', 0, cam_id)][sample_idx:sample_idx+1]
        K_f0   = render_data[('K', 0, cam_id)][sample_idx:sample_idx+1, :3, :3]

        # 第context_span帧相机（另一个重建帧，如第6帧）: cam_T_cam_0toN @ inv(c2e_0)
        # K不随帧变，直接复用frame 0的K
        cam_T_cam_key = ('cam_T_cam', 0, context_span)
        if cam_T_cam_key in input_all:
            cam_T_cam = input_all[cam_T_cam_key][sample_idx:sample_idx+1, cam_id]  # [1, 4, 4]
            e2c_fN = torch.matmul(cam_T_cam, e2c_f0)
        else:
            e2c_fN = e2c_f0
        K_fN = K_f0

        renders = {
            # 第0帧高斯 × 第0帧相机
            'frame0_gs_on_frame0_cam': (gs['frame0'], e2c_f0, K_f0),
            # 第N帧高斯 × 第N帧相机
            'frameN_gs_on_frameN_cam': (gs['frame1'], e2c_fN, K_fN),
            # 第0帧高斯 × 第N帧相机
            'frame0_gs_on_frameN_cam': (gs['frame0'], e2c_fN, K_fN),
            # 第N帧高斯 × 第0帧相机
            'frameN_gs_on_frame0_cam': (gs['frame1'], e2c_f0, K_f0),
        }

        for tag, (g, e2c, K_i) in renders.items():
            rgb = _do_rasterize(*g, e2c, K_i, model_width, model_height, sh_degree)
            save_path = os.path.join(save_base, f'{eval_resolution}_{tag}_cam_{cam_id}.png')
            save_rendered_image(rgb, save_path, upsample_to=upsample)

        # merge: 全部高斯合并，分别在 frame0 / frameN 相机下渲染，同时保存真值图
        save_merge = os.path.join(save_dir, scene_name, f'sample_{actual_sample_idx:04d}', 'merge')
        gs_all = (
            torch.cat([gs['frame0'][0], gs['frame1'][0]], dim=1),
            torch.cat([gs['frame0'][1], gs['frame1'][1]], dim=1),
            torch.cat([gs['frame0'][2], gs['frame1'][2]], dim=1),
            torch.cat([gs['frame0'][3], gs['frame1'][3]], dim=1),
            torch.cat([gs['frame0'][4], gs['frame1'][4]], dim=1),
        )
        for tag, e2c, K_i in [('frame0', e2c_f0, K_f0), ('frameN', e2c_fN, K_fN)]:
            rgb = _do_rasterize(*gs_all, e2c, K_i, model_width, model_height, sh_degree)
            save_rendered_image(rgb,
                os.path.join(save_merge, f'{eval_resolution}_merge_{tag}_cam_{cam_id}.png'),
                upsample_to=upsample)

        for frame_id, tag in [(0, 'frame0'), (context_span, 'frameN')]:
            gt_key = ('groudtruth', frame_id, cam_id)
            if gt_key in render_data:
                gt = render_data[gt_key][sample_idx]  # [C, H, W]
                save_rendered_image(gt,
                    os.path.join(save_merge, f'{eval_resolution}_gt_{tag}_cam_{cam_id}.png'),
                    upsample_to=upsample)


def to_device(data, device):
    if isinstance(data, dict):
        return {k: (v if k == 'vehicle_annotations' else to_device(v, device)) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device) for x in data)
    elif torch.is_tensor(data):
        return data.to(device)
    else:
        return data

def _extract_first_value(value, default=None):
    """Extract first value from list/tensor or return as-is"""
    if isinstance(value, list):
        return value[0] if value else default
    elif isinstance(value, torch.Tensor):
        if value.numel() > 0:
            return value[0].item() if value.dim() > 0 else value.item()
        return default
    return value

def _extract_scene_idx(scene_batch, default_idx=0):
    """Extract scene_idx from batch, checking multiple locations"""
    for key in ['target_frames', 'context_frames']:
        if key in scene_batch and 'scene_idx' in scene_batch[key]:
            return _extract_first_value(scene_batch[key]['scene_idx'], default_idx)
    return default_idx

def save_gaussians_ply(gaussians: dict, save_path: str):
    """标准 3DGS PLY：scale=log(s), opacity=logit(o), f_dc/f_rest channel-major。"""
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        print("  [ERROR] plyfile not installed: pip install plyfile")
        return

    xyz     = gaussians['xyz'].float()
    rot     = gaussians['rot'].float().numpy()
    scale   = gaussians['scale'].float().numpy()
    opacity = gaussians['opacity'].float().numpy()
    sh      = gaussians['sh'].float()   # [N, K, 3]
    N, K, _ = sh.shape

    f_dc   = sh[:, 0, :].numpy()                                      # [N, 3]
    f_rest = sh[:, 1:, :].permute(0, 2, 1).reshape(N, -1).numpy()    # [N, 3*(K-1)]

    xyz_np  = xyz.numpy()
    normals = np.zeros_like(xyz_np)

    scale_raw   = np.log(np.clip(scale, 1e-9, None))
    opacity_raw = np.log(np.clip(opacity, 1e-9, 1 - 1e-9) /
                         (1 - np.clip(opacity, 1e-9, 1 - 1e-9)))

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
    print(f"  → PLY: {N:,} Gaussians  {save_path}")


def save_gaussians(recontrast_data, sample_idx, save_dir, scene_name, actual_sample_idx,
                   model=None, render_data=None, eps2d=0.3):
    """Save frame 0 and frame N Gaussians as PLY.
    - uses xyz_transformed / rot_maps_transformed when available
    - eps2d scale boost (prevents degenerate thin splats)
    """
    num_cams = getattr(model, 'num_cams', 1) if model is not None else 1

    N_total        = recontrast_data['xyz'].shape[1]
    mid_point      = N_total // 2
    points_per_cam = mid_point // num_cams

    xyz_key = 'xyz_transformed' if 'xyz_transformed' in recontrast_data else 'xyz'
    rot_key = 'rot_maps_transformed' if 'rot_maps_transformed' in recontrast_data else 'rot_maps'

    save_base = os.path.join(save_dir, scene_name, f'sample_{actual_sample_idx:04d}', 'gaussians')

    for tag, half_start in [('frame0', 0), ('frameN', mid_point)]:
        all_xyz, all_rot, all_scale, all_opacity, all_sh = [], [], [], [], []

        for cam_id in range(num_cams):
            s = half_start + cam_id * points_per_cam
            e = s + points_per_cam

            xyz_cam  = recontrast_data[xyz_key][sample_idx, s:e].detach().cpu()
            rot_cam  = recontrast_data[rot_key][sample_idx, s:e].detach().cpu()
            sc_cam   = recontrast_data['scale_maps'][sample_idx, s:e].detach().cpu()
            op_cam   = recontrast_data['opacity_maps'][sample_idx, s:e].detach().cpu()
            sh_cam   = recontrast_data['sh_maps'][sample_idx, s:e].detach().cpu()

            if render_data is not None:
                try:
                    e2c = render_data[('e2c_extr', 0, cam_id)][sample_idx:sample_idx+1][0].cpu().float()
                    K   = render_data[('K',        0, cam_id)][sample_idx:sample_idx+1, :3, :3][0].cpu().float()
                    fx  = K[0, 0].item()
                    xyz_h  = torch.cat([xyz_cam.float(), torch.ones(xyz_cam.shape[0], 1)], dim=-1)
                    depth  = (e2c[:3] @ xyz_h.T).T[:, 2].clamp(min=0.01)
                    min_sc = (eps2d * depth / fx).unsqueeze(-1).expand_as(sc_cam)
                    sc_cam = torch.max(sc_cam, min_sc)
                except Exception as exc:
                    print(f"      [WARN] cam {cam_id} eps2d boost skipped: {exc}")

            all_xyz.append(xyz_cam)
            all_rot.append(rot_cam)
            all_scale.append(sc_cam)
            all_opacity.append(op_cam)
            all_sh.append(sh_cam)

        save_gaussians_ply({
            'xyz':     torch.cat(all_xyz,     dim=0),
            'rot':     torch.cat(all_rot,     dim=0),
            'scale':   torch.cat(all_scale,   dim=0),
            'opacity': torch.cat(all_opacity, dim=0),
            'sh':      torch.cat(all_sh,      dim=0),
        }, os.path.join(save_base, f'{tag}_gaussians.ply'))


def _process_scene_batch(model, scene_batch, device, gpu_id=0, save_renders=True, output_dir=None, novel_distances=[1.0, 2.0], eval_resolution='280x518', batch_idx=0, datamodule_type='vggt4dgs'):
    """Process a single scene batch: run predict_step and save split-frame renders."""
    scene_start_time = time.time()

    scene_name = scene_batch['scene_name']
    scene_token = scene_batch['scene_token']

    original_sample_indices = None
    frame_skip = 6

    if 'samples' in scene_batch:
        scene_samples = scene_batch['samples']
        scene_length = len(scene_samples)
        if frame_skip is not None and frame_skip > 1:
            original_sample_indices = list(range(0, scene_length, frame_skip))
            filtered_samples = [scene_samples[i] for i in original_sample_indices]
            scene_dataset = SceneSampleDataset(filtered_samples)
        else:
            scene_dataset = SceneSampleDataset(scene_samples)
            original_sample_indices = list(range(scene_length))
    else:
        scene_length = scene_batch['scene_length']
        all_indices = scene_batch['sample_indices']
        if frame_skip is not None and frame_skip > 1:
            positions_to_keep = list(range(0, len(all_indices), frame_skip))
            original_sample_indices = [all_indices[pos] for pos in positions_to_keep]
        else:
            original_sample_indices = all_indices
        scene_dataset = SceneSampleDataset(
            original_sample_indices,
            dataset=scene_batch['dataset'],
            scene_idx=scene_batch['scene_idx']
        )

    print(f"GPU {gpu_id}: Processing Scene: {scene_name} ({len(original_sample_indices)} samples)")

    from dataset.cvg_data_module import cvg_collate_fn
    if datamodule_type == 'cvg':
        def cvg_inference_collate_fn(batch):
            if not batch:
                return {}
            return cvg_collate_fn(batch)
        collate_fn_to_use = cvg_inference_collate_fn
    else:
        collate_fn_to_use = custom_collate_fn

    scene_loader = DataLoader(
        scene_dataset, batch_size=1, shuffle=False,
        pin_memory=False, num_workers=0, drop_last=False,
        collate_fn=collate_fn_to_use
    )

    batch_count = 0
    max_samples = 2
    for batch_data in scene_loader:
        if batch_count >= max_samples:
            break
        batch_count += 1
        actual_sample_idx = original_sample_indices[batch_count - 1] if batch_count - 1 < len(original_sample_indices) else batch_count - 1

        batch_data = to_device(batch_data, device)
        output = model.predict_step(batch_data, batch_idx)

        if isinstance(output, tuple) and output_dir:
            batch_recontrast_data, batch_render_data, _ = output
            render_split_frames(
                model, batch_recontrast_data, batch_render_data,
                device, scene_name, 0, output_dir, actual_sample_idx, eval_resolution
            )
            save_gaussians(batch_recontrast_data, 0, output_dir, scene_name, actual_sample_idx,
                           model=model, render_data=batch_render_data)
            print(f"GPU {gpu_id}: Saved split renders + gaussians for sample {actual_sample_idx}")

    scene_processing_time = time.time() - scene_start_time
    return {
        'scene_idx': scene_batch.get('scene_idx', 0),
        'scene_name': scene_name,
        'scene_token': scene_token,
        'processed_samples': batch_count,
        'processing_time': scene_processing_time,
    }


def _run_single_gpu_inference(model, scene_dataloader, device, save_results=True, output_dir=None, novel_distances=[1.0, 2.0], eval_resolution='280x518', datamodule_type='vggt4dgs'):
    """Run split-frame rendering on all scenes."""
    print(f"\nStarting inference on device: {device}")
    print(f"Number of scenes: {len(scene_dataloader)}")
    all_scene_results = []

    with torch.no_grad():
        for scene_idx, scene_batch in enumerate(scene_dataloader):
            scene_batch['scene_idx'] = scene_idx
            result = _process_scene_batch(model, scene_batch, device, gpu_id=0,
                                          save_renders=save_results, output_dir=output_dir,
                                          novel_distances=novel_distances, eval_resolution=eval_resolution,
                                          batch_idx=scene_idx, datamodule_type=datamodule_type)
            all_scene_results.append(result)

    print(f"\nInference completed. Scenes: {len(all_scene_results)}, "
          f"Samples: {sum(r['processed_samples'] for r in all_scene_results)}")
    return all_scene_results



def save_inference_results(results, output_dir):
    """Save inference results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    if 'overall_metrics' in results:
        # New format with overall metrics
        scene_results = results['scene_results']
        overall_metrics = results['overall_metrics']
        
        # Create summary
        summary = {
            'overall_metrics': overall_metrics,
            'total_scenes': len(scene_results),
            'total_samples': sum(r['processed_samples'] for r in scene_results),
            'total_time': sum(r['processing_time'] for r in scene_results),
            'scenes': [
                {
                    'scene_idx': r['scene_idx'],
                    'scene_name': r['scene_name'],
                    'scene_token': r['scene_token'],
                    'processed_samples': r['processed_samples'],
                    'processing_time': r['processing_time'],
                    'metrics': r['metrics']
                }
                for r in scene_results
            ]
        }
        
        # Save summary
        summary_file = os.path.join(output_dir, 'inference_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        detailed_file = os.path.join(output_dir, 'inference_detailed.json')
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save per-scene evaluation results
        for scene_result in scene_results:
            scene_name = scene_result['scene_name']
            scene_eval_file = os.path.join(output_dir, f'scene_{scene_name}_evaluation.json')
            scene_eval_data = {
                'scene_name': scene_name,
                'scene_token': scene_result['scene_token'],
                'metrics': scene_result['metrics'],
                'sample_metrics': scene_result['sample_metrics'],
                'processing_info': {
                    'processed_samples': scene_result['processed_samples'],
                    'processing_time': scene_result['processing_time'],
                    'avg_sample_time': scene_result['avg_sample_time']
                }
            }
            
            with open(scene_eval_file, 'w') as f:
                json.dump(scene_eval_data, f, indent=2)
        
        print(f"\nResults saved:")
        print(f"  Summary: {summary_file}")
        print(f"  Detailed: {detailed_file}")
        print(f"  Per-scene evaluations: {output_dir}/scene_*_evaluation.json")
        
    else:
        # Legacy format
        summary = {
            'total_scenes': len(results),
            'total_samples': sum(r['processed_samples'] for r in results),
            'total_time': sum(r['processing_time'] for r in results),
            'scenes': [
                {
                    'scene_idx': r['scene_idx'],
                    'scene_name': r['scene_name'],
                    'scene_token': r['scene_token'],
                    'processed_samples': r['processed_samples'],
                    'processing_time': r['processing_time'],
                    'scene_stats': r.get('scene_stats', {})
                }
                for r in results
            ]
        }
        
        # Save summary
        summary_file = os.path.join(output_dir, 'inference_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        detailed_file = os.path.join(output_dir, 'inference_detailed.json')
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved:")
        print(f"  Summary: {summary_file}")
        print(f"  Detailed: {detailed_file}")


def main():
    parser = argparse.ArgumentParser(description='Scene-based inference for VGGT3DGS')
    parser.add_argument('--cfg_path', type=str, required=True, help='Configuration file path')
    parser.add_argument('--restore_ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--max_scenes', type=int, default=None, help='Maximum number of scenes to process (default: all scenes)')
    parser.add_argument('--device', type=str, default=None, help='Device to use (e.g., cuda:0)')

    parser.add_argument('--no_renders', action='store_true', help='Disable saving rendered images and novel views')
    parser.add_argument('--novel_distances', type=str, default='0.5,1.0,2.0,3.0', 
                       help='Novel view translation distances in meters (comma-separated, e.g., "0.5,1.0,2.0,3.0")')
    parser.add_argument('--eval_resolution', type=str, default='original',# choices=['original', 'upsampled'],
                       help='Evaluation resolution mode: "original" for 280x518, "upsampled" for 900x1600')

    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.cfg_path}")
    with open(args.cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Set batch_size to 1 for inference (CRITICAL: must be 1 for proper scene processing)
    config['model_cfg']['batch_size'] = 1
    config['data_cfg']['batch_size'] = 1
    
    # Pass temporal config from data_cfg to model_cfg for time_delta calculation
    if 'context_span' in config['data_cfg']:
        config['model_cfg']['context_span'] = config['data_cfg']['context_span']
    if 'nuscenes_version' in config['data_cfg']:
        config['model_cfg']['nuscenes_version'] = config['data_cfg']['nuscenes_version']


    # Parse device
    if args.device:
        # Single GPU specified: "cuda:0" or "0"
        if args.device.startswith('cuda:'):
            device = args.device
        else:
            device = f"cuda:{args.device}"
    elif config.get('devices'):
        device = f"cuda:{config['devices'][0]}"
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {device}")
    print(f"Batch size: {config['data_cfg']['batch_size']}")
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(config['save_dir'], 'scene_inference_results')
    
    print(f"Output directory: {args.output_dir}")
    
    # Parse save renders flag
    save_renders = not args.no_renders
    
    # Parse novel view distances
    try:
        novel_distances = [float(d.strip()) for d in args.novel_distances.split(',')]
    except ValueError:
        raise ValueError(f"Invalid novel_distances format: {args.novel_distances}. Use comma-separated floats like '0.5,1.0,2.0,3.0'")
    
    print(f"Save renders: {save_renders}")
    print(f"Novel view distances: {novel_distances}")
    print(f"Evaluation resolution: {args.eval_resolution}")

    # CRITICAL: Ensure batch_size is 1 before creating data module
    print(f"Original batch_size in config: {config['data_cfg'].get('batch_size', 'not set')}")
    config['data_cfg']['batch_size'] = 1  # Must be 1 for proper scene processing
    print(f"Override batch_size to: {config['data_cfg']['batch_size']}")

    # Initialize scene-based data module
    print("Initializing scene-based data module...")
    datamodule_type = config.get('datamodule_type', 'vggt4dgs')
    if datamodule_type == 'cvg':
        print(f"Loading CVG data module...")
        data_module = CVGSceneSequence(
            cfg=config['data_cfg'],
        )
    else:
        print(f"Loading VGGT3DGS data module...")
        data_module = VGGT3DGS_SceneDataModule(
            cfg=config['data_cfg'],
        )

    # data_module = VGGT3DGS_SceneDataModule(cfg=config['data_cfg'])
    data_module.setup(stage='test')
    
    # Get scene dataloader
    scene_dataloader = data_module.test_scene_dataloader()
    total_scenes = len(scene_dataloader)
    
    if args.max_scenes:
        print(f"Limiting to {args.max_scenes} scenes (out of {total_scenes})")
        # Limit scenes if requested
        scene_list = []
        for i, scene_batch in enumerate(scene_dataloader):
            if i >= args.max_scenes:
                break
            scene_list.append(scene_batch)
        scene_dataloader = scene_list
    else:
        print(f"Processing all {total_scenes} scenes")


    # Run single-GPU inference
    results = run_inference(
        model_cfg=config['model_cfg'],
        checkpoint_path=args.restore_ckpt,
        scene_dataloader=scene_dataloader,
        device=device,
        save_results=save_renders,
        output_dir=args.output_dir,
        novel_distances=novel_distances,
        eval_resolution=args.eval_resolution,
        datamodule_type=datamodule_type
    )
    
    print(f"\nScene-based inference completed successfully!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
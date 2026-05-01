"""
Sky Model for ReconDrive
Adapted from DGGT's SkyGaussian (dggt/dggt/models/sky.py)

用于渲染天空背景的高斯点云模型。
通过投影当前相机视图到高斯点云上，动态生成天空背景。
"""

import numpy as np
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Set
from gsplat.rendering import rasterization


def fibonacci_sphere(samples=1):
    """在单位球面上均匀采样（使用 Fibonacci 球面分布）"""
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        z = 1 - (i / float(samples - 1))  # z goes from 1 to 0 (upper hemisphere, pole at +Z = up)
        radius = math.sqrt(1 - z ** 2)
        theta = phi * i

        x = math.cos(theta) * radius
        y = math.sin(theta) * radius
        points.append((x, y, z))

    return points


def k_nearest_neighbors(x: torch.Tensor, k: int):
    """计算 k 近邻距离（用于初始化点的尺度）"""
    try:
        from sklearn.neighbors import NearestNeighbors
        x_np = x.cpu().numpy()
        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)
        distances, indices = nn_model.kneighbors(x_np)
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)
    except ImportError:
        # 没有 sklearn 时的备用方案
        dist = torch.cdist(x, x)
        distances, indices = torch.topk(dist, k + 1, dim=1, largest=False)
        return distances[:, 1:].cpu().numpy().astype(np.float32), \
               indices[:, 1:].cpu().numpy().astype(np.float32)


class MLPHead(nn.Module):
    """MLP网络，从投影特征预测颜色和scale残差（与DGGT完全兼容）"""

    def __init__(self, in_dim: int = 3, hidden_dim: int = 64, out_dim: int = 6):
        super().__init__()
        # 与DGGT保持兼容的层索引
        self.layers = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim),   # layer 0
            nn.Linear(hidden_dim, out_dim),  # layer 1
        ])
        self.activation = nn.ReLU()
        self.out_activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.layers[0](x))
        x = self.out_activation(self.layers[1](x))
        return x


class Projector(nn.Module):
    """将 3D 点投影到图像平面并采样特征"""

    def forward(self, xyz: torch.Tensor, images: torch.Tensor,
                extrinsics: torch.Tensor, intrinsics: torch.Tensor):
        """
        投影天空点到源图像，采样其对应位置的颜色特征

        Args:
            xyz: (N, 3) - 世界坐标系中的点
            images: (B, C, H, W) - 输入图像
            extrinsics: (B, 4, 4) - 相机外参（world to camera）
            intrinsics: (B, 3, 3) - 相机内参

        Returns:
            sampled_features: (M, C) - 采样的图像特征（M <= N）
            proj_mask: (N,) - 投影有效掩码
            all_features: (N, C) - 所有点的特征（无效点为 0）
        """
        B, C, H, W = images.shape
        N = xyz.shape[0]
        device = xyz.device

        # 齐次坐标
        xyz_homo = torch.cat([xyz, torch.ones(N, 1, device=device)], dim=1)  # (N, 4)

        xyz_proj_list = []
        valid_mask_list = []

        for i in range(B):
            # 转换到相机坐标系
            xyz_cam = (extrinsics[i] @ xyz_homo.T).T  # (N, 4)

            # 只考虑摄像头前方的点（z > 0）
            valid_z = xyz_cam[:, 2] > 0

            # 投影到图像平面
            xyz_cam_3d = xyz_cam[:, :3]
            xyz_proj = (intrinsics[i] @ xyz_cam_3d.T).T  # (N, 3)
            uv = xyz_proj[:, :2] / (xyz_proj[:, 2:3] + 1e-6)  # (N, 2)

            # 归一化到 [-1, 1]（grid_sample 所需格式）
            u_norm = (2 * uv[:, 0] / (W - 1)) - 1
            v_norm = (2 * uv[:, 1] / (H - 1)) - 1

            # 检查是否在图像范围内
            in_frame = (u_norm >= -1) & (u_norm <= 1) & (v_norm >= -1) & (v_norm <= 1)
            proj_mask = valid_z & in_frame

            xyz_proj_list.append(torch.stack([u_norm, v_norm], dim=1))
            valid_mask_list.append(proj_mask)

        # 合并所有相机的掩码（要求所有相机都能看到）
        stacked_masks = torch.stack(valid_mask_list, dim=0)  # (B, N)
        global_mask = stacked_masks.all(dim=0)  # (N,)

        # 采样所有相机的特征
        all_features_list = []
        for i in range(B):
            grid = xyz_proj_list[i].unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)

            sampled = torch.nn.functional.grid_sample(
                images[i:i+1],
                grid,
                align_corners=False,
                mode='bilinear',
                padding_mode='border'
            )  # (1, C, 1, N)

            # 正确的转换：(1, C, 1, N) → (N, C)
            features = sampled.squeeze(0).squeeze(1).T  # (C, 1, N) -> (C, N) -> (N, C)
            all_features_list.append(features)

        # 平均多相机特征
        stacked_features = torch.stack(all_features_list, dim=0)  # (B, N, C)
        avg_features = stacked_features.mean(dim=0)  # (N, C)

        # 只返回有效的特征
        valid_features = avg_features[global_mask]  # (M, C)

        return valid_features, global_mask, avg_features


class SkyGaussian(nn.Module):
    """
    天空背景高斯点云模型

    在半径为 radius 的球面上均匀采样 resolution² 个点，
    每个点都是一个 3D 高斯，用于渲染天空背景。

    Args:
        resolution: 采样分辨率（默认 300，生成 90000 个点）
        radius: 球半径（默认 50）
        center: 球心位置（默认 [0, 0, 0]）
        feature_dim: 输入特征维度（默认 3 = RGB）
        hidden_dim: MLP 隐层维度（默认 64）
    """

    def __init__(self,
                 resolution: int = 300,
                 radius: float = 50.0,
                 center: Optional[np.ndarray] = None,
                 feature_dim: int = 3,
                 hidden_dim: int = 64):
        super().__init__()

        self.resolution = resolution
        self.radius = radius
        self.center = center if center is not None else np.array([0, 0, 0])
        self.projector = Projector()

        # MLP 用于从投影特征预测颜色和scale残差（与DGGT一致：6维输出）
        self.bg_field = MLPHead(in_dim=feature_dim, hidden_dim=hidden_dim, out_dim=6)

        # ============ 初始化天空点云 ============
        num_bg_points = resolution ** 2

        # Fibonacci 球面采样
        xyz = fibonacci_sphere(num_bg_points)
        xyz = np.array(xyz) * radius
        sky_pnt = xyz.astype(np.float32) + self.center

        # 初始化尺度：基于 k 近邻距离的平均值
        bg_distances, _ = k_nearest_neighbors(torch.from_numpy(sky_pnt), k=3)
        bg_distances = torch.from_numpy(bg_distances)
        avg_dist = bg_distances.mean(dim=-1, keepdim=True)
        bg_scales = torch.log(avg_dist.repeat(1, 3))

        # 注册为缓冲区（不需要梯度）
        self.register_buffer("bg_pcd", torch.tensor(sky_pnt, dtype=torch.float32))
        self.register_buffer("bg_scales", bg_scales.float())
        self.register_buffer("bg_opacity", torch.ones(num_bg_points, 1, dtype=torch.float32))

        # 初始旋转（无旋转 = [1, 0, 0, 0]）
        quat = torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float32).repeat(num_bg_points, 1)
        self.register_buffer("bg_quat", quat)

        print(f"[SkyGaussian] Initialized with {num_bg_points} sky points")
        print(f"  Resolution: {resolution}×{resolution}")
        print(f"  Radius: {radius}")

    def reset_geometry(self):
        """重新从 fibonacci_sphere 初始化点位和尺度，用于加载旧 checkpoint 后修正坐标轴。"""
        xyz = fibonacci_sphere(self.resolution ** 2)
        xyz = np.array(xyz) * self.radius
        sky_pnt = xyz.astype(np.float32) + self.center
        bg_distances, _ = k_nearest_neighbors(torch.from_numpy(sky_pnt), k=3)
        bg_distances = torch.from_numpy(bg_distances)
        avg_dist = bg_distances.mean(dim=-1, keepdim=True)
        bg_scales = torch.log(avg_dist.repeat(1, 3))
        device = self.bg_pcd.device
        self.bg_pcd.copy_(torch.tensor(sky_pnt, dtype=torch.float32).to(device))
        self.bg_scales.copy_(bg_scales.float().to(device))
        print("[SkyGaussian] reset_geometry: bg_pcd reinitialized with Z-up pole")

    def _get_background_color(self, source_images: torch.Tensor,
                             source_extrinsics: torch.Tensor,
                             source_intrinsics: torch.Tensor,
                             downsample: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从源图像投影特征到天空点，并预测颜色和scale残差

        Args:
            source_images: (B, C, H, W) - 源图像
            source_extrinsics: (B, 4, 4) - 源相机外参
            source_intrinsics: (B, 3, 3) - 源相机内参
            downsample: 下采样因子

        Returns:
            background_rgb: (M, 3) - 预测的背景颜色
            proj_mask: (N,) - 投影有效掩码
            background_scale_res: (M, 3) - 尺度残差
        """
        # 调整内参以匹配下采样
        if downsample > 1:
            source_intrinsics_scaled = source_intrinsics.clone()
            source_intrinsics_scaled[:, 0, 0] /= downsample
            source_intrinsics_scaled[:, 1, 1] /= downsample
            source_intrinsics_scaled[:, 0, 2] /= downsample
            source_intrinsics_scaled[:, 1, 2] /= downsample
        else:
            source_intrinsics_scaled = source_intrinsics

        # 投影到源图像
        sampled_feat, proj_mask, _ = self.projector(
            self.bg_pcd.reshape(-1, 3),
            source_images,
            source_extrinsics,
            source_intrinsics_scaled
        )

        # 预测背景颜色和scale残差（6维输出：RGB + 3个scale）
        mlp_output = self.bg_field(sampled_feat).float()  # (M, 6)

        # 分离RGB和scale残差
        background_rgb = mlp_output[:, :3]  # (M, 3) - RGB in [0, 1] after Tanh+scaling
        background_scale_res = mlp_output[:, 3:6]  # (M, 3) - scale residuals

        return background_rgb, proj_mask, background_scale_res

    def forward(self, images: torch.Tensor, extrinsics: torch.Tensor,
                intrinsics: torch.Tensor, downsample: int = 1) -> torch.Tensor:
        """
        在给定相机位置渲染天空背景

        Args:
            images: (B, C, H, W) - 输入图像（用于投影提取颜色）
            extrinsics: (B, 4, 4) - 相机外参
            intrinsics: (B, 3, 3) - 相机内参
            downsample: 下采样因子

        Returns:
            bg_render: (B, 3, H, W) - 渲染的背景
        """
        B, C, H, W = images.shape

        # 从输入图像提取背景颜色
        background_rgb, proj_mask, background_scale_res = self._get_background_color(
            source_images=images,
            source_extrinsics=extrinsics,
            source_intrinsics=intrinsics,
            downsample=downsample
        )

        # 选择投影有效的点
        valid_bg_pcd = self.bg_pcd[proj_mask]
        valid_bg_scales = torch.exp(self.bg_scales[proj_mask]) + background_scale_res
        valid_bg_opacity = self.bg_opacity.squeeze(-1)[proj_mask]
        valid_bg_quat = self.bg_quat[proj_mask]

        # 分块渲染（减少内存使用）
        chunk_size = 4
        chunked_renders = []

        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)

            bg_render, _, _ = rasterization(
                means=valid_bg_pcd,
                quats=valid_bg_quat,
                scales=valid_bg_scales,
                opacities=valid_bg_opacity,
                colors=background_rgb,
                viewmats=extrinsics[start:end],
                Ks=intrinsics[start:end],
                width=W,
                height=H
            )  # (chunk, H, W, 3)

            # 转换为 (chunk, 3, H, W)
            bg_render = bg_render.permute(0, 3, 1, 2)
            chunked_renders.append(bg_render)

        bg_render = torch.cat(chunked_renders, dim=0)  # (B, 3, H, W)
        return bg_render

    def forward_with_new_pose(self, images: torch.Tensor,
                             source_extrinsics: torch.Tensor,
                             source_intrinsics: torch.Tensor,
                             target_extrinsics: torch.Tensor,
                             target_intrinsics: torch.Tensor,
                             downsample: int = 1) -> torch.Tensor:
        """
        先在源相机位置提取颜色，再在目标相机位置渲染

        用于生成新视角的天空背景：
        1. 从源图像提取天空颜色特征
        2. 在目标相机位置渲染这些特征

        Args:
            images: (B_src, C, H, W) - 源图像
            source_extrinsics: (B_src, 4, 4)
            source_intrinsics: (B_src, 3, 3)
            target_extrinsics: (B_tgt, 4, 4)
            target_intrinsics: (B_tgt, 3, 3)
            downsample: 下采样因子

        Returns:
            bg_render: (B_tgt, 3, H, W) - 目标位置的渲染背景
        """
        # 从源图像提取颜色
        background_rgb, proj_mask, background_scale_res = self._get_background_color(
            source_images=images,
            source_extrinsics=source_extrinsics,
            source_intrinsics=source_intrinsics,
            downsample=downsample
        )

        B_tgt = target_extrinsics.shape[0]
        H, W = images.shape[-2:]

        # 选择投影有效的点
        valid_bg_pcd = self.bg_pcd[proj_mask]
        valid_bg_scales = torch.exp(self.bg_scales[proj_mask]) + background_scale_res
        valid_bg_opacity = self.bg_opacity.squeeze(-1)[proj_mask]
        valid_bg_quat = self.bg_quat[proj_mask]

        # 分块渲染到目标位置
        chunk_size = 4
        chunked_renders = []

        for start in range(0, B_tgt, chunk_size):
            end = min(start + chunk_size, B_tgt)

            bg_render, _, _ = rasterization(
                means=valid_bg_pcd,
                quats=valid_bg_quat,
                scales=valid_bg_scales,
                opacities=valid_bg_opacity,
                colors=background_rgb,
                viewmats=target_extrinsics[start:end],  # (chunk, 4, 4)
                Ks=target_intrinsics[start:end],        # (chunk, 3, 3)
                width=W,
                height=H
            )  # (chunk, H, W, 3)
            # 转换为 (chunk, 3, H, W)
            bg_render = bg_render.permute(0, 3, 1, 2)
            chunked_renders.append(bg_render)

        bg_render = torch.cat(chunked_renders, dim=0)  # (B_tgt, 3, H, W)
        return bg_render


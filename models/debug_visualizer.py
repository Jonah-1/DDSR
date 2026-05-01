"""
Debug Visualizer
用于定期保存和可视化训练过程的中间结果（支持有/无sky_model）
"""

import os
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional
import numpy as np


class DebugVisualizer:
    """通用调试可视化工具，支持有/无sky model"""

    def __init__(self, output_dir: str = "debug_outputs", max_samples_per_step: int = 2):
        """
        Args:
            output_dir: 输出目录
            max_samples_per_step: 每个 step 最多保存多少个样本（节省空间）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples_per_step

    def save_image(self, tensor: torch.Tensor, path: Path) -> None:
        """
        保存单个图像张量为 PNG

        Args:
            tensor: 任意维度张量，需要降维到 [C, H, W] 或 [H, W]，值范围 [0, 1]
            path: 保存路径
        """
        # 转换为 numpy
        if tensor.is_cuda:
            tensor = tensor.cpu()

        # Detach from computation graph
        tensor = tensor.detach()

        # 降维到 3D：不断取第一个样本直到得到 3D 张量
        while tensor.ndim > 3:
            tensor = tensor[0]

        # 处理单通道
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)  # [1, H, W] -> [H, W]
            # 转换为灰度值 [0, 255]
            np_array = (tensor.numpy() * 255).astype(np.uint8)
        else:
            # RGB: [C, H, W] -> [H, W, C]
            np_array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # 保存
        from PIL import Image
        Image.fromarray(np_array).save(str(path))

    def log_step(
        self,
        global_step: int,
        batch_splating_data: Dict,
        frame_ids: list = None,
        cam_ids: list = None,
        sky_renders_dict: Dict = None,
        sky_masks: Dict = None,
        gt_images: Dict = None,
    ) -> None:
        """
        定期保存调试图像

        Args:
            global_step: 全局步数
            batch_splating_data: 包含 gaussian_color 和 gaussian_alpha 的字典
            frame_ids: 指定保存哪些 frame（None=保存全部）
            cam_ids: 指定保存哪些 camera（None=保存全部）
            sky_renders_dict: 包含 sky_render 的字典（可选，用于sky model）
            sky_masks: 包含 sky_mask 的字典（可选）
            gt_images: 包含 gt_image 的字典（可选）
        """
        step_dir = self.output_dir / f"step_{global_step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (step_dir / "render").mkdir(exist_ok=True)
        (step_dir / "gt").mkdir(exist_ok=True)

        # 仅在使用sky_model时创建sky相关目录
        if sky_renders_dict is not None and sky_renders_dict:
            (step_dir / "sky").mkdir(exist_ok=True)
            (step_dir / "blended").mkdir(exist_ok=True)
            (step_dir / "alpha").mkdir(exist_ok=True)
            (step_dir / "comparisons").mkdir(exist_ok=True)

        if sky_masks is not None and sky_masks:
            (step_dir / "mask").mkdir(exist_ok=True)

        # 提取唯一的 frame_ids 和 cam_ids
        all_frame_ids = set()
        all_cam_ids = set()

        for key in batch_splating_data.keys():
            if isinstance(key, tuple) and len(key) == 3:
                _, fid, cid = key
                all_frame_ids.add(fid)
                all_cam_ids.add(cid)

        # 筛选
        if frame_ids is None:
            frame_ids = sorted(list(all_frame_ids))
        if cam_ids is None:
            cam_ids = sorted(list(all_cam_ids))

        # 限制数量
        frame_ids = frame_ids[:self.max_samples]

        saved_count = 0
        for frame_id in frame_ids:
            for cam_id in cam_ids:
                # 获取高斯颜色（渲染结果）
                gaussian_key = ('gaussian_color', frame_id, cam_id)
                if gaussian_key not in batch_splating_data:
                    continue

                gaussian_color = batch_splating_data[gaussian_key]
                if gaussian_color.is_cuda:
                    gaussian_color = gaussian_color.cpu()
                gaussian_color = gaussian_color.detach()

                # 规范化形状：取第一个样本如果是多帧 [B, C, H, W] -> [C, H, W]
                while gaussian_color.ndim > 3:
                    gaussian_color = gaussian_color[0]

                # ============ 保存渲染结果（gaussian）============
                render_path = step_dir / "render" / f"frame_{frame_id}_cam_{cam_id}.png"
                self.save_image(gaussian_color, render_path)

                # ============ 保存 GT 图像（如果提供）============
                if gt_images is not None:
                    gt_key = ('gt_image', frame_id, cam_id)
                    if gt_key in gt_images:
                        gt_image = gt_images[gt_key]
                        if gt_image.is_cuda:
                            gt_image = gt_image.cpu()
                        gt_image = gt_image.detach()

                        # 规范化形状
                        while gt_image.ndim > 3:
                            gt_image = gt_image[0]

                        gt_path = step_dir / "gt" / f"frame_{frame_id}_cam_{cam_id}.png"
                        self.save_image(gt_image, gt_path)

                # ============ 下面仅在使用sky_model时执行 ============
                if sky_renders_dict is None or not sky_renders_dict:
                    saved_count += 1
                    if saved_count >= self.max_samples:
                        break
                    continue

                # 获取天空渲染
                sky_key = ('sky_render', frame_id, cam_id)
                sky_render = sky_renders_dict.get(sky_key)
                if sky_render is None:
                    saved_count += 1
                    if saved_count >= self.max_samples:
                        break
                    continue

                if sky_render.is_cuda:
                    sky_render = sky_render.cpu()
                sky_render = sky_render.detach()

                # 规范化形状
                while sky_render.ndim > 3:
                    sky_render = sky_render[0]

                # 获取 alpha
                alpha_key = ('gaussian_alpha', frame_id, cam_id)
                gaussian_alpha = batch_splating_data.get(alpha_key)
                if gaussian_alpha is not None:
                    if gaussian_alpha.is_cuda:
                        gaussian_alpha = gaussian_alpha.cpu()
                    gaussian_alpha = gaussian_alpha.detach()

                    # 规范化形状：移除所有 singleton 维度和批次维度，得到 [H, W]
                    while gaussian_alpha.ndim > 2 and gaussian_alpha.shape[-1] == 1:
                        gaussian_alpha = gaussian_alpha.squeeze(-1)
                    if gaussian_alpha.ndim == 3:
                        gaussian_alpha = gaussian_alpha[0]

                # 保存天空渲染结果
                sky_path = step_dir / "sky" / f"frame_{frame_id}_cam_{cam_id}.png"
                self.save_image(sky_render, sky_path)

                # 保存融合结果（使用 alpha blending）
                if gaussian_alpha is not None:
                    alpha_expanded = gaussian_alpha.unsqueeze(0)  # [1, H, W]
                    blended = gaussian_color * alpha_expanded + sky_render * (1 - alpha_expanded)
                else:
                    blended = (gaussian_color + sky_render) / 2.0

                blended_path = step_dir / "blended" / f"frame_{frame_id}_cam_{cam_id}.png"
                self.save_image(blended, blended_path)

                # 保存 alpha 通道（灰度）
                if gaussian_alpha is not None:
                    alpha_path = step_dir / "alpha" / f"frame_{frame_id}_cam_{cam_id}.png"
                    self.save_image(gaussian_alpha.unsqueeze(0), alpha_path)

                # 保存 mask（如果提供）
                if sky_masks is not None:
                    mask_key = ('sky_mask', frame_id, cam_id)
                    if mask_key in sky_masks:
                        gt_mask = sky_masks[mask_key]
                        if gt_mask.is_cuda:
                            gt_mask = gt_mask.cpu()
                        gt_mask = gt_mask.detach()

                        # 规范化形状
                        while gt_mask.ndim > 2:
                            gt_mask = gt_mask[0]

                        mask_path = step_dir / "mask" / f"frame_{frame_id}_cam_{cam_id}.png"
                        self.save_image(gt_mask.unsqueeze(0), mask_path)

                # 保存对比图
                try:
                    self._save_comparison(
                        step_dir / "comparisons" / f"frame_{frame_id}_cam_{cam_id}.png",
                        gaussian_color, sky_render, blended, gaussian_alpha
                    )
                except Exception as e:
                    print(f"[Warning] 对比图保存失败: {e}")

                saved_count += 1
                if saved_count >= self.max_samples:
                    break

        print(f"[DebugVisualizer] Step {global_step}: 保存了 {saved_count} 个样本到 {step_dir}")

    def _save_comparison(
        self,
        path: Path,
        gaussian: torch.Tensor,
        sky: torch.Tensor,
        blended: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
    ) -> None:
        """
        保存对比图（4 合 1）

        布局：
        [高斯  | 天空]
        [融合  | Alpha]
        """
        try:
            from PIL import Image
            import numpy as np

            def to_uint8_rgb(t):
                if t.is_cuda:
                    t = t.cpu()
                t = t.detach()

                # 确保是 3D: [C, H, W]
                while t.ndim > 3:
                    t = t[0]
                while t.ndim < 3:
                    t = t.unsqueeze(0)

                # [C, H, W] -> [H, W, C]
                return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # 转换为 uint8 RGB
            img_gaussian = to_uint8_rgb(gaussian)
            img_sky = to_uint8_rgb(sky)
            img_blended = to_uint8_rgb(blended)

            # 处理 alpha
            if alpha is not None:
                if alpha.is_cuda:
                    alpha = alpha.cpu()
                alpha = alpha.detach()

                # 降维到 2D [H, W]
                while alpha.ndim > 2:
                    alpha = alpha[0]

                # alpha: [H, W] -> [H, W, 3] (灰度)
                alpha_uint8 = (alpha.numpy() * 255).astype(np.uint8)
                img_alpha = np.stack([alpha_uint8, alpha_uint8, alpha_uint8], axis=-1)
            else:
                h, w = img_gaussian.shape[:2]
                img_alpha = np.full((h, w, 3), 128, dtype=np.uint8)

            # 创建 2x2 网格
            top = np.hstack([img_gaussian, img_sky])
            bottom = np.hstack([img_blended, img_alpha])
            comparison = np.vstack([top, bottom])

            Image.fromarray(comparison).save(str(path))
        except Exception as e:
            print(f"[Warning] 保存对比图失败: {e}")


def create_debug_visualizer(config: Dict = None) -> DebugVisualizer:
    """创建调试可视化工具"""
    output_dir = "debug_outputs"
    max_samples = 2

    if config:
        output_dir = config.get('debug_output_dir', output_dir)
        max_samples = config.get('debug_max_samples', max_samples)

    return DebugVisualizer(output_dir=output_dir, max_samples_per_step=max_samples)


# 保持向后兼容性
SkyDebugVisualizer = DebugVisualizer

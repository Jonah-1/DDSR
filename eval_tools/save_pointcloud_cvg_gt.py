#!/usr/bin/env python3
#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

"""
保存 CVG 数据集的 LiDAR 真值点云（Ground Truth）。

数据来源：直接读取 cvg_data_pipeline/data/<scene_name>/extracted/ 下的
原始 LiDAR 点云（cloud_body/*.pcd）和 ego-pose（Image_Pose_new_filtered.txt.json）。

坐标系：所有位姿在载入时归一化到第 0 帧自车坐标系（避免全局坐标大数值导致 float32 精度损失）。
可通过 --reference_frame 指定任意帧作为参考坐标系，输出文件名带 _ref<N> 后缀。
颜色按帧索引用 rainbow 色系区分，方便在 CloudCompare / MeshLab 中区分帧间运动。

输出目录结构：
  <output_dir>/<scene_name>/merge/
      <frame_idx>.ply               — 逐帧真值点云（world / 第 0 帧坐标系）
      <frame_idx>_ref<N>.ply        — 逐帧真值点云（第 N 帧坐标系，--reference_frame N）
      merged_pointcloud.ply         — 所有帧合并（--save_merged）
      merged_pointcloud_ref<N>.ply  — 合并点云（--reference_frame N + --save_merged）

用法示例：
  # 保存 scene_000 指定帧（第 0 帧坐标系）
  python eval_tools/save_pointcloud_cvg_gt.py \\
      --scene scene_000 \\
      --frames 0,6,12,18,24 \\
      --save_merged

  # 以第 10 帧为参考帧保存
  python eval_tools/save_pointcloud_cvg_gt.py \\
      --scene scene_000 \\
      --frames 0,5,10,15 \\
      --reference_frame 10 \\
      --save_merged

  # 保存所有场景，每6帧取一帧
  python eval_tools/save_pointcloud_cvg_gt.py \\
      --scene all \\
      --frame_skip 6 \\
      --save_merged
"""

import argparse
import colorsys
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp

# cvg_data_pipeline/data 相对脚本位置
DATA_ROOT = Path(__file__).resolve().parent.parent.parent / 'cvg_data_pipeline' / 'data'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ego 位姿插值
# ---------------------------------------------------------------------------

class EgoPoseInterpolator:
    def __init__(self, ego_poses: Dict[int, np.ndarray]):
        self.timestamps = sorted(ego_poses.keys())
        self.poses = [ego_poses[ts] for ts in self.timestamps]

        if len(self.timestamps) < 2:
            raise ValueError("需要至少2个位姿点来进行插值")

        self.positions = np.array([pose[:3, 3] for pose in self.poses])
        self.rotations = R.from_matrix(np.array([pose[:3, :3] for pose in self.poses]))
        self.interp_position = interp1d(
            self.timestamps, self.positions, axis=0, fill_value='extrapolate'
        )
        self.slerp = Slerp(np.array(self.timestamps), self.rotations)
        self.min_ts = self.timestamps[0]
        self.max_ts = self.timestamps[-1]

    def interpolate(self, target_ts: int) -> Optional[np.ndarray]:
        if target_ts < self.min_ts or target_ts > self.max_ts:
            logger.warning(f"时间戳 {target_ts} 超出范围 [{self.min_ts}, {self.max_ts}]")
            return None
        position = self.interp_position(target_ts)
        rotation_matrix = self.slerp(target_ts).as_matrix()
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = position
        return T


# ---------------------------------------------------------------------------
# CVG 场景 LiDAR 数据读取
# ---------------------------------------------------------------------------

class CVGLiDARReader:
    """读取 CVG 场景的 LiDAR 点云并变换到 world 坐标系。"""

    def __init__(self, scene_root: Path):
        self.scene_root = Path(scene_root)
        self.extracted_dir = self.scene_root / 'extracted'
        self.lidar_to_ego = np.eye(4, dtype=np.float32)
        self.ego_poses: Dict[int, np.ndarray] = {}
        self.lidar_timestamps: List[int] = []
        self.interpolator: Optional[EgoPoseInterpolator] = None
        self._load_ego_poses()
        self._scan_lidar_files()
        self._normalize_to_frame0()

    def _load_ego_poses(self):
        pose_file = self.extracted_dir / 'Image_Pose_new_filtered.txt.json'
        if not pose_file.exists():
            raise FileNotFoundError(f"位姿文件不存在: {pose_file}")
        with open(pose_file) as f:
            data = json.load(f)
        features = data.get('features', data) if isinstance(data, dict) else data
        for feature in features:
            props = feature.get('properties', {})
            ts_str = props.get('timestamp')
            extrinsic = props.get('extrinsic', {})
            if not ts_str or not extrinsic:
                continue
            ts_ns = int(ts_str)
            rotation_matrix = None
            R_data = extrinsic.get('Rwc_mat_row_major')
            if R_data:
                rotation_matrix = np.array(R_data).reshape(3, 3)
            else:
                quat_wxyz = extrinsic.get('Rwc_quat_wxyz')
                if quat_wxyz:
                    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
                    rotation_matrix = R.from_quat(quat_xyzw).as_matrix()
            center = extrinsic.get('center')
            if rotation_matrix is None or center is None:
                continue
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = rotation_matrix
            T[:3, 3] = np.array(center)
            self.ego_poses[ts_ns] = T
        logger.info(f"加载了 {len(self.ego_poses)} 个位姿点")
        if len(self.ego_poses) >= 2:
            self.interpolator = EgoPoseInterpolator(self.ego_poses)
        else:
            raise ValueError("位姿数据不足（需要至少2个点）")

    def _scan_lidar_files(self):
        cloud_dir = self.extracted_dir / 'cloud_body'
        if not cloud_dir.exists():
            logger.warning(f"LiDAR 目录不存在: {cloud_dir}")
            return
        pcd_files = sorted(cloud_dir.glob('*.pcd'))
        self.lidar_timestamps = [int(f.stem) for f in pcd_files]
        logger.info(f"找到 {len(self.lidar_timestamps)} 个 LiDAR 帧")

    def _normalize_to_frame0(self):
        """将所有位姿变换到以第 0 帧 LiDAR 为原点的局部坐标系，避免大数值 float32 精度损失。"""
        if not self.lidar_timestamps:
            return
        valid = [ts for ts in self.lidar_timestamps
                 if self.interpolator.min_ts <= ts <= self.interpolator.max_ts]
        if not valid:
            raise ValueError("没有 LiDAR 帧的时间戳在位姿范围内")
        n_dropped = len(self.lidar_timestamps) - len(valid)
        if n_dropped:
            logger.warning(f"过滤掉 {n_dropped} 个超出位姿范围的 LiDAR 帧")
            self.lidar_timestamps = valid
        ts0 = self.lidar_timestamps[0]
        T0 = self.interpolator.interpolate(ts0).astype(np.float64) @ self.lidar_to_ego.astype(np.float64)
        T_inv = np.linalg.inv(T0)
        self.ego_poses = {ts: (T_inv @ pose.astype(np.float64)).astype(np.float32)
                         for ts, pose in self.ego_poses.items()}
        self.interpolator = EgoPoseInterpolator(self.ego_poses)
        logger.info(f"位姿已归一化到第 0 帧坐标系")

    def load_points(self, timestamp_ns: int) -> Optional[np.ndarray]:
        pcd_file = self.extracted_dir / 'cloud_body' / f'{timestamp_ns}.pcd'
        if not pcd_file.exists():
            return None
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(pcd_file))
            return np.array(pcd.points, dtype=np.float32)
        except Exception as e:
            logger.error(f"读取 PCD 失败 {pcd_file}: {e}")
            return None

    def transform_to_world(self, points: np.ndarray, ts: int) -> Optional[np.ndarray]:
        ego_to_world = self.interpolator.interpolate(ts)
        if ego_to_world is None:
            return None
        lidar_to_world = ego_to_world @ self.lidar_to_ego
        Rmat = lidar_to_world[:3, :3]
        t = lidar_to_world[:3, 3]
        return (Rmat @ points.T + t[:, None]).T.astype(np.float32)

    def get_frame_count(self) -> int:
        return len(self.lidar_timestamps)


# ---------------------------------------------------------------------------
# 点云保存（标准 RGB PLY）
# ---------------------------------------------------------------------------

def save_pointcloud_ply(xyz_np: np.ndarray, rgb_np: np.ndarray, save_path: str):
    """保存标准彩色点云 PLY（xyz + uint8 rgb），CloudCompare / MeshLab 可直接读取。"""
    N = len(xyz_np)
    rgb_u8 = (np.clip(rgb_np, 0.0, 1.0) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    try:
        from plyfile import PlyData, PlyElement
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        verts = np.empty(N, dtype=dtype)
        verts['x'], verts['y'], verts['z']         = xyz_np[:, 0], xyz_np[:, 1], xyz_np[:, 2]
        verts['red'], verts['green'], verts['blue'] = rgb_u8[:, 0], rgb_u8[:, 1], rgb_u8[:, 2]
        PlyData([PlyElement.describe(verts, 'vertex')]).write(save_path)
        logger.info(f"      → PLY: {N:,} pts  {save_path}")
    except ImportError:
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_np)
            pcd.colors = o3d.utility.Vector3dVector(rgb_np.clip(0, 1).astype(np.float64))
            o3d.io.write_point_cloud(save_path, pcd)
            logger.info(f"      → PLY: {N:,} pts  {save_path}")
        except ImportError:
            logger.error("plyfile 和 open3d 均未安装，无法保存点云。pip install plyfile")


def _frame_color(idx: int, n_total: int) -> np.ndarray:
    """按帧索引生成 rainbow 颜色，返回 float32 [3]。"""
    hue = idx / max(n_total, 1)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
    return np.array([r, g, b], dtype=np.float32)


# ---------------------------------------------------------------------------
# 相机标定加载与视锥过滤
# ---------------------------------------------------------------------------

class FrustumFilter:
    """前向相机视锥过滤器：保留在 frame-0 相机视野内的点。

    GT 点云已归一化到 frame-0 body 坐标系，因此 frame-0 相机 pose 就是
    calibration 中固定的 T_cam2body，直接用 inv(T_cam2body) 投影即可。
    """

    def __init__(self, calib_dir: Path, cam_name: str = 'camera_front_wide'):
        yaml_path = calib_dir / f'{cam_name}.yaml'
        if not yaml_path.exists():
            raise FileNotFoundError(f"相机标定文件不存在: {yaml_path}")

        import yaml
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        # sensor-to-body transform (Rodrigues rotation vector)
        r_s2b = np.array(cfg['r_s2b'], dtype=np.float64)
        t_s2b = np.array(cfg['t_s2b'], dtype=np.float64)
        R_mat = R.from_rotvec(r_s2b).as_matrix()
        T_cam2body = np.eye(4, dtype=np.float64)
        T_cam2body[:3, :3] = R_mat
        T_cam2body[:3, 3] = t_s2b
        self.T_body2cam = np.linalg.inv(T_cam2body).astype(np.float32)

        self.fx  = float(cfg['fx'])
        self.fy  = float(cfg['fy'])
        self.cx  = float(cfg['cx'])
        self.cy  = float(cfg['cy'])
        self.w   = int(cfg['width'])
        self.h   = int(cfg['height'])
        self.kc2 = float(cfg.get('kc2', 0.0))
        self.kc3 = float(cfg.get('kc3', 0.0))
        self.kc4 = float(cfg.get('kc4', 0.0))
        self.kc5 = float(cfg.get('kc5', 0.0))
        logger.info(f"FrustumFilter({cam_name}): {self.w}x{self.h}  "
                    f"fx={self.fx:.0f}  fy={self.fy:.0f}")

    def mask(self, points_world: np.ndarray, max_depth: float = 80.0) -> np.ndarray:
        """返回 bool mask，True 表示该点落在相机视锥内。"""
        N = len(points_world)
        if N == 0:
            return np.zeros(N, dtype=bool)

        pts_h = np.hstack([points_world, np.ones((N, 1), dtype=np.float32)])
        p_cam = (self.T_body2cam @ pts_h.T)[:3, :].T  # (N, 3)

        z = p_cam[:, 2]
        valid = (z > 0.1) & (z <= max_depth)
        if not valid.any():
            return valid

        pv = p_cam[valid]
        x_n = pv[:, 0] / pv[:, 2]
        y_n = pv[:, 1] / pv[:, 2]

        # polynomial radial distortion: polyn model (kc2=k1, kc3=k2, ...)
        r2 = x_n**2 + y_n**2
        d = 1.0 + self.kc2*r2 + self.kc3*r2**2 + self.kc4*r2**3 + self.kc5*r2**4
        u = self.fx * x_n * d + self.cx
        v = self.fy * y_n * d + self.cy

        in_img = (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
        valid[valid] = in_img
        return valid

    def apply(self, points_world: np.ndarray, max_depth: float = 80.0) -> np.ndarray:
        return points_world[self.mask(points_world, max_depth)]
# ---------------------------------------------------------------------------

def process_scene(scene_path: str, output_dir: str,
                  frame_skip: int = 1, frame_indices: Optional[List[int]] = None,
                  max_frames: int = 0, save_merged: bool = False,
                  reference_frame: Optional[int] = None,
                  sv_filters: Optional[List[FrustumFilter]] = None,
                  sv_max_depth: float = 80.0) -> int:
    """读取单个 CVG 场景的 LiDAR 数据，变换到 world 坐标系后保存彩色点云 PLY。"""
    scene_start = time.time()
    scene_name = os.path.basename(scene_path)

    logger.info(f"  Loading scene: {scene_path}")
    reader = CVGLiDARReader(Path(scene_path))
    total_frames = reader.get_frame_count()

    if total_frames == 0:
        logger.warning(f"  {scene_name}: 无 LiDAR 帧，跳过")
        return 0

    # 确定要处理的帧索引（对应 lidar_timestamps 的位置）
    if frame_indices is not None:
        keep = [i for i in frame_indices if i < total_frames]
    elif frame_skip > 1:
        keep = list(range(0, total_frames, frame_skip))
    elif max_frames > 0:
        indices = np.linspace(0, total_frames - 1, min(total_frames, max_frames), dtype=int)
        keep = [int(i) for i in indices]
    else:
        keep = list(range(total_frames))

    logger.info(f"  {scene_name}: {len(keep)}/{total_frames} frames")

    # 参考帧变换：world → reference_frame 坐标系
    world_to_ref = None
    if reference_frame is not None:
        if reference_frame >= total_frames:
            logger.error(f"  参考帧索引 {reference_frame} 超出范围（共 {total_frames} 帧）")
            return 0
        ref_ts = reader.lidar_timestamps[reference_frame]
        ref_ego_to_world = reader.interpolator.interpolate(ref_ts)
        if ref_ego_to_world is None:
            logger.error(f"  参考帧 {reference_frame} 位姿插值失败")
            return 0
        world_to_ref = np.linalg.inv(ref_ego_to_world @ reader.lidar_to_ego)
        logger.info(f"  Reference frame: {reference_frame} (ts={ref_ts})")

    saved = 0
    merged_xyz_list = []
    merged_rgb_list = []
    n_keep = len(keep)

    for sample_idx, frame_idx in enumerate(keep):
        ts = reader.lidar_timestamps[frame_idx]
        points = reader.load_points(ts)
        if points is None:
            logger.warning(f"    frame {frame_idx} (ts={ts}): 加载失败，跳过")
            continue

        points_world = reader.transform_to_world(points, ts)
        if points_world is None:
            logger.warning(f"    frame {frame_idx} (ts={ts}): 位姿插值失败，跳过")
            continue

        if world_to_ref is not None:
            R_ref = world_to_ref[:3, :3]
            t_ref = world_to_ref[:3, 3]
            points_world = (R_ref @ points_world.T + t_ref[:, None]).T.astype(np.float32)

        # 视锥过滤：保留各相机视野并集内的点
        if sv_filters:
            combined = np.zeros(len(points_world), dtype=bool)
            for f in sv_filters:
                combined |= f.mask(points_world, sv_max_depth)
            points_world = points_world[combined]
            if len(points_world) == 0:
                logger.warning(f"    frame {frame_idx}: 视锥内无点，跳过")
                continue

        color = _frame_color(sample_idx, n_keep)
        rgb = np.tile(color, (len(points_world), 1))

        sample_dir = os.path.join(output_dir, scene_name, 'merge')
        sv_tag = '_sv' if sv_filters else ''
        if reference_frame is not None:
            fname = f'{frame_idx}_ref{reference_frame}{sv_tag}.ply'
        else:
            fname = f'{frame_idx}{sv_tag}.ply'
        save_pointcloud_ply(points_world, rgb, os.path.join(sample_dir, fname))
        saved += 1

        if save_merged:
            merged_xyz_list.append(points_world)
            merged_rgb_list.append(rgb)

    if save_merged and merged_xyz_list:
        merged_xyz = np.vstack(merged_xyz_list)
        merged_rgb = np.vstack(merged_rgb_list)
        sv_tag = '_sv' if sv_filters else ''
        if reference_frame is not None:
            merged_name = f'merged_pointcloud_ref{reference_frame}{sv_tag}.ply'
        else:
            merged_name = f'merged_pointcloud{sv_tag}.ply'
        merged_path = os.path.join(output_dir, scene_name, 'merge', merged_name)
        logger.info(f"  Saving merged pointcloud: {len(merged_xyz):,} pts → {merged_path}")
        save_pointcloud_ply(merged_xyz, merged_rgb, merged_path)

    elapsed = time.time() - scene_start
    logger.info(f"  {scene_name}: saved {saved} frames  ({elapsed:.1f}s)")
    return saved


# ---------------------------------------------------------------------------
# 场景路径解析
# ---------------------------------------------------------------------------

def resolve_scene_dirs(scene_arg: str, data_root: Path) -> List[str]:
    """
    解析场景参数，返回场景目录绝对路径列表：
      'all'        → data_root 下全部 scene_* 子目录
      '0'          → data_root/scene_000
      'scene_000'  → data_root/scene_000
      '0,1,2'      → [data_root/scene_000, ...]
    """
    if scene_arg.strip().lower() == 'all':
        dirs = sorted([
            str(data_root / d)
            for d in os.listdir(data_root)
            if d.startswith('scene_') and (data_root / d).is_dir()
        ])
        return dirs

    parts = [p.strip() for p in scene_arg.split(',')]
    result = []
    for p in parts:
        name = f"scene_{int(p):03d}" if p.isdigit() else p
        path = data_root / name
        if not path.is_dir():
            raise FileNotFoundError(
                f"场景目录不存在: {path}\n  data_root: {data_root}"
            )
        result.append(str(path))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='CVG 场景点云保存：读取 LiDAR 数据并保存标准彩色点云 PLY'
    )
    parser.add_argument('--scene', required=True,
                        help="场景名（scene_000）、纯数字（0）、逗号分隔（0,1,2）或 all")
    parser.add_argument('--data_root', default=str(DATA_ROOT),
                        help=f'CVG 数据根目录（默认 {DATA_ROOT}）')
    parser.add_argument('--output_dir', default='../cvg_data_pipeline/data',
                        help='输出根目录（默认 ../cvg_data_pipeline/data）')

    # 帧选择
    parser.add_argument('--frame_skip', type=int, default=1,
                        help='每隔 N 帧取一帧（默认 1 = 全部）')
    parser.add_argument('--frames', type=str, default=None,
                        help='指定帧索引，逗号分隔，优先级高于 --frame_skip')
    parser.add_argument('--max_frames', type=int, default=0,
                        help='最多处理帧数（均匀采样），0 = 不限制')

    # 合并
    parser.add_argument('--save_merged', action='store_true', default=True,
                        help='额外保存场景合并点云 merged_pointcloud.ply（默认开启）')
    parser.add_argument('--reference_frame', type=int, default=None,
                        help='参考帧索引，指定后所有帧变换到该帧坐标系，合并点云命名为 merged_pointcloud_ref<N>.ply')

    # 视锥过滤
    parser.add_argument('--singleview', action='store_true', default=False,
                        help='只保留指定相机视野并集内的点（用于多视角评测），输出文件名加 _sv 后缀')
    parser.add_argument('--cam_name', default='camera_front_wide',
                        help='--singleview 使用的相机名，逗号分隔可指定多个（默认 camera_front_wide）')
    parser.add_argument('--sv_max_depth', type=float, default=80.0,
                        help='--singleview 深度截断（米，默认 80）')

    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        logger.error(f"数据根目录不存在: {data_root}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output: {args.output_dir}")

    # 帧列表
    frame_indices = None
    if args.frames:
        frame_indices = [int(x.strip()) for x in args.frames.split(',')]
        logger.info(f"Frames (explicit): {frame_indices}")
    elif args.frame_skip > 1:
        logger.info(f"Frame skip: every {args.frame_skip} frames")
    elif args.max_frames > 0:
        logger.info(f"Max frames: {args.max_frames}")

    # 解析场景目录
    try:
        scene_dirs = resolve_scene_dirs(args.scene, data_root)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Scenes: {len(scene_dirs)}")

    # 构建视锥过滤器（各场景共用同一相机标定）
    sv_filters = None
    if args.singleview:
        first_calib_dir = Path(scene_dirs[0]) / 'extracted' / 'calibration'
        cam_names = [c.strip() for c in args.cam_name.split(',')]
        sv_filters = [FrustumFilter(first_calib_dir, c) for c in cam_names]
        logger.info(f"Singleview mode: {cam_names}  max_depth={args.sv_max_depth} m")

    t0 = time.time()
    summary = []

    for scene_path in scene_dirs:
        scene_name = os.path.basename(scene_path)
        logger.info(f"\n{'='*50}")
        logger.info(f"Scene: {scene_name}")

        try:
            saved = process_scene(
                scene_path=scene_path,
                output_dir=args.output_dir,
                frame_skip=args.frame_skip,
                frame_indices=frame_indices,
                max_frames=args.max_frames,
                save_merged=args.save_merged,
                reference_frame=args.reference_frame,
                sv_filters=sv_filters,
                sv_max_depth=args.sv_max_depth,
            )
        except Exception as e:
            logger.error(f"  {scene_name} 处理失败: {e}")
            import traceback; traceback.print_exc()
            saved = 0

        summary.append({'scene': scene_name, 'saved': saved})

    elapsed = time.time() - t0
    total_saved = sum(r['saved'] for r in summary)

    summary_data = {
        'total_scenes':  len(summary),
        'total_samples': total_saved,
        'total_time_s':  round(elapsed, 2),
        'data_root':     str(data_root),
        'frame_skip':    args.frame_skip,
        'frames':        args.frames,
        'max_frames':    args.max_frames,
        'scenes':        summary,
    }
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Done: {len(summary)} scenes, {total_saved} frames ({elapsed:.1f}s)")
    logger.info(f"Summary → {summary_path}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()

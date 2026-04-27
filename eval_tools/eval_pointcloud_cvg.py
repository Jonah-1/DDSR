#!/usr/bin/env python3
#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

"""
CVG 场景点云重建质量批量评测脚本。

计算四项标准指标：
  - Accuracy      （准确度）：预测点云中每个点到真实点云最近邻的距离
  - Completeness  （完整性）：真实点云中每个点到预测点云最近邻的距离
  - F1 Score               ：Accuracy 与 Completeness 的调和平均值
  - Chamfer Distance (CD)  ：L1-CD 与 L2-CD

目录结构约定（pred 与 gt 镜像）：
  <pred_dir>/<scene_name>/sample_<idx>/pointcloud.ply
  <gt_dir>/<scene_name>/sample_<idx>/pointcloud.ply

或使用 --merged 模式：
  <pred_dir>/<scene_name>/merged_pointcloud.ply
  <gt_dir>/<scene_name>/merged_pointcloud.ply

用法示例：
  # 评测 scene_000 逐帧点云
  python eval_tools/eval_pointcloud_cvg.py \\
      --pred_dir work_dirs/cvg_1cam_eval/pointcloud \\
      --gt_dir /path/to/gt_pointclouds \\
      --scene scene_000

  # 评测所有场景，使用合并点云
  python eval_tools/eval_pointcloud_cvg.py \\
      --pred_dir work_dirs/cvg_1cam_eval/pointcloud \\
      --gt_dir /path/to/gt_pointclouds \\
      --scene all \\
      --merged

  # 指定距离阈值（米），降采样上限
  python eval_tools/eval_pointcloud_cvg.py \\
      --pred_dir work_dirs/cvg_1cam_eval/pointcloud \\
      --gt_dir /path/to/gt_pointclouds \\
      --scene scene_000 \\
      --threshold 0.1 \\
      --max_points 500000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 默认路径（相对于 ReconDrive/ 工作目录）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_SCRIPT_DIR)
DEFAULT_PRED_DIR = os.path.join(_REPO_ROOT, "work_dirs", "cvg_1cam_eval", "pointcloud")
DEFAULT_GT_ROOT  = os.path.join(_REPO_ROOT, "..", "cvg_data_pipeline", "data")


# ──────────────────────────────────────────────────────────────────────────────
# 点云加载
# ──────────────────────────────────────────────────────────────────────────────

def load_pointcloud(path: str, max_points: Optional[int] = None) -> np.ndarray:
    """从文件加载点云，返回形状 (N, 3) 的 float32 数组。支持 .ply / .pcd / .xyz / .npy / .npz。"""
    suffix = path.rsplit(".", 1)[-1].lower()

    if suffix in {"ply", "pcd"}:
        # 优先尝试 plyfile（轻量，无需 open3d）
        if suffix == "ply":
            try:
                from plyfile import PlyData
                plydata = PlyData.read(path)
                v = plydata["vertex"]
                pts = np.stack([np.asarray(v["x"]),
                                np.asarray(v["y"]),
                                np.asarray(v["z"])], axis=1).astype(np.float32)
            except Exception:
                pts = _load_with_open3d(path)
        else:
            pts = _load_with_open3d(path)

    elif suffix == "xyz":
        pts = np.loadtxt(path, dtype=np.float32)
        pts = pts[:, :3]

    elif suffix == "npy":
        pts = np.load(path).astype(np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)
        pts = pts[:, :3]

    elif suffix == "npz":
        data = np.load(path)
        key = next((k for k in ("points", "pts", "xyz") if k in data), None)
        if key is None:
            key = list(data.keys())[0]
        pts = data[key].astype(np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)
        pts = pts[:, :3]

    else:
        raise ValueError(f"不支持的文件格式：.{suffix}，支持：ply / pcd / xyz / npy / npz")

    if len(pts) == 0:
        raise ValueError(f"点云文件为空：{path}")

    if max_points is not None and len(pts) > max_points:
        idx = np.random.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]

    return pts


def _load_with_open3d(path: str) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("读取该格式需要 open3d，请执行 pip install open3d")
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 核心指标计算
# ──────────────────────────────────────────────────────────────────────────────

def compute_nn_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """对 source 每个点查找 target 中最近邻距离，返回 (N,) float32 数组。"""
    from scipy.spatial import KDTree
    tree = KDTree(target)
    dists, _ = tree.query(source, workers=-1)
    return dists.astype(np.float32)


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float = 0.05,
) -> Dict[str, float]:
    """计算点云重建指标：Accuracy / Completeness / F1 / Chamfer Distance L1。"""
    pred_to_gt = compute_nn_distances(pred, gt)
    gt_to_pred = compute_nn_distances(gt, pred)

    accuracy      = float((pred_to_gt < threshold).mean())
    mean_acc_dist = float(pred_to_gt.mean())

    completeness   = float((gt_to_pred < threshold).mean())
    mean_comp_dist = float(gt_to_pred.mean())

    f1 = (2.0 * accuracy * completeness / (accuracy + completeness)
          if accuracy + completeness > 0 else 0.0)

    cd = (mean_acc_dist + mean_comp_dist) / 2.0

    return {
        "accuracy":       accuracy,
        "completeness":   completeness,
        "f1":             f1,
        "cd":             cd,
        "mean_acc_dist":  mean_acc_dist,
        "mean_comp_dist": mean_comp_dist,
        "threshold":      threshold,
        "n_pred":         int(len(pred)),
        "n_gt":           int(len(gt)),
    }


def aggregate_metrics(metric_list: List[Dict]) -> Dict[str, float]:
    """对多帧指标字典列表求均值。"""
    if not metric_list:
        return {}
    scalar_keys = ["accuracy", "completeness", "f1", "cd",
                   "mean_acc_dist", "mean_comp_dist"]
    agg = {}
    for k in scalar_keys:
        vals = [m[k] for m in metric_list if k in m]
        agg[k] = float(np.mean(vals)) if vals else 0.0
    agg["n_samples"] = len(metric_list)
    agg["threshold"] = metric_list[0].get("threshold", 0.05)
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# 打印工具
# ──────────────────────────────────────────────────────────────────────────────

_SEP = "─" * 60


def print_sample_result(metrics: Dict, label: str) -> None:
    print(f"  [{label}]  "
          f"Acc={metrics['accuracy']*100:.1f}%  "
          f"Comp={metrics['completeness']*100:.1f}%  "
          f"F1={metrics['f1']*100:.1f}%  "
          f"CD={metrics['cd']:.5f}  "
          f"(pred={metrics['n_pred']:,}  gt={metrics['n_gt']:,})")


def print_scene_summary(agg: Dict, scene_name: str) -> None:
    print(_SEP)
    print(f"  场景汇总：{scene_name}  （{agg['n_samples']} 个样本）")
    print(_SEP)
    print(f"  Accuracy     （准确度） : {agg['accuracy'] * 100:.2f} %")
    print(f"  Completeness （完整性） : {agg['completeness'] * 100:.2f} %")
    print(f"  F1 Score               : {agg['f1'] * 100:.2f} %")
    print(_SEP)
    print(f"  Chamfer Distance       : {agg['cd']:.6f}")
    print(f"  预测→真实 平均距离     : {agg['mean_acc_dist']:.6f}")
    print(f"  真实→预测 平均距离     : {agg['mean_comp_dist']:.6f}")
    print(_SEP)


def print_overall_summary(agg: Dict) -> None:
    print(_SEP)
    print(f"  全局汇总  （{agg['n_samples']} 个场景）")
    print(_SEP)
    print(f"  Accuracy     （准确度） : {agg['accuracy'] * 100:.2f} %")
    print(f"  Completeness （完整性） : {agg['completeness'] * 100:.2f} %")
    print(f"  F1 Score               : {agg['f1'] * 100:.2f} %")
    print(_SEP)
    print(f"  Chamfer Distance       : {agg['cd']:.6f}")
    print(_SEP)


# ──────────────────────────────────────────────────────────────────────────────
# 场景发现与匹配
# ──────────────────────────────────────────────────────────────────────────────

def discover_scenes(pred_dir: str) -> List[str]:
    """返回 pred_dir 下所有 scene_* 子目录的名称列表（排序）。"""
    entries = sorted(
        e for e in os.listdir(pred_dir)
        if os.path.isdir(os.path.join(pred_dir, e))
    )
    return entries


def resolve_scene_names(scene_arg: str, pred_dir: str) -> List[str]:
    """
    根据命令行 --scene 参数解析场景名列表：
      'all'        → pred_dir 下全部子目录
      '0'          → 'scene_000'
      'scene_000'  → ['scene_000']
      '0,1,2'      → ['scene_000', 'scene_001', 'scene_002']
      'scene_000,scene_001' → ['scene_000', 'scene_001']
    """
    if scene_arg.lower() == "all":
        return discover_scenes(pred_dir)

    parts = [p.strip() for p in scene_arg.split(",")]
    names = []
    for p in parts:
        if p.isdigit():
            names.append(f"scene_{int(p):03d}")
        else:
            names.append(p)
    return names


def find_sample_pairs(pred_scene_dir: str, gt_scene_dir: str,
                      pc_filename: str = "pointcloud.ply") -> List[Tuple[str, str, str]]:
    """
    在 pred_scene_dir 下找所有 sample_*/pointcloud.ply，
    并在 gt_scene_dir 下寻找对应文件。

    返回 [(sample_name, pred_path, gt_path), ...] 只包含两端都存在的样本。
    """
    pattern = os.path.join(pred_scene_dir, "sample_*", pc_filename)
    pred_files = sorted(glob.glob(pattern))

    pairs = []
    for pred_path in pred_files:
        sample_name = os.path.basename(os.path.dirname(pred_path))
        gt_path = os.path.join(gt_scene_dir, sample_name, pc_filename)
        if os.path.isfile(gt_path):
            pairs.append((sample_name, pred_path, gt_path))
        else:
            print(f"  [WARN] GT 不存在，跳过：{gt_path}")
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# 场景评测
# ──────────────────────────────────────────────────────────────────────────────

def eval_scene(
    scene_name: str,
    pred_dir: str,
    gt_dir: str,
    merged: bool,
    threshold: float,
    max_points: Optional[int],
    gt_file: Optional[str] = None,
) -> Optional[Dict]:
    """
    评测单个场景，返回 {scene_name, samples, aggregate} 的结果字典。
    若无有效样本则返回 None。

    gt_file: 若非 None，则直接用该文件作为 GT（忽略 gt_dir 目录结构）。
    """
    pred_scene = os.path.join(pred_dir, scene_name)

    if not os.path.isdir(pred_scene):
        print(f"[SKIP] 预测目录不存在：{pred_scene}")
        return None

    gt_label = gt_file if gt_file else os.path.join(gt_dir, scene_name)
    print(f"\n{'='*60}")
    print(f"  评测场景：{scene_name}")
    print(f"  pred: {pred_scene}")
    print(f"  gt  : {gt_label}")
    print(f"{'='*60}")

    # 若 gt_file 指定，预先加载一次 GT（所有样本共用）
    shared_gt_pts: Optional[np.ndarray] = None
    if gt_file is not None:
        try:
            shared_gt_pts = load_pointcloud(gt_file, max_points)
        except Exception as e:
            print(f"  [ERR ] 加载 GT 文件失败：{e}")
            return None

    sample_results = {}

    if merged:
        # ── 合并点云模式 ────────────────────────────────────────────────────
        pred_path = os.path.join(pred_scene, "merged_pointcloud.ply")
        if not os.path.isfile(pred_path):
            print(f"  [SKIP] merged_pointcloud.ply 不存在：{pred_path}")
            return None

        if shared_gt_pts is not None:
            gt_pts = shared_gt_pts
        else:
            gt_scene = os.path.join(gt_dir, scene_name)
            gt_path  = os.path.join(gt_scene, "merged_pointcloud.ply")
            if not os.path.isfile(gt_path):
                print(f"  [SKIP] GT merged_pointcloud.ply 不存在：{gt_path}")
                return None
            gt_pts = load_pointcloud(gt_path, max_points)

        pred_pts = load_pointcloud(pred_path, max_points)
        m = compute_metrics(pred_pts, gt_pts, threshold)
        sample_results["merged"] = m
        print_sample_result(m, "merged")

    else:
        # ── 逐帧模式 ────────────────────────────────────────────────────────
        if shared_gt_pts is not None:
            # GT 为单文件时，直接枚举 pred 下所有 sample_*/pointcloud.ply
            pattern = os.path.join(pred_scene, "sample_*", "pointcloud.ply")
            pred_files = sorted(glob.glob(pattern))
            if not pred_files:
                print(f"  [SKIP] 未找到任何预测样本")
                return None
            pairs_iter = [
                (os.path.basename(os.path.dirname(p)), p, None)
                for p in pred_files
            ]
        else:
            gt_scene = os.path.join(gt_dir, scene_name)
            if not os.path.isdir(gt_scene):
                print(f"  [SKIP] GT 目录不存在：{gt_scene}")
                return None
            pairs_iter = find_sample_pairs(pred_scene, gt_scene)
            if not pairs_iter:
                print(f"  [SKIP] 未找到任何匹配样本对")
                return None

        for sample_name, pred_path, gt_path in pairs_iter:
            try:
                pred_pts = load_pointcloud(pred_path, max_points)
                gt_pts   = shared_gt_pts if shared_gt_pts is not None else load_pointcloud(gt_path, max_points)
            except Exception as e:
                print(f"  [ERR ] {sample_name}: {e}")
                continue
            m = compute_metrics(pred_pts, gt_pts, threshold)
            sample_results[sample_name] = m
            print_sample_result(m, sample_name)

    if not sample_results:
        return None

    agg = aggregate_metrics(list(sample_results.values()))
    print_scene_summary(agg, scene_name)

    return {
        "scene":     scene_name,
        "samples":   sample_results,
        "aggregate": agg,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="批量评测 CVG 场景点云重建质量（Accuracy / Completeness / F1 / Chamfer Distance）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pred_dir", default=None,
        help="预测点云根目录（含 scene_xxx/ 子目录），通常为 work_dirs/cvg_1cam_eval/pointcloud/；--auto 时可省略",
    )
    parser.add_argument(
        "--gt_dir", default=None,
        help=(
            "真实点云路径：可以是目录（结构与 pred_dir 镜像），"
            "也可以直接是单个 .ply 文件；--auto 时可省略"
        ),
    )
    parser.add_argument(
        "--auto", action="store_true",
        help=(
            "自动模式：只需指定 --scene，自动从 work_dirs/cvg_1cam_eval/pointcloud/<scene>/merged_pointcloud.ply "
            "读取预测，从 ../cvg_data_pipeline/data/<scene>/merge/merged_pointcloud.ply 读取真值，隐式启用 --merged"
        ),
    )
    parser.add_argument(
        "--scene", default="all",
        help=(
            "要评测的场景：'all'=全部，纯数字如 '0'=scene_000，"
            "逗号分隔如 '0,1,2' 或 'scene_000,scene_001'"
        ),
    )
    parser.add_argument(
        "--merged", action="store_true",
        help="评测 merged_pointcloud.ply 而非逐帧 sample_xxxx/pointcloud.ply",
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.05,
        help="判定'正确'的距离阈值（与点云坐标单位相同，通常为米）",
    )
    parser.add_argument(
        "--max_points", "-n", type=int, default=200_000,
        help="每个点云最多保留的点数（随机降采样），0 表示不限制",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子，保证降采样可复现",
    )
    parser.add_argument(
        "--output_json", default=None,
        help="将评测结果保存为 JSON 文件（默认保存到 pred_dir/eval_results.json）",
    )
    parser.add_argument(
        "--no_save", action="store_true",
        help="不保存 JSON，只打印结果",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    max_pts = args.max_points if args.max_points > 0 else None

    # ── 直接文件对比模式：pred 和 gt 均为 .ply 文件 ───────────────────────────
    if (args.pred_dir and os.path.isfile(args.pred_dir) and
            args.gt_dir and os.path.isfile(args.gt_dir)):
        pred_file = os.path.abspath(args.pred_dir)
        gt_file   = os.path.abspath(args.gt_dir)
        print(f"评测配置（直接文件模式）：")
        print(f"  pred : {pred_file}")
        print(f"  gt   : {gt_file}")
        print(f"  阈值 : {args.threshold} m  最大点数: {max_pts or '不限'}")

        try:
            pred_pts = load_pointcloud(pred_file, max_pts)
            gt_pts   = load_pointcloud(gt_file,   max_pts)
        except Exception as e:
            print(f"[ERROR] 加载点云失败：{e}", file=sys.stderr)
            sys.exit(1)

        m = compute_metrics(pred_pts, gt_pts, args.threshold)
        print_sample_result(m, "direct")
        agg = aggregate_metrics([m])
        print_scene_summary(agg, "direct")

        if not args.no_save:
            save_path = args.output_json or os.path.join(
                os.path.dirname(pred_file), "eval_results.json")
            output = {
                "config": {
                    "pred_file": pred_file, "gt_file": gt_file,
                    "threshold": args.threshold, "max_points": max_pts,
                    "seed": args.seed,
                },
                "result": m,
            }
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存至：{save_path}")
        return

    # ── --auto 模式：自动推导 pred_dir / gt_root，强制 merged 模式 ──────────────
    if args.auto:
        pred_dir = os.path.abspath(args.pred_dir or DEFAULT_PRED_DIR)
        gt_root  = os.path.abspath(args.gt_dir  or DEFAULT_GT_ROOT)
        merged   = True
    else:
        if not args.pred_dir or not args.gt_dir:
            print("[ERROR] 请指定 --pred_dir 和 --gt_dir，或使用 --auto 模式", file=sys.stderr)
            sys.exit(1)
        pred_dir = os.path.abspath(args.pred_dir)
        gt_root  = None
        merged   = args.merged

    if not os.path.isdir(pred_dir):
        print(f"[ERROR] 预测目录不存在（也不是 .ply 文件）：{pred_dir}", file=sys.stderr)
        sys.exit(1)

    scene_names = resolve_scene_names(args.scene, pred_dir)
    if not scene_names:
        print("[ERROR] 未找到任何场景", file=sys.stderr)
        sys.exit(1)

    # non-auto 模式：走原有逻辑
    if not args.auto:
        gt_path   = os.path.abspath(args.gt_dir)
        gt_is_file = os.path.isfile(gt_path)
        gt_is_dir  = os.path.isdir(gt_path)
        if not gt_is_file and not gt_is_dir:
            print(f"[ERROR] GT 路径不存在：{gt_path}", file=sys.stderr)
            sys.exit(1)
        gt_file_arg = gt_path if gt_is_file else None
        gt_dir_arg  = gt_path if gt_is_dir  else ""

        print(f"评测配置：")
        print(f"  pred_dir   : {pred_dir}")
        if gt_is_file:
            print(f"  gt_file    : {gt_path}  （单文件，所有场景共用）")
        else:
            print(f"  gt_dir     : {gt_path}")
        print(f"  场景列表   : {scene_names}")
        print(f"  模式       : {'合并点云' if merged else '逐帧点云'}")
        print(f"  阈值       : {args.threshold} m")
        print(f"  最大点数   : {max_pts or '不限'}")

        all_results = {}
        scene_agg_list = []
        for sname in scene_names:
            result = eval_scene(
                scene_name=sname,
                pred_dir=pred_dir,
                gt_dir=gt_dir_arg,
                merged=merged,
                threshold=args.threshold,
                max_points=max_pts,
                gt_file=gt_file_arg,
            )
            if result is not None:
                all_results[sname] = result
                scene_agg_list.append(result["aggregate"])

        save_gt_ref = gt_path
    else:
        # ── auto 模式：每个场景单独指定 GT 文件 ─────────────────────────────────
        print(f"评测配置（auto 模式）：")
        print(f"  pred_dir   : {pred_dir}")
        print(f"  gt_root    : {gt_root}")
        print(f"  场景列表   : {scene_names}")
        print(f"  模式       : 合并点云（自动）")
        print(f"  阈值       : {args.threshold} m")
        print(f"  最大点数   : {max_pts or '不限'}")

        all_results = {}
        scene_agg_list = []
        for sname in scene_names:
            gt_file_auto = os.path.join(gt_root, sname, "merge", "merged_pointcloud.ply")
            if not os.path.isfile(gt_file_auto):
                print(f"[SKIP] GT 文件不存在：{gt_file_auto}")
                continue
            result = eval_scene(
                scene_name=sname,
                pred_dir=pred_dir,
                gt_dir="",
                merged=True,
                threshold=args.threshold,
                max_points=max_pts,
                gt_file=gt_file_auto,
            )
            if result is not None:
                all_results[sname] = result
                scene_agg_list.append(result["aggregate"])

        save_gt_ref = gt_root

    # ── 全局汇总 ──────────────────────────────────────────────────────────────
    if len(scene_agg_list) > 1:
        overall = aggregate_metrics(scene_agg_list)
        overall["n_samples"] = len(scene_agg_list)
        print_overall_summary(overall)
    else:
        overall = scene_agg_list[0] if scene_agg_list else {}

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    if not args.no_save:
        save_path = args.output_json or os.path.join(pred_dir, "eval_results.json")
        output = {
            "config": {
                "pred_dir":   pred_dir,
                "gt":         save_gt_ref,
                "auto":       args.auto,
                "scenes":     scene_names,
                "merged":     merged,
                "threshold":  args.threshold,
                "max_points": max_pts,
                "seed":       args.seed,
            },
            "scenes":  all_results,
            "overall": overall,
        }
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至：{save_path}")

    if not all_results:
        print("\n[WARN] 未成功评测任何场景。")
        sys.exit(1)


if __name__ == "__main__":
    main()

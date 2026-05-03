# eval_tools

ReconDrive 推理与高斯保存工具集。

> 运行目录：`FeedforwardGS-RD/ReconDrive/`

---

## 脚本总览

| 脚本 | 数据集 | 功能 |
|------|--------|------|
| `scripts/inference.py` | NuScenes / CVG | 场景推理，输出渲染图 + 指标 |
| `eval_tools/eval_rendering.py` | CVG | 场景推理，输出重建帧 / 中间帧 / 总体三项渲染指标 |
| `eval_tools/render_to_video.py` | — | 将渲染图序列转为视频，支持多视角合成 grid 视频 |
| `eval_tools/save_gaussians_cvg.py` | CVG | 逐帧保存 Gaussian PLY，可选保存合并高斯 |
| `eval_tools/save_pointcloud_cvg.py` | CVG | 逐帧保存彩色点云 PLY，可选保存合并点云 |
| `eval_tools/save_pointcloud_split.py` | CVG | 逐帧按相机分别保存点云，区分 frame i / frame i+6 坐标系 |
| `eval_tools/save_pointcloud_cvg_gt.py` | CVG | 保存 CVG 数据集 LiDAR 真值点云，支持参考帧坐标系 |
| `eval_tools/eval_pointcloud_cvg.py` | CVG | 批量评测点云重建质量（Acc / Comp / F1 / CD） |
| `eval_tools/render_split.py` | CVG | 拆分 frame 0 / frame N 高斯，渲染 4 种交叉组合图像并保存 PLY |

---

## scripts/inference.py

场景级推理脚本，支持 NuScenes 和 CVG 数据。逐场景输出渲染图像并计算 PSNR / SSIM / LPIPS。

```bash
# NuScenes
python scripts/inference.py \
    --cfg_path /home/jinheng.li/project/DDSR/configs/nuscenes/recondrive.yaml \
    --restore_ckpt  /train-syncdata/jinheng.li/project/DDSR/nucense/v1_5.1/best_module-v2.ckpt\
    --output_dir work_dirs/inference_results

# CVG（config 中 datamodule_type: cvg）
python scripts/inference.py \
    --cfg_path configs/cvg/recondrive_cvg_3cam.yaml \
    --restore_ckpt checkpoints/recondrive_stage2.ckpt \
    --output_dir work_dirs/cvg_inference_results

```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cfg_path` | 必填 | YAML 配置文件路径 |
| `--restore_ckpt` | 必填 | 模型 checkpoint 路径 |
| `--output_dir` | `<save_dir>/scene_inference_results` | 结果输出目录 |
| `--device` | config 中 devices[0] | 运行设备，如 `cuda:0` |
| `--max_scenes` | 全部 | 最多处理 N 个场景 |
| `--no_renders` | false | 只计算指标，不保存图像 |
| `--novel_distances` | `0.5,1.0,2.0,3.0` | 侧向平移距离（米），逗号分隔 |
| `--eval_resolution` | `original` | 评测分辨率，如 `280x518` |

---

## eval_tools/eval_rendering.py

CVG 场景渲染评测脚本，分别输出**重建帧**（frame 0）、**中间帧**（frames 1–5）和**总体**三套 PSNR / SSIM / LPIPS 指标。总体指标按两类帧的数量加权（直接拼接后取均值）。

同时可保存各帧渲染图及侧向平移（ego translation）的新视角图像。

```bash
# 基础评测（只计算 test split 指标，不保存渲染图）
python eval_tools/eval_rendering.py \
    --cfg_path /home/jinheng.li/project/DDSR/configs/nuscenes/recondrive.yaml \
    --restore_ckpt /train-syncdata/jinheng.li/project/DDSR/nucense/v1_5.1/best_module-v2.ckpt \
    --output_dir work_dirs/inference_novel-view_results \
    --no_renders

# 完整评测（保存所有帧的新视角渲染图 ）
python eval_tools/eval_rendering.py \
    --cfg_path configs/cvg/recondrive_cvg_3cam.yaml \
    --restore_ckpt work_dirs/cvg_cam3_training-v2/ckpt/best_module-v1.ckpt \
    --output_dir work_dirs/inference_novel-view_results \
    --novel_distances 1.0,2.0 \
    --split all

# 只跑前 2 个场景做快速验证
python eval_tools/eval_rendering.py \
    --cfg_path configs/cvg/recondrive_cvg_3cam.yaml \
    --restore_ckpt work_dirs/cvg_cam3_training-v2/ckpt/best_module.ckpt \
    --max_scenes 2 \
    --no_renders

# 评测所有数据（不过滤 split）
python eval_tools/eval_rendering.py \
    --cfg_path configs/cvg/recondrive_cvg_1cam.yaml \
    --restore_ckpt work_dirs/cvg_cam1_training/ckpt/best_module.ckpt \
    --split all \
    --no_renders
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cfg_path` | 必填 | YAML 配置文件路径 |
| `--restore_ckpt` | 必填 | 模型 checkpoint 路径 |
| `--output_dir` | `<save_dir>/scene_inference_results` | 结果输出目录 |
| `--device` | config 中 devices[0] | 运行设备，如 `cuda:0` |
| `--max_scenes` | 全部 | 最多处理 N 个场景（用于快速验证） |
| `--no_renders` | false | 只计算指标，不保存渲染图和新视角图 |
| `--novel_distances` | `0.5,1.0,2.0,3.0` | 侧向平移新视角的距离（米），逗号分隔 |
| `--eval_resolution` | `original` | 评测分辨率，如 `280x518`；`original` = 模型原始分辨率 |
| `--split` | `test` | 数据集划分：`train` / `val` / `test` / `all`（默认只跑 test 10%）|

**输出指标说明：**

- **Scene Reconstruction (Frame 0)**：frame 0 渲染 vs GT，衡量重建质量
- **Novel View Synthesis (Middle Frames)**：frames 1–5 渲染 vs GT，衡量时序新视角质量
- **Overall**：两类帧拼合后取均值，按各自帧数自然加权（1 重建帧 + 5 中间帧 = 权重比 1:5）

**输出文件：**

```
<output_dir>/
├── inference_summary.json          ← 总体指标 + 各场景摘要
├── inference_detailed.json         ← 完整逐样本数据
└── scene_<name>/
    └── sample_<N>/
        ├── gt_views/               ← 渲染图（pred/gt 对）
        │   ├── 280x518_cam_0_pred.png
        │   └── 280x518_cam_0_gt.png
        └── left_1.0m/              ← 侧向平移新视角
            └── 280x518_cam_0.png
```

---

## eval_tools/render_split.py

CVG 场景**分帧高斯渲染**脚本。模型推理输出 frame 0 与 frame N（默认 frame 6）两组高斯，本脚本将两组高斯拆开，生成 4 种交叉组合渲染图，便于分析每帧高斯的质量与对齐情况。同时将两帧高斯分别保存为标准 3DGS PLY 文件。

**每个 sample 每个相机输出 4 张渲染图：**

| 文件名 | 高斯来源 | 相机来源 | 说明 |
|--------|----------|----------|------|
| `frame0_gs_on_frame0_cam_cam_{id}.png` | frame 0 | frame 0 | 重建参考帧自身渲染 |
| `frameN_gs_on_frameN_cam_cam_{id}.png` | frame N | frame N | 另一重建帧自身渲染 |
| `frame0_gs_on_frameN_cam_cam_{id}.png` | frame 0 | frame N | frame 0 高斯在 frame N 视角下渲染 |
| `frameN_gs_on_frame0_cam_cam_{id}.png` | frame N | frame 0 | frame N 高斯在 frame 0 视角下渲染 |

```bash
python eval_tools/render_split.py \
    --cfg_path configs/cvg/recondrive_cvg_3cam.yaml \
    --restore_ckpt checkpoints/recondrive_stage2.ckpt \
    --output_dir work_dirs/split_render_results

# 只处理前 2 个场景（快速验证）
python eval_tools/render_split.py \
    --cfg_path configs/cvg/recondrive_cvg_3cam.yaml \
    --restore_ckpt /train-syncdata/jinheng.li/project/DDS/v3_4.28/best_module-v1.ckpt \
    --output_dir work_dirs/split_render_results \
    --max_scenes 2
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cfg_path` | 必填 | YAML 配置文件路径 |
| `--restore_ckpt` | 必填 | 模型 checkpoint 路径 |
| `--output_dir` | `<save_dir>/scene_inference_results` | 结果输出目录 |
| `--device` | config 中 devices[0] | 运行设备，如 `cuda:0` |
| `--max_scenes` | 全部 | 最多处理 N 个场景 |
| `--no_renders` | false | 只跑推理，不保存渲染图和 PLY |
| `--eval_resolution` | `original` | 渲染分辨率，如 `280x518` |

> **注意：** 脚本内部硬编码 `max_samples=2`（每场景仅处理前 2 个 sample）、`frame_skip=6`（每 6 帧取一帧），主要用于快速分帧质量验证。

**输出目录结构：**

```
<output_dir>/
└── scene_xxx/
    └── sample_0000/
        ├── split_frames/
        │   ├── 280x518_frame0_gs_on_frame0_cam_cam_0.png
        │   ├── 280x518_frameN_gs_on_frameN_cam_cam_0.png
        │   ├── 280x518_frame0_gs_on_frameN_cam_cam_0.png
        │   └── 280x518_frameN_gs_on_frame0_cam_cam_0.png
        └── gaussians/
            ├── frame0_gaussians.ply
            └── frameN_gaussians.ply
```

---

## eval_tools/render_to_video.py

将 `eval_rendering.py` 输出的图像序列转为视频，并支持将多个视角合成为带文字标签的 grid 视频。

依赖：`ffmpeg`、`ffprobe`、`opencv-python`。中文标签需安装 CJK 字体（`apt install fonts-noto-cjk`）。

```bash
# 查看发现的 view 类型和相机
python eval_tools/render_to_video.py \
    --input-dir work_dirs/inference_results/scene_004 \
    --list

# 单视角相机新视角视频合成
python eval_tools/render_to_video.py \
    --input-dir work_dirs/inference_novel-view_results/scene-0103 \
    --output-dir work_dirs/videos/scene_0103 \
    --fps 20 --cams 0 \
    --layout "left_1.0m,gt_views_gt,right_1.0m;left_2.0m,gt_views_pred,right_2.0m" \
    --labels "left 1m,GT,right 1m;left 2m,render,right 2m"


# 3视角相机视频合成（行=view类型，列=相机，一条命令生成 3 相机宽幅视频）
python eval_tools/render_to_video.py \
    --input-dir work_dirs/inference_novel-view_results/scene-0103 \
    --output-dir work_dirs/videos/scene_0103 \
    --fps 20 --cams 0 \
    --layout "gt_views_gt;gt_views_pred" \
    --labels "GT;Rendered" \
    --cam-per-col
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-dir` | 必填 | 包含 `sample_*` 子目录的场景目录 |
| `--output-dir` | `<input-dir>/videos` | 视频输出目录 |
| `--fps` | `10` | 视频帧率 |
| `--views` | 全部 | 逗号分隔，只转换指定 view 类型 |
| `--cams` | 全部 | 逗号分隔，合成时使用的相机 ID |
| `--layout` | 无 | grid 布局：行用 `;` 分隔，列用 `,` 分隔（view 类型） |
| `--labels` | 无 | 每格文字标签，格式与 `--layout` 相同 |
| `--composite-only` | false | 跳过单独视频生成，直接合成（需已有 `.mp4`） |
| `--cam-per-col` | false | 多相机合成模式：`--layout` 行=view类型，列=`--cams` 指定的相机，输出 `composite_allcams_<fps>fps.mp4` |
| `--list` | false | 打印发现的 view 类型和相机后退出 |

**`--layout` / `--labels` 格式示例（2 行 3 列）：**

```
--layout "left_1.0m,gt_views_gt,right_1.0m;left_2.0m,gt_views_pred,right_2.0m"
--labels "left 1m,GT,right 1m;left 2m,render,right 2m"
```

合成结果：

```
┌──────────┬──────┬───────────┐
│ left_1.0m│  GT  │ right_1.0m│  ← row 1
│  left 1m │  GT  │  right 1m │  ← labels
├──────────┼──────┼───────────┤
│ left_2.0m│render│ right_2.0m│  ← row 2
│  left 2m │render│  right 2m │  ← labels
└──────────┴──────┴───────────┘
```

输出文件：`composite_cam0_10fps.mp4`

**`--cam-per-col` 3视角合成（`--layout "gt_views_gt;gt_views_pred" --cams 0,1,2 --labels "GT;渲染"`）：**

```
┌──────────┬──────────┬──────────┐
│GT cam0   │GT cam1   │GT cam2   │  ← row 1: gt_views_gt
│    GT    │    GT    │    GT    │  ← labels（自动扩展至 3 列）
├──────────┼──────────┼──────────┤
│pred cam0 │pred cam1 │pred cam2 │  ← row 2: gt_views_pred
│   渲染   │   渲染   │   渲染   │  ← labels（自动扩展至 3 列）
└──────────┴──────────┴──────────┘
```

输出文件：`composite_allcams_10fps.mp4`

---

## eval_tools/save_gaussians_cvg.py

CVG 场景高斯保存脚本，逐帧输出标准 3DGS PLY 文件（SuperSplat / SIBR / gsplat 可读）。

**特性：**
- `--frames` / `--frame_skip` 控制保存哪些帧
- `--cam_ids` 控制保存哪些相机（默认仅 cam-0）
- `--max_dist` 过滤距相机超过指定距离的高斯（默认 80 m）
- `--save_merged` 额外输出全帧合并的 `merged_gaussians.ply`
- `--scene all` 批量处理所有场景

```bash
# 保存 场景指定帧，cam-0
python eval_tools/save_gaussians_cvg.py \
    --cfg_path configs/cvg/recondrive_cvg_3cam.yaml \
    --restore_ckpt work_dirs/cvg_cam3_training-v2/ckpt/best_module-v1.ckpt \
    --scene scene_004 \
    --frames 0,18,24 \
    --cam_ids 0,1,2 \
    --save_merged

# 每6帧取一帧，同时保存合并高斯
python eval_tools/save_gaussians_cvg.py \
    --cfg_path configs/cvg/recondrive_cvg_1cam.yaml \
    --restore_ckpt /remote-sync/yuchi.zuo/best_module.ckpt \
    --scene scene_000 \
    --frame_skip 6 \
    --save_merged
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cfg_path` | 必填 | YAML 配置文件路径 |
| `--restore_ckpt` | 必填 | 模型 checkpoint 路径 |
| `--scene` | 必填 | 场景名（`scene_000`）、纯数字（`0`）或 `all` |
| `--output_dir` | `<save_dir>/cvg_1cam_eval` | 输出根目录 |
| `--device` | config 中 devices[0] | 运行设备 |
| `--cam_ids` | `0` | 相机索引，逗号分隔；`all` = 全部相机 |
| `--frame_skip` | `1` | 每 N 帧取一帧（1 = 全部） |
| `--frames` | 无 | 指定帧列表，优先级高于 `--frame_skip` |
| `--max_dist` | `80.0` | 距离过滤阈值（米）；`0` = 不过滤 |
| `--save_merged` | false | 额外保存场景合并高斯 `merged_gaussians.ply` |

**输出目录结构：**

```
<output_dir>/
└── scene_000/
    ├── sample_0000/
    │   └── gaussians.ply
    ├── sample_0006/
    │   └── gaussians.ply
    ├── ...
    ├── merged_gaussians.ply   ← 仅 --save_merged 时生成
    └── summary.json
```

---

## eval_tools/save_pointcloud_cvg.py

CVG 场景彩色点云保存脚本，输出标准 RGB PLY（CloudCompare / MeshLab 可直接读取）。

颜色由模型输出的 SH 系数按 cam-0 视向计算，无 SH 支持时退化为 DC 分量近似色。

```bash
# 保存 场景 指定帧
python eval_tools/save_pointcloud_cvg.py \
    --cfg_path configs/cvg/recondrive_cvg_3cam.yaml \
    --restore_ckpt work_dirs/cvg_cam3_training-v2/ckpt/best_module-v1.ckpt \
    --scene scene_004 \
    --frames 0,6,12,18,24 \
    --cam_ids 0,1,2 \
    --save_merged

# 每6帧取一帧，同时保存合并点云
python eval_tools/save_pointcloud_cvg.py \
    --cfg_path configs/cvg/recondrive_cvg_1cam.yaml \
    --restore_ckpt checkpoints/recondrive_stage2.ckpt \
    --scene scene_000 \
    --frame_skip 6 \
    --save_merged
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cfg_path` | 必填 | YAML 配置文件路径 |
| `--restore_ckpt` | 必填 | 模型 checkpoint 路径 |
| `--scene` | 必填 | 场景名（`scene_000`）、纯数字（`0`）或 `all` |
| `--output_dir` | `<save_dir>/cvg_1cam_eval` | 输出根目录 |
| `--device` | config 中 devices[0] | 运行设备 |
| `--cam_ids` | `0` | 相机索引，逗号分隔；`all` = 全部相机 |
| `--frame_skip` | `1` | 每 N 帧取一帧（1 = 全部） |
| `--frames` | 无 | 指定帧列表，优先级高于 `--frame_skip` |
| `--max_dist` | `80.0` | 距离过滤阈值（米）；`0` = 不过滤 |
| `--save_merged` | false | 额外保存场景合并点云 `merged_pointcloud.ply` |

**输出目录结构：**

```
<output_dir>/
└── scene_000/
    ├── sample_0000/
    │   └── pointcloud.ply
    ├── sample_0006/
    │   └── pointcloud.ply
    ├── ...
    ├── merged_pointcloud.ply   ← 仅 --save_merged 时生成
    └── summary.json
```

---

## eval_tools/save_pointcloud_split.py

CVG 场景彩色点云分坐标系保存脚本。模型每次推理输入 frame i 和 frame i+6 共 2×N 张图像，本脚本将两帧在**各相机**下的点云拆开保存，并区分不同坐标系，便于可视化分析对齐情况。

每个 sample 每个相机输出三个 PLY 文件：

| 文件名 | 坐标系 | 说明 |
|--------|--------|------|
| `cam{id}_frame0.ply` | ego_i | frame i，相机 cam_id，在第 i 帧自车坐标系下 |
| `cam{id}_frameN_egoN.ply` | ego_{i+6} | frame i+6，相机 cam_id，在第 i+6 帧自车坐标系下 |
| `cam{id}_frameN_egoI.ply` | ego_i | frame i+6，相机 cam_id，变换回第 i 帧自车坐标系 |

```bash
# 保存 scene_004 指定帧，3 个相机，分坐标系输出
python eval_tools/save_pointcloud_split.py \
    --cfg_path configs/cvg/cvg_3cam_training.yaml \
    --restore_ckpt work_dirs/cvg_cam3_training-v1/ckpt/best_module.ckpt \
    --scene scene_004 \
    --frames 0 \
    --cam_ids 0,1,2 \
    --output_dir /home/jinheng.li/project/FeedforwardGS-RD/ReconDrive/work_dirs/3d_output/poindcloud_split

# 每 6 帧取一帧，只保存 cam-0
python eval_tools/save_pointcloud_split.py \
    --cfg_path configs/cvg/recondrive_cvg_1cam.yaml \
    --restore_ckpt checkpoints/recondrive_stage2.ckpt \
    --scene scene_000 \
    --frame_skip 6 \
    --cam_ids 0 \
    --output_dir /path/to/output
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cfg_path` | 必填 | YAML 配置文件路径 |
| `--restore_ckpt` | 必填 | 模型 checkpoint 路径 |
| `--scene` | 必填 | 场景名（`scene_000`）、纯数字（`0`）或 `all` |
| `--output_dir` | `<save_dir>/3d_output/pointcloud` | 输出根目录 |
| `--device` | config 中 devices[0] | 运行设备 |
| `--cam_ids` | `0` | 相机索引，逗号分隔；`all` = 全部相机 |
| `--frame_skip` | `1` | 每 N 帧取一帧（1 = 全部） |
| `--frames` | 无 | 指定帧列表，优先级高于 `--frame_skip` |
| `--max_dist` | `80.0` | 距离过滤阈值（米）；`0` = 不过滤 |

**输出目录结构：**

```
<output_dir>/
└── scene_004/
    ├── sample_0000/
    │   ├── cam0_frame0.ply          ← frame i，ego_i 坐标系
    │   ├── cam0_frameN_egoN.ply     ← frame i+6，ego_{i+6} 坐标系
    │   ├── cam0_frameN_egoI.ply     ← frame i+6，变换回 ego_i 坐标系
    │   ├── cam1_frame0.ply
    │   ├── cam1_frameN_egoN.ply
    │   ├── cam1_frameN_egoI.ply
    │   └── ...
    ├── sample_0006/
    │   └── ...
    └── summary.json
```

---

## eval_tools/save_pointcloud_cvg_gt.py

保存 CVG 数据集的 LiDAR **真值点云**（Ground Truth）。直接读取原始 LiDAR PCD 文件，无需模型推理。

**特性：**
- 不依赖模型推理，直接读取原始 LiDAR 数据
- 所有位姿在载入时归一化到第 0 帧自车坐标系，避免全局坐标大数值导致的 float32 精度损失
- 颜色按帧索引用 rainbow 色系区分
- `--reference_frame` 指定参考帧后，所有帧变换到该帧坐标系，文件名带 `_ref<N>` 后缀
- `--singleview` 只保留指定相机视野**并集**内的点，支持逗号分隔多个相机，输出文件名加 `_sv` 后缀（用于评测对齐）

```bash
# 保存 scene_000 指定帧（world 坐标系，即第 0 帧坐标系）
python eval_tools/save_pointcloud_cvg_gt.py \
    --data_root /home/jinheng.li/project/FeedforwardGS-RD/cvg_data_pipeline/data \
    --scene scene_004 \
    --frames 0,5,10,15 \
    --save_merged \
    --output_dir /home/jinheng.li/project/FeedforwardGS-RD/cvg_data_pipeline/data

# 以第 10 帧为参考帧保存
python eval_tools/save_pointcloud_cvg_gt.py \
    --scene scene_000 \
    --frames 0,5,10,15 \
    --reference_frame 10 \
    --save_merged

# 只保留前向单目视野内的点
python eval_tools/save_pointcloud_cvg_gt.py \
    --scene scene_000 \
    --frames 0,6,12,18,24 \
    --singleview \
    --save_merged

# 保留三视角视野并集内的点
python eval_tools/save_pointcloud_cvg_gt.py \
    --scene scene_004 \
    --frames 0,6,12,18,24 \
    --singleview --cam_name "camera_front_wide,camera_left_front,camera_right_front" \
    --save_merged \
    --data_root /home/jinheng.li/project/FeedforwardGS-RD/cvg_data_pipeline/data \
    --output_dir /home/jinheng.li/project/FeedforwardGS-RD/cvg_data_pipeline/data
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--scene` | 必填 | 场景名（`scene_000`）、纯数字（`0`）或 `all` |
| `--data_root` | `../cvg_data_pipeline/data` | CVG 数据根目录 |
| `--output_dir` | `../cvg_data_pipeline/data` | 输出根目录 |
| `--frame_skip` | `1` | 每 N 帧取一帧（1 = 全部） |
| `--frames` | 无 | 指定帧列表，优先级高于 `--frame_skip` |
| `--max_frames` | `0` | 最多处理帧数（均匀采样），0 = 不限制 |
| `--save_merged` | false | 额外保存合并点云 |
| `--reference_frame` | 无 | 参考帧索引，指定后所有帧变换到该帧坐标系 |
| `--singleview` | false | 只保留指定相机视野并集内的点，文件名加 `_sv` 后缀 |
| `--cam_name` | `camera_front_wide` | `--singleview` 使用的相机名，逗号分隔可指定多个 |
| `--sv_max_depth` | `80.0` | `--singleview` 深度截断（米） |

**输出目录结构：**

```
../cvg_data_pipeline/data/
└── scene_000/
    └── merge/
        ├── 0.ply                        ← world（第 0 帧）坐标系
        ├── 6.ply
        ├── ...
        ├── merged_pointcloud.ply        ← 仅 --save_merged 时生成
        ├── 0_ref10.ply                  ← --reference_frame 10 时
        ├── 6_ref10.ply
        ├── ...
        ├── merged_pointcloud_ref10.ply
        ├── 0_sv.ply                     ← --singleview 时
        ├── merged_pointcloud_sv.ply     ← --singleview + --save_merged 时
        └── merged_pointcloud_ref10_sv.ply
```

---

## eval_tools/eval_pointcloud_cvg.py

CVG 场景点云重建质量批量评测脚本，对比预测点云与真实点云。评测前需要先跑 `save_pointcloud_cvg_gt.py` 和 `save_pointcloud_cvg.py` 分别获取真值激光点云的合并点云和模型预测出的点云。

**指标：**
- **Accuracy**：预测点中落在真实点 threshold 范围内的比例（%）
- **Completeness**：真实点中被预测点覆盖到 threshold 范围内的比例（%）
- **F1 Score**：Accuracy 与 Completeness 的调和平均值
- **CD**：Chamfer Distance（两方向平均最近邻距离之和）

**目录结构约定：** pred 与 gt 目录镜像对应。

```bash
# auto 模式：只需指定场景，自动从默认路径读取 pred / gt 合并点云
python eval_tools/eval_pointcloud_cvg.py \
    --auto --scene 0

# auto 模式评测多个场景，自定义阈值
python eval_tools/eval_pointcloud_cvg.py \
    --auto --scene 0,1,2 --threshold 1

# 手动指定路径，评测 scene_000 逐帧点云
python eval_tools/eval_pointcloud_cvg.py \
    --pred_dir work_dirs/cvg_1cam_eval/pointcloud \
    --gt_dir /path/to/gt_pointclouds \
    --scene scene_000

# 手动指定路径，评测所有场景合并点云，阈值 1 m
python eval_tools/eval_pointcloud_cvg.py \
    --pred_dir /home/jinheng.li/project/FeedforwardGS-RD/ReconDrive/work_dirs/cvg_1cam_eval/pointcloud/scene_000/merged_pointcloud.ply \
    --gt_dir /home/jinheng.li/project/FeedforwardGS-RD/cvg_data_pipeline/data/scene_000/merge/merged_pointcloud_sv.ply \
    --scene all \
    --merged \
    --threshold 1 --max_points 0
    
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--auto` | false | 自动模式：只需 `--scene`，自动找 pred/gt 合并点云，隐式启用 `--merged` |
| `--pred_dir` | 无（`--auto` 时可省略） | 预测点云根目录（含 `scene_xxx/` 子目录） |
| `--gt_dir` | 无（`--auto` 时可省略） | 真实点云路径（目录或单个 `.ply` 文件） |
| `--scene` | `all` | 场景名（`scene_000`）、纯数字（`0`）、逗号分隔或 `all` |
| `--merged` | false | 评测 `merged_pointcloud.ply` 而非逐帧样本 |
| `--threshold` | `0.05` | 距离阈值（米），用于 Accuracy / Completeness / F1 |
| `--max_points` | `200000` | 每个点云最大点数（随机降采样），`0` = 不限 |
| `--seed` | `42` | 随机种子，保证降采样可复现 |
| `--output_json` | `<pred_dir>/eval_results.json` | 结果 JSON 保存路径 |
| `--no_save` | false | 只打印结果，不保存 JSON |

**auto 模式默认路径：**
- pred：`work_dirs/cvg_1cam_eval/pointcloud/<scene>/merged_pointcloud.ply`
- gt：`../cvg_data_pipeline/data/<scene>/merge/merged_pointcloud.ply`

**输出 JSON 结构：**

```json
{
  "config": { "pred_dir": "...", "gt": "...", "auto": true, "threshold": 0.05, ... },
  "scenes": {
    "scene_000": {
      "scene": "scene_000",
      "samples": {
        "merged": { "accuracy": 0.85, "completeness": 0.78, "f1": 0.81, "cd": 0.032, ... }
      },
      "aggregate": { "accuracy": 0.85, "completeness": 0.78, "f1": 0.81, ... }
    }
  },
  "overall": { "accuracy": 0.82, "completeness": 0.75, "f1": 0.78, "cd": 0.035, ... }
}
```

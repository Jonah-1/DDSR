# VGGT Stage 1 — 双路径静态高斯重建

## 设计目标

在 VGGT 24 层交叉注意力 aggregator 中，利用 SAM2 提供的动态掩码，将前 3 层改造为**双路径结构**，分别提取静态区域和动态区域的特征，融合后送入后 21 层完成标准 VGGT 流程，最终输出用于静态场景重建的高斯参数。

---

## 架构概览

```
输入图像
  ├── SAM2 → 动态掩码
  └── DINOv2 PatchEmbed → patch tokens
          │
     ┌────┴─────────────────────┐
     │      前 3 层（双路径）     │
     │                          │
     │  路径1（静态路径）         │  路径2（静态信息提取路径）
     │  冻结权重                 │  LoRA 微调
     │  动态区域 K = 0           │  标准 attention（无 K 置零）
     │                          │  ↓ 第3层输出后
     │                          │  动态 token → Deformable Attention
     │                          │
     └────────┬─────────────────┘
              │ 动态区域 token → MLP 融合
              │ 静态区域 token 取路径1结果
              ↓
     后 21 层（标准 VGGT 流程）
     第 12、18、24 层开启 LoRA 微调
     输入：I_dino + masks
              ↓
     Gaussian Head
              ↓
     Static GS Params → Static pointmaps
              ↓
     Adaptive merge（+ Diffusion model / VLM）
              ↓
     Static Gaussians + Loss
```

---

## 前 3 层双路径详解

### 路径 1：绝对静态路径（K=0 遮蔽）**已实现**

- 动态掩码对应位置的 token，在做交叉注意力时将其 **K 向量置 0**
- 效果：静态区域的 token 在 attention 时无法"看到"动态区域，静态特征提取不受动态物体干扰
- 权重**冻结**，不参与梯度更新
- 实现：`Aggregator.forward(dynamic_mask)` → `_compute_patch_key_mask()` 将像素掩码降采样至 patch 级；前 `num_mask_layers`（默认 3）层的 frame/global block 均传入此 mask；`Attention.forward()` 对动态 token 的 K 调用 `masked_fill(..., 0.0)` 置零

### 路径 2：动态 token 中的静态信息提取路径（标准 Attention + LoRA → DA）

动态掩码区域的 token 往往杂糅了静态背景信息，路径 2 的目标是**从这些动态 token 中把静态信息抠出来**，与路径 1 提取的绝对静态特征融合，共同构成完整的静态场景表示。

- 不做 K 置零，允许动态 token 与全局上下文交互
- 前 3 层走**标准 attention**，通过 **LoRA** 轻量微调，参数可训练
- 第 3 层输出后，对**动态掩码区域的 token** 单独做一次 **Deformable Attention（DA）**，捕捉动态物体形变与位移，从中提炼静态信息
- 处理后与路径 1 的动态区域 token 进行 MLP 融合

### 融合

- 前 3 层结束后，取两条路径中**动态掩码区域**的 token，通过一个 **MLP** 融合为最终表示
- 静态区域 token 直接使用路径 1 的输出
- 融合后的完整 token 序列送入后 21 层

---

## 后 21 层

采用标准 VGGT global/frame 交替注意力流程，输入为融合后的 token（携带 I_dino 特征与 mask 信息）。

**第 12、18、24 层**（0-indexed: 11、17、23）额外开启 **LoRA 微调**（`lora_r=8, lora_alpha=16`），在关键深层注入可训练参数，其余层权重冻结。

---

## 输出

- **Static pointmaps**：每像素静态区域 3D 点
- **Static GS Params**：高斯中心、旋转、尺度、不透明度、球谐系数
- 与 Diffusion model / VLM 生成结果通过 **Adaptive merge** 融合 → 最终 Static Gaussians
- **Grid-based Gaussian Grouping**：将静态高斯按网格分组，供 Stage 2 使用

---

## Loss 计算中的动态区域排除

`compute_gaussian_loss` 和 `compute_project_loss` 均已加入动态掩码排除逻辑：

- `render_splating_imgs`：对每个 `(frame_id, cam_id)` 从 `recontrast_data['dynamic_mask']`（shape `[B, S, H, W]`）中取对应切片（索引 = `frame_id × num_cams + cam_id`），resize 至渲染分辨率后存为 `('dynamic_mask', frame_id, cam_id)`
- `render_project_imgs`：同理，取 `ref_frame_id`（固定 0）对应的动态掩码，存为 `('dynamic_mask', ref_frame_id, src_frame_id, cam_id)`
- Loss 函数中：`mask = warped_mask × (1 − dynamic_mask)`，动态区域权重置零

若 SAM2 未检测到任何动态物体（`dynamic_mask is None`），行为与原逻辑完全一致。

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `models/vggt_stage1/models/aggregator.py` | 双路径改动的核心实现位置 |
| `models/vggt_stage1/models/vggt.py` | Stage 1 模型入口 |
| `models/recondrive_model.py` | 完整前向流程（含 Stage 2） |
| `docs/summery.txt` | 设计思路原始文字描述 |
| `.claude/skills/model_design/SKILL.md` | 模型设计 skill，含 Stage 2 方案 |

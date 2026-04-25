# VGGT Stage 1 — 双路径静态高斯重建

## 设计目标

在 VGGT 24 层交叉注意力 aggregator 中，利用 SAM2 提供的动态掩码，将前 6 层改造为**双路径结构**，分别提取静态区域和动态区域的特征，融合后送入后 18 层完成标准 VGGT 流程，最终输出用于静态场景重建的高斯参数。

---

## 架构概览

```
输入图像
  ├── SAM2 → 动态掩码
  └── DINOv2 PatchEmbed → patch tokens
          │
     ┌────┴─────────────────────┐
     │      前 6 层（双路径）     │
     │                          │
     │  路径1（静态路径）         │  路径2（动态路径）
     │  冻结权重                 │  LoRA 微调
     │  动态区域 K = 0           │  动态区域用 Deformable Attention
     │                          │
     └────────┬─────────────────┘
              │ 动态区域 token → MLP 融合
              │ 静态区域 token 取路径1结果
              ↓
     后 18 层（标准 VGGT 流程）
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

## 前 6 层双路径详解

### 路径 1：静态路径（K=0 遮蔽）

- 动态掩码对应位置的 token，在做交叉注意力时将其 **K 向量置 0**
- 效果：静态区域的 token 在 attention 时无法"看到"动态区域，静态特征提取不受动态物体干扰
- 权重**冻结**，不参与梯度更新

### 路径 2：动态路径（Deformable Attention）

- 不做 K 置零
- 动态掩码区域的 token 在交叉注意力时改用 **Deformable Attention（DA）**，适配动态物体的形变和位移
- 静态区域 token 仍走标准 attention
- 通过 **LoRA** 进行轻量微调，该路径参数可训练

### 融合

- 前 6 层结束后，取两条路径中**动态掩码区域**的 token，通过一个 **MLP** 融合为最终表示
- 静态区域 token 直接使用路径 1 的输出
- 融合后的完整 token 序列送入后 18 层

---

## 后 18 层

采用标准 VGGT global/frame 交替注意力流程，输入为融合后的 token（携带 I_dino 特征与 mask 信息）。

---

## 输出

- **Static pointmaps**：每像素静态区域 3D 点
- **Static GS Params**：高斯中心、旋转、尺度、不透明度、球谐系数
- 与 Diffusion model / VLM 生成结果通过 **Adaptive merge** 融合 → 最终 Static Gaussians
- **Grid-based Gaussian Grouping**：将静态高斯按网格分组，供 Stage 2 使用

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `models/vggt_stage1/models/aggregator.py` | 双路径改动的核心实现位置 |
| `models/vggt_stage1/models/vggt.py` | Stage 1 模型入口 |
| `models/recondrive_model.py` | 完整前向流程（含 Stage 2） |
| `docs/summery.txt` | 设计思路原始文字描述 |
| `.claude/skills/model_design/SKILL.md` | 模型设计 skill，含 Stage 2 方案 |

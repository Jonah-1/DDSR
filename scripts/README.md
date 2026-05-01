# NuScenes 训练（带 LoRA 初始化）
# 加载顺序：
#   1. vggt.pt          — 由 model_cfg.vggt_checkpoint 自动加载（基础权重）
#   2. --lora_ckpt       — 从 recondrive_stage2 转换的 LoRA 权重，叠加到 path2 和第 12/18/24 层
# 可训练参数：depth_head、gs_head、frame/global_blocks[11,17,23] LoRA、
#             path2_frame/global_blocks LoRA、path2_da、dynamic_fusion_mlp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m scripts.trainer \
    --cfg_path configs/nuscenes/recondrive.yaml \
    --lora_ckpt /home/jinheng.li/project/FeedforwardGS-RD/ReconDrive/checkpoints/lora_converted_4to24.pt \
    --devices 8 \
    --ckpt_dir /train-syncdata/jinheng.li/project/DDSR/nucense/v1_5.1

# lora_converted.pt 生成方式（只需跑一次）：
# python scripts/convert_lora_ckpt.py
pxvd19er5zqk2bvzagadbj9mrqkdyawr4lnm8jw6
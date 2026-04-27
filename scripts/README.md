#!/bin/bash
conda activate recondrive
export PYTHONNOUSERSITE=1

CONFIG_PATH='./configs/nuscenes/recondrive.yaml'
PRETRAINED_CHECKPOINT_PATH='./checkpoints/recondrive_stage2.ckpt'

python -m scripts.trainer \
    --cfg_path=${CONFIG_PATH} \
    --train_4d \
    --devices="${1:-1}"" \
    --pretrained_ckpt=${PRETRAINED_CHECKPOINT_PATH}

python -m scripts.trainer  --cfg_path configs/cvg/cvg_1cam_training.yaml   --devices 5 --pretrained_ckpt=checkpoints/recondrive_stage2.ckpt

CONFIG_PATH='./configs/cvg/recondrive_cvg_1cam.yaml'
CHECKPOINT_PATH='./checkpoints/recondrive_stage2.ckpt' 
OUTPUT_DIR='work_dirs/cvg_1cam_infer'

CHECKPOINT_PATH='/home/yuchi.zuo/ReconDrive/work_dirs/cvg_training/ckpt/best_module.ckpt' 
OUTPUT_DIR='work_dirs/cvg_1cam_infer_sft'
DEVICE='0'

CONFIG_PATH='./configs/nuscenes/recondrive_single_view.yaml'
CHECKPOINT_PATH='./checkpoints/recondrive_stage2.ckpt' 
OUTPUT_DIR='work_dirs/nuscenes_1cam_infer'

python -m scripts.inference \
    --cfg_path="$CONFIG_PATH" \
    --restore_ckpt="$CHECKPOINT_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --device="$DEVICE" \
    --novel_distances="$NOVEL_VIEW_DISTANCES" \
    --eval_resolution="$EVAL_RESOLUTION" \
    $([ "$SAVE_NOVEL_RENDERS" = false ] && echo "--no_renders")

CHECKPOINT_PATH='./checkpoints/recondrive_stage2.ckpt' 
OUTPUT_DIR='work_dirs/cvg_1cam_infer'
python scripts/inference_single_view_direct.py \
    --cfg_path ./configs/cvg/recondrive_cvg_1cam.yaml \
    --ckpt_path /home/yuchi.zuo/ReconDrive/work_dirs/cvg_training/ckpt/best_module.ckpt \
    --output_dir work_dirs/cvg_1cam_infer_sft


python -m scripts.inference_cvg /home/yuchi.zuo/FeedforwardGS-RD/cvg_data_pipeline/data/scene_078 \
    --checkpoint /home/yuchi.zuo/ReconDrive/work_dirs/cvg_training/ckpt/best_module.ckpt \
    --output-dir ./cvg_results


python /home/yuchi.zuo/ReconDrive/tests/render_to_video.py \
    --input-dir /home/yuchi.zuo/ReconDrive/work_dirs/cvg_1cam_infer/scene_078 \
    --output-dir /home/yuchi.zuo/ReconDrive/work_dirs/cvg_1cam_infer/videos \
    --fps 20 \
    --layout "left_2.0m,gt_views_gt,right_2.0m" \
             "left_3.0m,gt_views_pred,right_3.0m" \
    --grid-fps 20

#训练代码
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m scripts.trainer  --cfg_path configs/nuscenes/recondrive.yaml   --devices 6
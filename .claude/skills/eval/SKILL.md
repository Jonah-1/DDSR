# Eval Skill

Run evaluation on a trained ReconDrive checkpoint and report results.

## Steps

1. Check that `./checkpoints/recondrive_stage2.ckpt` exists (or a user-specified checkpoint).
2. Run inference using `scripts/inference.sh` (or construct the command with any overrides the user specified).
3. After inference completes, read `./work_dirs/recondrive_stage2_eval_output/metrics.json` and display the results in a formatted table (PSNR, SSIM, LPIPS per scene and overall average).
4. If novel-view renders were saved, list the output directory contents so the user can locate the images.

## Usage

```
/eval
/eval --ckpt ./checkpoints/my_checkpoint.ckpt
/eval --ckpt ./checkpoints/my_checkpoint.ckpt --output_dir ./work_dirs/my_eval
/eval --novel_distances 1.0,2.0,3.0 --save_renders
```

## Arguments (all optional)

| Argument | Default | Description |
|----------|---------|-------------|
| `--ckpt` | `./checkpoints/recondrive_stage2.ckpt` | Path to checkpoint to evaluate |
| `--output_dir` | `./work_dirs/recondrive_stage2_eval_output` | Directory for outputs |
| `--device` | `0` | GPU device id |
| `--novel_distances` | `1.0,2.0,3.0` | Lateral offset distances (meters) for novel-view synthesis |
| `--save_renders` | false | Save novel-view rendered images |
| `--resolution` | `280x518` | Evaluation resolution (HxW) |

## Implementation Notes

- Inference only supports single-GPU mode.
- Config is always `./configs/nuscenes/recondrive.yaml`.
- Default test scenes: scene-0014, 0018, 0906, 0098, 0100, 0103, 0270, 0271, 0278, 0553, 0558, 0802, 0968, 1065.
- Pass `--no_renders` flag when `--save_renders` is not set to skip writing images.

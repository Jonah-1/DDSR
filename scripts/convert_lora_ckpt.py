"""
Convert LoRA weights from recondrive_stage2.ckpt to match current model naming.

Source naming  (recondrive_stage2):
  model.aggregator.frame_blocks.{i}.attn.qkv.lora_down.weight   [r, dim]
  model.aggregator.frame_blocks.{i}.attn.qkv.lora_up.weight     [3*dim, r]
  model.aggregator.frame_blocks.{i}.attn.proj.lora_down.weight  [r, dim]
  model.aggregator.frame_blocks.{i}.attn.proj.lora_up.weight    [dim, r]
  (same for global_blocks)

Current model naming:
  model.aggregator.frame_blocks.{i}.attn.lora_qkv_A.weight      [r, dim]
  model.aggregator.frame_blocks.{i}.attn.lora_qkv_B.weight      [3*dim, r]
  model.aggregator.frame_blocks.{i}.attn.lora_proj_A.weight     [r, dim]
  model.aggregator.frame_blocks.{i}.attn.lora_proj_B.weight     [dim, r]
  (same for global_blocks)

Converted layers:
  - frame_blocks[0-23]  <- source frame_blocks[0-23]  (all 24 layers)
  - global_blocks[0-23] <- source global_blocks[0-23] (all 24 layers)

NOT converted (initialized from scratch during training):
  - path2_frame_blocks, path2_global_blocks

Output: checkpoints/lora_converted.pt
"""

import torch

SRC_CKPT = "/home/jinheng.li/project/FeedforwardGS-RD/ReconDrive/checkpoints/recondrive_stage2.ckpt"
OUT_CKPT = "/home/jinheng.li/project/DDSR/checkpoints/lora_converted.pt"

SUFFIX_MAP = {
    "attn.qkv.lora_down.weight":  "attn.lora_qkv_A.weight",
    "attn.qkv.lora_up.weight":    "attn.lora_qkv_B.weight",
    "attn.proj.lora_down.weight": "attn.lora_proj_A.weight",
    "attn.proj.lora_up.weight":   "attn.lora_proj_B.weight",
}


def convert(src_sd: dict) -> dict:
    out = {}

    def copy_block(block_name: str, idx: int):
        prefix_src = f"model.aggregator.{block_name}.{idx}."
        prefix_dst = f"model.aggregator.{block_name}.{idx}."
        copied = 0
        for src_suf, dst_suf in SUFFIX_MAP.items():
            src_key = prefix_src + src_suf
            dst_key = prefix_dst + dst_suf
            if src_key in src_sd:
                out[dst_key] = src_sd[src_key].clone()
                copied += 1
            else:
                print(f"  WARN  missing: {src_key}")
        return copied

    total = 0
    for block_name in ("frame_blocks", "global_blocks"):
        for i in range(24):
            n = copy_block(block_name, i)
            total += n

    print(f"Total converted keys: {total}  (expected {2 * 24 * 4} = {2*24*4})")
    return out


if __name__ == "__main__":
    print(f"Loading: {SRC_CKPT}")
    ckpt = torch.load(SRC_CKPT, map_location="cpu", weights_only=False)
    converted = convert(ckpt["state_dict"])
    torch.save({"state_dict": converted}, OUT_CKPT)
    print(f"Saved to: {OUT_CKPT}")

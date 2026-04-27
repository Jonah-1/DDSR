# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt_stage1.layers import PatchEmbed
from vggt_stage1.layers.block import Block
from vggt_stage1.layers.deformable_attention import DeformableTokenAttention
from vggt_stage1.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt_stage1.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        num_mask_layers=6,
        lora_r=8,
        lora_alpha=16.0,
        main_lora_r=0,
        main_lora_alpha=1.0,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    lora_r=main_lora_r,
                    lora_alpha=main_lora_alpha,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    lora_r=main_lora_r,
                    lora_alpha=main_lora_alpha,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        self.num_mask_layers = num_mask_layers
        self.lora_r = lora_r

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

        # ---------- Path 2: standard blocks with LoRA (first num_mask_layers layers) + one DA pass after ----------
        self.path2_frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                )
                for _ in range(num_mask_layers)
            ]
        )
        self.path2_global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                )
                for _ in range(num_mask_layers)
            ]
        )
        # Single DA applied to dynamic tokens after the num_mask_layers layers of Path 2
        self.path2_da = DeformableTokenAttention(
            dim=embed_dim,
            num_heads=num_heads,
        )
        # MLP that fuses path1 and path2 dynamic tokens after num_mask_layers layers
        self.dynamic_fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor, dynamic_mask: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): [B, S, 3, H, W], range [0, 1].
            dynamic_mask (torch.Tensor, optional): [B, S, H, W] binary mask, 1 = dynamic pixel.
                Enables dual-path processing for the first num_mask_layers layers:
                  Path 1 (frozen): K-zeroing for dynamic tokens in frame/global attention.
                  Path 2 (trainable): DeformableAttention for dynamic tokens, no K zeroing.
                After num_mask_layers layers the two dynamic-token streams are fused via
                dynamic_fusion_mlp, then processing continues with the standard VGGT flow.

        Returns:
            (list[torch.Tensor], int): aggregated token list and patch_start_idx.
        """
        B, S, C_in, H, W = images.shape
        H_p = H // self.patch_size
        W_p = W // self.patch_size

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P_patch, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H_p, W_p, device=images.device)

        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        # Compute patch-level key mask (Path 1: K zeroing for dynamic tokens)
        patch_key_mask = None
        if dynamic_mask is not None:
            patch_key_mask = self._compute_patch_key_mask(dynamic_mask, B, S, H, W, images.device)

        # Path 2 starts from the same token state as Path 1
        tokens_p2 = tokens.clone() if (patch_key_mask is not None and self.num_mask_layers > 0) else None

        frame_idx = 0
        global_idx = 0
        p2_frame_idx = 0
        p2_global_idx = 0
        output_list = []

        for iter_idx in range(self.aa_block_num):
            use_dual = (tokens_p2 is not None and iter_idx < self.num_mask_layers)

            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, patch_key_mask=patch_key_mask
                    )
                    if use_dual:
                        tokens_p2, p2_frame_idx, _ = self._process_path2_frame_attention(
                            tokens_p2, B, S, P, C, p2_frame_idx, pos=pos,
                        )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, patch_key_mask=patch_key_mask
                    )
                    if use_dual:
                        tokens_p2, p2_global_idx, _ = self._process_path2_global_attention(
                            tokens_p2, B, S, P, C, p2_global_idx, pos=pos,
                        )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            # After the last dual-path iteration, apply DA to dynamic tokens in path2, then fuse
            if use_dual and iter_idx == self.num_mask_layers - 1:
                # Ensure tokens_p2 is in [B*S, P, C] for DA
                if tokens_p2.shape != (B * S, P, C):
                    tokens_p2 = tokens_p2.view(B, S, P, C).view(B * S, P, C)
                tokens_p2 = self.path2_da(tokens_p2, patch_key_mask, self.patch_start_idx, H_p, W_p)
                tokens = self._fuse_dynamic_tokens(tokens, tokens_p2, patch_key_mask, B, S, P, C)
                tokens_p2 = None  # free memory; path2 is done

            for i in range(len(frame_intermediates)):
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    def _compute_patch_key_mask(
        self, dynamic_mask: torch.Tensor, B: int, S: int, H: int, W: int, device: torch.device
    ) -> torch.Tensor:
        """
        Downsample pixel-level dynamic_mask to patch-level and prepend False for special tokens.

        Args:
            dynamic_mask: [B, S, H, W] binary float/bool, 1 = dynamic pixel
        Returns:
            patch_key_mask: [B*S, patch_start_idx + Ph*Pw] bool, True = dynamic token
        """
        mask_flat = dynamic_mask.float().view(B * S, 1, H, W)
        # Max pool: any dynamic pixel in a patch makes the whole patch dynamic
        mask_patch = F.max_pool2d(mask_flat, kernel_size=self.patch_size, stride=self.patch_size)
        Ph, Pw = H // self.patch_size, W // self.patch_size
        mask_patch = mask_patch.squeeze(1).view(B * S, Ph * Pw).bool()
        # Camera and register tokens are never masked
        special_false = torch.zeros(B * S, self.patch_start_idx, dtype=torch.bool, device=device)
        return torch.cat([special_false, mask_patch], dim=1)  # [B*S, P]

    def _process_path2_frame_attention(
        self, tokens, B, S, P, C, p2_frame_idx, pos=None, **kwargs
    ):
        """Path-2 frame attention: standard Block with LoRA, no K zeroing."""
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []
        for _ in range(self.aa_block_size):
            block = self.path2_frame_blocks[p2_frame_idx]
            if self.training:
                tokens = checkpoint(
                    block, tokens, pos, None,
                    use_reentrant=self.use_reentrant,
                )
            else:
                tokens = block(tokens, pos=pos, key_mask=None)
            p2_frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
        return tokens, p2_frame_idx, intermediates

    def _process_path2_global_attention(self, tokens, B, S, P, C, p2_global_idx, pos=None):
        """Path-2 global attention: standard Block, no K zeroing (key_mask=None)."""
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)
        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(
                    self.path2_global_blocks[p2_global_idx], tokens, pos, None,
                    use_reentrant=self.use_reentrant,
                )
            else:
                tokens = self.path2_global_blocks[p2_global_idx](tokens, pos=pos, key_mask=None)
            p2_global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
        return tokens, p2_global_idx, intermediates

    def _fuse_dynamic_tokens(self, tokens_p1, tokens_p2, patch_key_mask, B, S, P, C):
        """
        Merge path-1 and path-2 dynamic tokens via dynamic_fusion_mlp.
        Static positions keep path-1 values unchanged.
        """
        # Normalise both to [B*S, P, C] (last block may have left them in [B, S*P, C])
        if tokens_p1.shape != (B * S, P, C):
            tokens_p1 = tokens_p1.view(B, S, P, C).view(B * S, P, C)
        if tokens_p2.shape != (B * S, P, C):
            tokens_p2 = tokens_p2.view(B, S, P, C).view(B * S, P, C)

        merged = tokens_p1.clone()
        if patch_key_mask.any():
            dyn_p1 = tokens_p1[patch_key_mask]  # [N_dyn, C]
            dyn_p2 = tokens_p2[patch_key_mask]  # [N_dyn, C]
            merged[patch_key_mask] = self.dynamic_fusion_mlp(torch.cat([dyn_p1, dyn_p2], dim=-1)).to(merged.dtype)
        return merged

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None, patch_key_mask=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            key_mask = patch_key_mask if (frame_idx < self.num_mask_layers and patch_key_mask is not None) else None
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, key_mask, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos, key_mask=key_mask)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, patch_key_mask=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        # Reshape patch_key_mask from [B*S, P] to [B, S*P] for global attention
        global_key_mask = patch_key_mask.view(B, S * P) if patch_key_mask is not None else None

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            key_mask = global_key_mask if (global_idx < self.num_mask_layers and global_key_mask is not None) else None
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, key_mask, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos, key_mask=key_mask)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableTokenAttention(nn.Module):
    """
    Single-scale deformable attention for dynamic patch tokens on a 2D grid.

    Each dynamic query token predicts `num_points` 2D sampling offsets from Q.
    K/V features are bilinearly sampled at those positions from the patch
    feature map. Attention weights are computed via scaled Q·K dot product
    over the sampled points (not all tokens).

    Only positions where dynamic_mask=True are written to the output;
    all other positions are left unchanged (caller is responsible for merging).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_points: int = 4,
        qkv_bias: bool = True,
        proj_bias: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_points = num_points
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # Predicts 2D sampling offsets per head per point from Q
        self.offset_attn = nn.Linear(dim, num_heads * num_points * 2)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        # Init to zero so training starts from uniform local attention
        nn.init.zeros_(self.offset_attn.weight)
        nn.init.zeros_(self.offset_attn.bias)

    def forward(
        self,
        x: torch.Tensor,             # [BS, P, C]
        dynamic_mask: torch.Tensor,  # [BS, P] bool, True = dynamic patch token
        patch_start_idx: int,
        H_p: int,
        W_p: int,
    ) -> torch.Tensor:
        """Returns [BS, P, C] with dynamic positions updated."""
        BS, P, C = x.shape
        dtype = x.dtype

        # Build K/V feature maps from patch tokens (float32 for grid_sample)
        patch_tokens = x[:, patch_start_idx:]          # [BS, H_p*W_p, C]
        kv = self.kv(patch_tokens)                      # [BS, H_p*W_p, 2C]
        k_map = kv[..., :C].view(BS, H_p, W_p, C).permute(0, 3, 1, 2).float()  # [BS, C, H_p, W_p]
        v_map = kv[..., C:].view(BS, H_p, W_p, C).permute(0, 3, 1, 2).float()

        out = x.clone()

        for bs in range(BS):
            dyn_idx = dynamic_mask[bs, patch_start_idx:].nonzero(as_tuple=True)[0]
            if dyn_idx.numel() == 0:
                continue

            N_dyn = dyn_idx.numel()
            x_dyn = x[bs, patch_start_idx + dyn_idx]   # [N_dyn, C]

            # Project to query space, then predict sampling offsets from Q
            q_dyn = self.q(x_dyn)                       # [N_dyn, C]
            offsets = self.offset_attn(q_dyn).float().view(N_dyn, self.num_heads, self.num_points, 2).tanh() * 0.5
            q_dyn = q_dyn.float()

            # Reference positions normalised to [-1, 1] for grid_sample (x=w, y=h)
            ref_h = (dyn_idx // W_p).float() / max(H_p - 1, 1) * 2 - 1
            ref_w = (dyn_idx % W_p).float() / max(W_p - 1, 1) * 2 - 1
            ref = torch.stack([ref_w, ref_h], dim=-1)   # [N_dyn, 2]

            head_outputs = []
            for h_idx in range(self.num_heads):
                # Sample coordinates for this head: reference + predicted offset
                coords = (ref.unsqueeze(1) + offsets[:, h_idx]).clamp(-1, 1)  # [N_dyn, num_pts, 2]

                s, e = h_idx * self.head_dim, (h_idx + 1) * self.head_dim
                k_h = k_map[bs, s:e].unsqueeze(0).expand(N_dyn, -1, -1, -1)  # [N_dyn, hd, H_p, W_p]
                v_h = v_map[bs, s:e].unsqueeze(0).expand(N_dyn, -1, -1, -1)

                grid = coords.unsqueeze(1)  # [N_dyn, 1, num_pts, 2]
                k_s = F.grid_sample(k_h, grid,
                                    mode='bilinear', padding_mode='border', align_corners=True)
                v_s = F.grid_sample(v_h, grid,
                                    mode='bilinear', padding_mode='border', align_corners=True)
                # → [N_dyn, head_dim, num_pts]
                k_s = k_s.squeeze(2)
                v_s = v_s.squeeze(2)

                # Scaled Q·K dot product over sampled points → attention weights
                q_head = q_dyn[:, s:e]                                         # [N_dyn, head_dim]
                attn_w = (q_head.unsqueeze(2) * k_s).sum(dim=1) * self.scale  # [N_dyn, num_pts]
                attn_w = attn_w.softmax(dim=-1)

                head_out = (v_s * attn_w.unsqueeze(1)).sum(dim=-1)             # [N_dyn, head_dim]
                head_outputs.append(head_out)

            agg = torch.cat(head_outputs, dim=-1)  # [N_dyn, C]
            agg = self.proj(agg)
            out[bs, patch_start_idx + dyn_idx] = agg.to(out.dtype)

        return out

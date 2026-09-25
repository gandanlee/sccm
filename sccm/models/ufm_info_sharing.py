"""
exp31_ufm_info_sharing — UFM/RoMa-v2 style Cross-View Info-sharing Transformer.

Replaces the GP module at scale 16 with bidirectional cross-view attention,
then computes match embeddings via cosine similarity.

GP:    K_xy · K_yy⁻¹ · fourier_PE(y_coords)       → (B, gp_dim, H, W)
Ours:  cross_attn(f_A, f_B) → cosine_sim → softmax → weighted fourier_PE
                                                     → (B, gp_dim, H, W)

Based on:
  - RoMa v2 matcher.py: MatchTransformer + match embedding
  - UFM ufm.py: info_sharing cross-view transformer

Interface: matches GP.forward(x, y) → (B, gp_dim, H, W)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossViewBlock(nn.Module):
    """
    Single cross-view attention block.

    Concatenates A and B tokens, runs self-attention (allowing cross-view interaction),
    then splits back. This is the RoMa v2 "even block" pattern.
    """

    def __init__(self, dim: int, n_heads: int, ffn_ratio: float = 4.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        ffn_dim = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x_A: torch.Tensor, x_B: torch.Tensor):
        """
        Args:
            x_A: (B, N, D) — source tokens
            x_B: (B, N, D) — target tokens
        Returns:
            x_A, x_B: updated tokens (same shapes)
        """
        N = x_A.shape[1]
        x = torch.cat([x_A, x_B], dim=1)  # (B, 2N, D)

        # Self-attention (cross-view)
        h = self.norm1(x)
        B, L, D = h.shape
        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        x = x + self.proj(attn_out)

        # FFN
        x = x + self.ffn(self.norm2(x))

        return x[:, :N], x[:, N:]


class SelfViewBlock(nn.Module):
    """
    Single self-view attention block.

    Each view attends only within itself (no cross-view interaction).
    This is the RoMa v2 "odd block" pattern.
    """

    def __init__(self, dim: int, n_heads: int, ffn_ratio: float = 4.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        ffn_dim = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, N, D)
        Returns:
            x: (B, N, D)
        """
        h = self.norm1(x)
        B, N, D = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class UFMInfoSharing(nn.Module):
    """
    Cross-view info-sharing transformer — drop-in replacement for GP.

    Architecture (following RoMa v2 MatchTransformer):
      1. Project encoder features to internal dim
      2. N alternating blocks:
         - CrossViewBlock: A+B tokens attend freely (feature enrichment)
         - SelfViewBlock: each view attends within itself (spatial reasoning)
      3. Match embedding: cosine_sim(enriched_A, enriched_B) → softmax → weighted pos_emb
      4. Output: (B, gp_dim, H, W)

    Interface: forward(x, y) → (B, gp_dim, H, W) — same as GP.
    """

    def __init__(
        self,
        in_dim: int = 512,
        dim: int = 512,
        n_blocks: int = 6,
        n_heads: int = 8,
        ffn_ratio: float = 4.0,
        temp: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.temp = temp

        # Input/output projections
        self.input_proj = nn.Linear(in_dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        # Alternating blocks (cross-view, self-view, cross-view, self-view, ...)
        self.cross_blocks = nn.ModuleList()
        self.self_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.cross_blocks.append(CrossViewBlock(dim, n_heads, ffn_ratio))
            self.self_blocks.append(SelfViewBlock(dim, n_heads, ffn_ratio))

        # Fourier PE — same basis as GP: cos(8π · Conv2d(2, dim))
        self.pos_conv = nn.Conv2d(2, dim, 1, 1)

    def _get_pos_emb(self, B: int, H: int, W: int, device: torch.device):
        """GP-compatible Fourier positional encoding: cos(8π · Conv2d(coords))."""
        coords = torch.meshgrid(
            torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=device),
            torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=device),
            indexing="ij",
        )
        coords = torch.stack((coords[1], coords[0]), dim=-1)[None].expand(B, H, W, 2)
        coords = rearrange(coords, "b h w d -> b d h w")
        pe = torch.cos(8 * math.pi * self.pos_conv(coords))  # (B, dim, H, W)
        return rearrange(pe, "b d h w -> b (h w) d")

    def forward(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        """
        Drop-in replacement for GP.forward(x, y).

        Args:
            x: source features (B, C, H, W) — f1 at scale 16
            y: target features (B, C, H, W) — f2 at scale 16

        Returns:
            output: (B, dim, H, W) — same shape as GP output
        """
        B, C, H, W = x.shape
        N = H * W

        # 1. Flatten and project
        x_tok = self.input_proj(rearrange(x.float(), "b c h w -> b (h w) c"))
        y_tok = self.input_proj(rearrange(y.float(), "b c h w -> b (h w) c"))

        # 2. Alternating attention blocks
        for cross_blk, self_blk in zip(self.cross_blocks, self.self_blocks):
            x_tok, y_tok = cross_blk(x_tok, y_tok)
            x_tok = self_blk(x_tok)
            y_tok = self_blk(y_tok)

        # 3. Normalize
        x_tok = self.norm(x_tok)
        y_tok = self.norm(y_tok)

        # 4. Match embedding (RoMa v2 style)
        #    cosine_sim(enriched_A, enriched_B) → softmax → weighted pos_emb
        x_n = F.normalize(x_tok, dim=-1)
        y_n = F.normalize(y_tok, dim=-1)
        attn = torch.softmax(
            torch.bmm(x_n, y_n.transpose(1, 2)) / self.temp,  # (B, N, N)
            dim=-1,
        )

        pos_emb = self._get_pos_emb(B, H, W, x.device)  # (B, N, dim)
        match_emb = torch.bmm(attn, pos_emb)  # (B, N, dim)

        # 5. Output: match_emb only (GP-compatible, bounded [-1,1])
        # CRITICAL: do NOT add x_tok — GP outputs Fourier PE weighted avg only
        return rearrange(match_emb, "b (h w) d -> b d h w", h=H, w=W)

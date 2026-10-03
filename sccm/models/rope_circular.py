"""
2D Rotary Position Embedding (RoPE) with longitude-axis periodicity for ERP.

Key idea:
    RoPE applies a position-dependent rotation to Q, K before attention. Dot
    product q·k then depends only on *relative* position.
    For ERP, longitude is 2π-periodic. Choosing INTEGER frequencies for the
    longitude rotation makes the rotation angle θ = k·lon automatically
    periodic: exp(i·k·(lon+2π)) = exp(i·k·lon). This means attention between
    tokens on either side of the seam (lon=±π) is treated identically to
    tokens that are actually adjacent.

    Latitude is bounded ∈ [-π/2, +π/2] (not periodic). Continuous frequencies.

Layout (head_dim=64 typical, n_heads=8 → dim=512):
    first  32 dims: latitude RoPE, 16 pairs, freqs = 10000^{-2i/32}
    second 32 dims: longitude RoPE, 16 pairs, freqs = {1, 2, ..., 16} (integers)

Same pattern used for both Q and K.

Interface contract:
    Given Q of shape (B, H_heads, L, head_dim) and the token grid (H, W),
    `apply_rope_2d` returns Q_rotated of same shape, with tokens indexed in
    row-major order (row h, col w ↔ index h*W + w).

Reference: RoFormer (Su et al., 2021) for the base RoPE formulation; we
extend the frequency choice to enforce seam-wrap equivariance.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE2DCircular(nn.Module):
    """Cached RoPE tables for ERP 2D grids with circular longitude.

    At construction, builds cos/sin tables for a specific (H, W) feature grid.
    Tables are registered as buffers so they move with `.to(device)`.

    Args:
        head_dim: per-head dimension (must be divisible by 4: half for lat,
                  half for lon, each further split into pairs)
        H, W: feature-map resolution (32×64 at stride-14 medium)
        lon_int_freq_max: highest integer longitude frequency. Pairs = head_dim/4.
                          If None, defaults to head_dim/4.
        lat_theta_base: base for latitude geometric frequencies (standard RoPE)
    """

    def __init__(
        self,
        head_dim: int,
        H: int,
        W: int,
        lon_int_freq_max: int | None = None,
        lat_theta_base: float = 10000.0,
        equal_area_latitudes: bool = False,
        lon_geometric: bool = False,
    ) -> None:
        super().__init__()
        assert head_dim % 4 == 0, f"head_dim must be divisible by 4, got {head_dim}"
        self.head_dim = head_dim
        self.H = H
        self.W = W
        half = head_dim // 2            # bytes per axis
        n_pairs_per_axis = half // 2    # per-axis rotation pair count
        self.n_pairs = n_pairs_per_axis

        # Latitude frequencies (geometric, standard RoPE)
        # θ_i = base^{-2i/half}, i=0..n_pairs-1
        lat_freqs = lat_theta_base ** (
            -torch.arange(0, n_pairs_per_axis, dtype=torch.float32) * 2.0 / half
        )                                                        # (n_pairs,)

        # Longitude frequencies.
        if lon_geometric:
            # GEOMETRIC (standard RoPE) — ABLATION ONLY: irrational/geometric series,
            # NOT 2π-periodic → relative phase is seam-discontinuous and only locally
            # yaw-equivariant. Isolates the value of the integer-harmonic choice.
            lon_freqs = lat_theta_base ** (
                -torch.arange(0, n_pairs_per_axis, dtype=torch.float32) * 2.0 / half
            )
        else:
            # INTEGERS (default) — natural 2π-periodicity (circular-harmonic RoPE)
            max_int = n_pairs_per_axis if lon_int_freq_max is None else int(lon_int_freq_max)
            lon_freqs = torch.arange(1, max_int + 1, dtype=torch.float32)
            # Pad / truncate to exactly n_pairs_per_axis
            if lon_freqs.numel() < n_pairs_per_axis:
                # Tile up
                reps = (n_pairs_per_axis + lon_freqs.numel() - 1) // lon_freqs.numel()
                lon_freqs = lon_freqs.repeat(reps)[:n_pairs_per_axis]
            else:
                lon_freqs = lon_freqs[:n_pairs_per_axis]

        # Position grids in radians (matches data/matterport3d/preprocess._py360_dirs_grid)
        # v_norm ∈ [-1 + 1/H, 1 - 1/H], lat = -(π/2)·v_norm
        # u_norm ∈ [-1 + 1/W, 1 - 1/W], lon = π·u_norm
        v_norm = torch.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H)
        u_norm = torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W)
        lat = -(math.pi / 2.0) * v_norm                           # (H,)
        if equal_area_latitudes:
            sin_lat = torch.sin(lat)
            lat = torch.asin(torch.linspace(
                float(sin_lat[0]), float(sin_lat[-1]), H
            ).clamp(-1 + 1e-6, 1 - 1e-6))
        lon = math.pi * u_norm                                    # (W,)

        lat_grid = lat[:, None].expand(H, W).reshape(H * W)       # (L,)
        lon_grid = lon[None, :].expand(H, W).reshape(H * W)       # (L,)

        # Angles: (L, n_pairs)
        lat_ang = lat_grid[:, None] * lat_freqs[None, :]
        lon_ang = lon_grid[:, None] * lon_freqs[None, :]

        # Cache cos/sin
        # Register as non-persistent buffers (recompute at load time if H/W differ)
        self.register_buffer("lat_cos", lat_ang.cos(), persistent=False)
        self.register_buffer("lat_sin", lat_ang.sin(), persistent=False)
        self.register_buffer("lon_cos", lon_ang.cos(), persistent=False)
        self.register_buffer("lon_sin", lon_ang.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate x of shape (..., L, 2*P) using 'split-half' RoPE variant.

        Splits last dim into (real, imag). Output:
          real' = real * cos - imag * sin
          imag' = real * sin + imag * cos

        cos, sin: (L, P)  (or broadcastable).
        """
        P = cos.shape[-1]
        real = x[..., :P]
        imag = x[..., P:2 * P]
        # cos/sin broadcast from (L, P) to (..., L, P)
        while cos.dim() < real.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        return torch.cat(
            [real * cos - imag * sin, real * sin + imag * cos],
            dim=-1,
        )

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D RoPE to x of shape (..., L, head_dim) where L = H*W.

        First half of head_dim ↔ latitude rotation (continuous freqs).
        Second half                   ↔ longitude rotation (integer freqs, 2π-periodic).
        """
        half = self.head_dim // 2
        x_lat = x[..., :half]
        x_lon = x[..., half:]
        x_lat_rot = self._rotate_half(x_lat, self.lat_cos, self.lat_sin)
        x_lon_rot = self._rotate_half(x_lon, self.lon_cos, self.lon_sin)
        return torch.cat([x_lat_rot, x_lon_rot], dim=-1)


class CrossViewBlockRoPE(nn.Module):
    """CrossViewBlock with 2D RoPE-circular applied to Q, K.

    Structure identical to CrossViewBlock: concat A+B tokens, self-attn, split.
    Only difference: Q and K get RoPE rotation before attention. V is untouched
    (RoPE is a relative-position bias, not a value transformation).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_ratio: float = 4.0,
        rope: RoPE2DCircular | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        ffn_dim = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim)
        )
        self.rope = rope   # shared across blocks; may be None to disable

    def forward(self, x_A: torch.Tensor, x_B: torch.Tensor):
        N = x_A.shape[1]
        x = torch.cat([x_A, x_B], dim=1)                        # (B, 2N, D)

        h = self.norm1(x)
        B, L, D = h.shape
        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)                                  # (B, L, nH, hd)
        q = q.transpose(1, 2).contiguous()                       # (B, nH, L, hd)
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        if self.rope is not None:
            # L = 2N, where first N tokens are from view A and next N from view B.
            # Both have the SAME (H, W) grid, so apply RoPE with the same table
            # to each half independently.
            q_A, q_B = q[:, :, :N], q[:, :, N:]
            k_A, k_B = k[:, :, :N], k[:, :, N:]
            q = torch.cat([self.rope.apply(q_A), self.rope.apply(q_B)], dim=2)
            k = torch.cat([self.rope.apply(k_A), self.rope.apply(k_B)], dim=2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        x = x + self.proj(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x[:, :N], x[:, N:]


class SelfViewBlockRoPE(nn.Module):
    """SelfViewBlock with RoPE-circular on Q, K."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_ratio: float = 4.0,
        rope: RoPE2DCircular | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        ffn_dim = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim)
        )
        self.rope = rope

    def forward(self, x: torch.Tensor):
        h = self.norm1(x)
        B, N, D = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        if self.rope is not None:
            q = self.rope.apply(q)
            k = self.rope.apply(k)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x

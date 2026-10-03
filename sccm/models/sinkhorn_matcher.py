"""
exp32_sinkhorn_matching / exp34_dual_softmax_matching
— Sinkhorn OT matching: unified GP replacement.

Unifies GP (Bayesian posterior) and UFM (softmax attention) under the
Optimal Transport framework:

  GP:    μ = K_AB · (K_BB + σ²I)⁻¹ · PE(B)   — O(n³), col-normalized
  UFM:   μ = softmax(S / τ)       · PE(B)   — O(n²), row-only
  Ours:  μ = Sinkhorn(S / τ, K)   · PE(B)   — O(n²K), doubly-stochastic

Sinkhorn iteration alternates row and column normalization, converging
to the optimal transport plan. This gives GP's column normalization
benefit (prevents many-to-one) at UFM's O(n²) cost.

Matching modes:
  - "sinkhorn": K iterations of row+col normalize (default)
  - "dual_softmax": P = softmax(row) * softmax(col) (1-step approx, LoFTR-style)
  - "softmax": P = softmax(row) only (standard UFM)

Interface: forward(x, y) → (B, gp_dim, H, W) — same as GP.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sccm.models.ufm_info_sharing import CrossViewBlock, SelfViewBlock


class SinkhornMatcher(nn.Module):

    def __init__(
        self,
        in_dim: int = 512,
        dim: int = 512,
        n_cross_blocks: int = 4,
        n_heads: int = 8,
        ffn_ratio: float = 4.0,
        temp: float = 0.1,
        sinkhorn_iters: int = 5,
        matching_mode: str = "sinkhorn",  # "sinkhorn", "dual_softmax", "softmax"
    ):
        super().__init__()
        self.dim = dim
        self.temp = temp
        self.sinkhorn_iters = sinkhorn_iters
        self.matching_mode = matching_mode

        # Input/output projections
        self.input_proj = nn.Linear(in_dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        # Cross-view feature enrichment (reuse blocks from ufm_info_sharing)
        self.cross_blocks = nn.ModuleList([
            CrossViewBlock(dim, n_heads, ffn_ratio) for _ in range(n_cross_blocks)
        ])
        self.self_blocks = nn.ModuleList([
            SelfViewBlock(dim, n_heads, ffn_ratio) for _ in range(n_cross_blocks)
        ])

        # Fourier PE — same basis as GP: cos(8π · Conv2d(2, dim))
        self.pos_conv = nn.Conv2d(2, dim, 1, 1)

    def _get_fourier_pe(self, B: int, H: int, W: int, device: torch.device):
        """GP-compatible Fourier positional encoding."""
        coords = torch.meshgrid(
            torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=device),
            torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=device),
            indexing="ij",
        )
        coords = torch.stack((coords[1], coords[0]), dim=-1)[None].expand(B, H, W, 2)
        coords = rearrange(coords, "b h w d -> b d h w")
        pe = torch.cos(8 * math.pi * self.pos_conv(coords))  # (B, dim, H, W)
        return rearrange(pe, "b d h w -> b (h w) d")  # (B, N, dim)

    def _sinkhorn(self, S: torch.Tensor) -> torch.Tensor:
        """Log-domain Sinkhorn with detached iterations (gradient-stable).

        Iterations find optimal scaling factors without gradient.
        P is reconstructed from S with fixed scalings, so gradient flows through S.
        """
        with torch.no_grad():
            log_P = S.detach()
            log_r = torch.zeros(S.shape[0], S.shape[1], 1, device=S.device)
            log_c = torch.zeros(S.shape[0], 1, S.shape[2], device=S.device)
            for _ in range(self.sinkhorn_iters):
                log_r = -torch.logsumexp(log_P, dim=-1, keepdim=True)
                log_P = S.detach() + log_r
                log_c = -torch.logsumexp(log_P, dim=-2, keepdim=True)
                log_P = S.detach() + log_r + log_c
        # Reconstruct with gradient through S
        return (S + log_r.detach() + log_c.detach()).exp()

    def _dual_softmax(self, S: torch.Tensor) -> torch.Tensor:
        """Dual softmax: geometric mean of row and col softmax.

        Plain product softmax(row)*softmax(col) collapses to ~0 when N is large
        (each entry ~1/N², sum ~1/N). Use sqrt of product to keep proper scale.
        """
        P_row = F.softmax(S, dim=-1)
        P_col = F.softmax(S, dim=-2)
        # Geometric mean preserves scale: each entry ~1/N, sum ~1
        P = (P_row * P_col).sqrt()
        # Re-normalize rows so they sum to 1 (for valid weighted average)
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-8)
        return P

    def forward(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: source features (B, C, H, W)
            y: target features (B, C, H, W)
        Returns:
            (B, dim, H, W) — same interface as GP
        """
        B, C, H, W = x.shape
        N = H * W

        # 1. Project + cross-view enrichment
        x_tok = self.input_proj(rearrange(x.float(), "b c h w -> b (h w) c"))
        y_tok = self.input_proj(rearrange(y.float(), "b c h w -> b (h w) c"))

        for c_blk, s_blk in zip(self.cross_blocks, self.self_blocks):
            x_tok, y_tok = c_blk(x_tok, y_tok)
            x_tok = s_blk(x_tok)
            y_tok = s_blk(y_tok)

        x_tok = self.norm(x_tok)
        y_tok = self.norm(y_tok)

        # 2. Compute similarity + normalize
        x_n = F.normalize(x_tok, dim=-1)
        y_n = F.normalize(y_tok, dim=-1)
        S = torch.bmm(x_n, y_n.transpose(1, 2)) / self.temp  # (B, N, N)

        if self.matching_mode == "sinkhorn":
            P = self._sinkhorn(S)
        elif self.matching_mode == "dual_softmax":
            P = self._dual_softmax(S)
        else:  # "softmax"
            P = F.softmax(S, dim=-1)

        # Optional cache of the plan for downstream coarse-CE supervision.
        if getattr(self, "_cache_plan", False):
            self._last_plan = P
            self._last_HW = (H, W)

        # 3. Match embedding with GP-compatible Fourier PE
        # CRITICAL: output must match GP's value distribution (std~0.07, bounded)
        # GP outputs cos(8π·Conv2d(coords)) weighted average — naturally bounded [-1,1]
        # We output the same: weighted Fourier PE only (no x_tok addition)
        pe = self._get_fourier_pe(B, H, W, x.device)  # (B, N, dim)
        match_emb = torch.bmm(P, pe)  # (B, N, dim) — bounded like GP output

        return rearrange(match_emb, "b (h w) d -> b d h w", h=H, w=W)

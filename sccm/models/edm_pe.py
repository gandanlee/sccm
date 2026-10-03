"""
EDM-style 3D Cartesian Positional Embedding (re-implementation, Plan B baseline).

Reference: Jung et al., EDM: Equirectangular projection-oriented dense kernelized
matching, CVPR 2025. Eq. (1) for sphere-Cartesian mapping; Eq. (9) for
positional embedding.

EDM's official code is unreleased; this module faithfully re-implements only the
positional embedding so that the EDM PE concept can be ablated inside our
scaffold (R1 cross-attn + dual-softmax) for a fair, controlled comparison.

EDM PE: chi = cos(W * S + b) where
  - S = (S^x, S^y, S^z) = (sin theta cos phi, sin phi, cos theta cos phi)
        with theta = longitude (lambda in our notation),
             phi   = latitude  (varphi in our notation).
    => Identical to our unit ray r(varphi, lambda) used in TPB.
  - W : Linear projection (3 -> D_pe), implemented as 1x1 conv.
  - b : learnable bias (broadcast).
  - cos : element-wise.

The PE is an absolute (input-side) embedding: added to per-token features
*before* the attention cascade. No relative phase mechanism — by construction
this is the ablation target distinguishing EDM (input-side absolute) from our
SPA (intra-attention relative-bias) approach.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class EdmCartesianPE(nn.Module):
    """EDM-style 3D Cartesian positional embedding.

    Parameters
    ----------
    H, W : int
        Coarse-feature spatial dims (rows, cols).
    d_model : int
        Token feature dim. The PE is projected to this dim then added to tokens.
    zero_init : bool
        If True, initialize the linear projection such that the PE output is
        approximately zero at step 0 (identity-init contract). Default True.
    """

    def __init__(
        self,
        H: int,
        W: int,
        d_model: int,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.H = int(H)
        self.W = int(W)
        self.d_model = int(d_model)

        # Precompute unit-ray buffer (H*W, 3) on (lat, lon) grid.
        # Matches our paper notation: r(varphi, lambda).
        ph = torch.linspace(math.pi / 2, -math.pi / 2, self.H)  # rows top->bottom
        lam = torch.linspace(-math.pi, math.pi, self.W + 1)[:-1]  # cols (no dup)
        grid_phi, grid_lam = torch.meshgrid(ph, lam, indexing="ij")
        # EDM Eq. (1): S^x = sin(theta) cos(phi), S^y = sin(phi), S^z = cos(theta) cos(phi)
        # with theta=longitude, phi=latitude. == our r(varphi, lambda) below.
        rx = torch.sin(grid_lam) * torch.cos(grid_phi)
        ry = torch.sin(grid_phi)
        rz = torch.cos(grid_lam) * torch.cos(grid_phi)
        ray = torch.stack([rx, ry, rz], dim=-1).reshape(-1, 3)  # (N, 3)
        self.register_buffer("ray", ray.contiguous())

        # Linear projection: 3 -> d_model (EDM Eq. 9 calls this a 1x1 conv).
        self.proj = nn.Linear(3, d_model, bias=True)
        if zero_init:
            with torch.no_grad():
                nn.init.zeros_(self.proj.weight)
                # Identity-init: cos(0 + pi/2) = 0  =>  PE = 0 at step 0
                #                                       (bit-equivalent to R1).
                # Gradient through cos at this point: -sin(pi/2) = -1, so
                # proj.weight receives a non-zero gradient on the first
                # backward (cos's gradient is *not* dead unlike the b=0 case).
                self.proj.bias.fill_(math.pi / 2.0)

    @torch.no_grad()
    def _grid(self) -> torch.Tensor:
        return self.ray  # (N, 3)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        """Add EDM PE to tokens.

        tok : (B, N, d_model)
        """
        if tok.shape[-1] != self.d_model:
            raise ValueError(
                f"EdmCartesianPE: expected d_model={self.d_model}, "
                f"got tok.shape[-1]={tok.shape[-1]}"
            )
        if tok.shape[1] != self.ray.shape[0]:
            raise ValueError(
                f"EdmCartesianPE: token count {tok.shape[1]} mismatches "
                f"H*W={self.ray.shape[0]} (H={self.H}, W={self.W})"
            )
        ray = self.ray.to(tok.dtype)  # (N, 3)
        pe = torch.cos(self.proj(ray))  # (N, d_model)
        return tok + pe.unsqueeze(0)  # broadcast over batch

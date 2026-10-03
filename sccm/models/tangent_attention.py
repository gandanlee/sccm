"""Tangent-Relative attention positional encoding for ERP.

Replaces RoPE's *global* (lat, lon) phase encoding with a *query-local* sphere
geometry encoding: for every (query, key) pair the relative position is the
log-map of the key's ray onto the query's tangent frame. The resulting 2D
tangent vector is encoded via Fourier features and projected to a scalar
attention bias.

Mathematically:

    ray_q = lat_lon_to_ray(lat_q, lon_q)         (B, Lq, 3)
    ray_k = lat_lon_to_ray(lat_k, lon_k)         (B, Lk, 3)
    basis_q = tangent_basis_erp(ray_q)            (B, Lq, 3, 2)
    delta_3d(q,k) = log_map_sphere(ray_q, ray_k)  (B, Lq, Lk, 3)
    delta_2d(q,k) = einsum(basis_q, delta_3d)     (B, Lq, Lk, 2)
    pe_bias(q,k)  = MLP(fourier(delta_2d))        (B, Lq, Lk, 1)

    attn(q,k) = softmax(q · k / sqrt(d) + pe_bias)

Note that ``delta_2d`` depends only on the (H, W) grid and is precomputed once
as a buffer. Only the small MLP is learned.

This is the **structural** sphere-native primitive that the ERP-native SCM
ablation rests on (cf. ``exp001_scm_tangent_*.yaml``).
"""
from __future__ import annotations

import math
import torch
# ─── SCCM_POLE_EPS: pole-safety clamp for cos(lat) (Eq.6 LST / Eq.7 LVC).
#     Override via env: SCCM_POLE_EPS=1e-4|1e-3|1e-2 (default 1e-3 matches paper).
import os
_SCCM_POLE_EPS = float(os.environ.get("SCCM_POLE_EPS", "1e-3"))

import torch.nn as nn

from sccm.utils.utils_sphere import (
    log_map_sphere,
    tangent_basis_erp,
)


class TangentPE2D(nn.Module):
    """Tangent-relative position encoding producing a learnable (N, N) attn bias.

    Args:
        H, W: feature grid dimensions (lat × lon, e.g. 32 × 64 at stride-14
            medium).
        n_freqs: number of Fourier frequencies for encoding ``delta_2d``.
        hidden_dim: hidden dim of the projection MLP.
        zero_init: if True, the final linear is zero-initialised so the bias
            is identically 0 at init — the attention behaves like plain
            scaled dot-product, which preserves the warm-started baseline.
    """

    def __init__(
        self,
        H: int,
        W: int,
        n_freqs: int = 8,
        hidden_dim: int = 32,
        zero_init: bool = True,
        conformal_enabled: bool = False,
        conformal_alpha_init: float = 0.0,
    ) -> None:
        """
        conformal_enabled (E6 — Conformal Tangent):
            If True, scale the per-query log-map vector by cos(lat_q)^α before
            Fourier encoding. α is a learnable scalar (init=`conformal_alpha_init`,
            default 0 → cos(·)^0 = 1 = identity → safety contract preserved).
            Mathematically:
                δ̃(q,k) = cos(φ_q)^α · δ(q,k)   (α learnable scalar)
                pe_bias(q,k) = MLP(fourier(δ̃))
            This embeds ERP's metric distortion (cos(φ) anisotropy) into the
            *tangent coordinate* itself rather than as an external prior, keeping
            the contribution along the Tangent CA design line.
        conformal_alpha_init: initial value for α (unconstrained scalar). 0.0
            recommended for safety contract.
        """
        super().__init__()
        self.H, self.W = int(H), int(W)
        self.n_freqs = int(n_freqs)
        N = self.H * self.W

        # ----- Precompute delta_2d table once (depends only on grid) -----
        with torch.no_grad():
            v_norm = torch.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, dtype=torch.float32)
            u_norm = torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, dtype=torch.float32)
            lat = -(math.pi / 2.0) * v_norm                      # (H,)
            lon = math.pi * u_norm                                # (W,)
            lat_grid = lat[:, None].expand(H, W).reshape(N)       # (N,)
            lon_grid = lon[None, :].expand(H, W).reshape(N)       # (N,)

            cos_lat = torch.cos(lat_grid)
            ray = torch.stack(
                [
                    cos_lat * torch.sin(lon_grid),  # x
                    torch.sin(lat_grid),            # y
                    cos_lat * torch.cos(lon_grid),  # z
                ],
                dim=-1,
            )                                                     # (N, 3)

            basis = tangent_basis_erp(ray)                        # (N, 3, 2)

            ray_i = ray[:, None, :].expand(N, N, 3).contiguous()
            ray_j = ray[None, :, :].expand(N, N, 3).contiguous()
            delta_3d = log_map_sphere(ray_i, ray_j)               # (N, N, 3)

            basis_i = basis[:, None, :, :].expand(N, N, 3, 2).contiguous()
            delta_2d = torch.einsum(
                "ijab,ija->ijb", basis_i, delta_3d
            )                                                     # (N, N, 2)

        # buffer (non-persistent — recomputed at load if (H, W) changes)
        self.register_buffer("delta_2d", delta_2d, persistent=False)
        self.register_buffer("lat_grid", lat_grid, persistent=False)   # (N,) per-token latitude
        # geometric Fourier frequencies: 1, 2, 4, ..., 2^(n_freqs-1)
        freqs = (2.0 ** torch.arange(self.n_freqs)).float()
        self.register_buffer("freqs", freqs, persistent=False)

        # ----- Learnable MLP -----
        D_ff = 4 * self.n_freqs                                   # (sin, cos) × (x, y) × n_freqs
        self.pe_proj = nn.Sequential(
            nn.Linear(D_ff, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        if zero_init:
            nn.init.zeros_(self.pe_proj[-1].weight)
            nn.init.zeros_(self.pe_proj[-1].bias)

        # ----- E6 — Conformal Tangent extension -----
        # Scales δ(q,k) → cos(φ_q)^α · δ(q,k) before Fourier encoding.
        # α=0 init → cos(·)^0 = 1 → identity (safety contract: SCCM+E6 ≡ SCCM at init).
        self.conformal_enabled = bool(conformal_enabled)
        if self.conformal_enabled:
            self.conformal_alpha = nn.Parameter(
                torch.tensor(float(conformal_alpha_init), dtype=torch.float32)
            )
        else:
            self.conformal_alpha = None

        # Inference-time TPB bias cache: the (N,N)/(2N,2N) bias is a fixed
        # grid-geometry prior (independent of image content), so recomputing
        # the MLP over N×N pairs every forward is the dominant runtime cost.
        # Cached in eval mode; cleared whenever .train(mode=True) is called so
        # training behaviour is byte-identical to the uncached path.
        self._cached_bias_nn = None
        self._cached_bias_2n = None

    def train(self, mode: bool = True):
        if mode:
            self._cached_bias_nn = None
            self._cached_bias_2n = None
        return super().train(mode)

    @property
    def N(self) -> int:
        return self.H * self.W

    def bias_NN(self) -> torch.Tensor:
        """Return (N, N) attention-bias tensor (single view)."""
        if (not self.training) and self._cached_bias_nn is not None \
                and not os.environ.get("TPB_NO_CACHE"):
            return self._cached_bias_nn
        # Force fp32 for the geometric/fourier path; only the projection is
        # cast back to autocast dtype at the end.
        with torch.cuda.amp.autocast(enabled=False):
            d = self.delta_2d.float()                                  # (N, N, 2)
            # E6 — Conformal Tangent: scale per-query by cos(φ_q)^α.
            if self.conformal_enabled and self.conformal_alpha is not None:
                cos_lat_q = torch.cos(self.lat_grid.float()).clamp(min=_SCCM_POLE_EPS)  # (N,)
                # α may be negative; pow is well-defined since cos_lat ≥ 1e-3 > 0.
                scale_q = cos_lat_q.pow(self.conformal_alpha.float())          # (N,)
                d = d * scale_q.view(-1, 1, 1)                                  # (N, N, 2) — query side
            phases = d[..., None] * self.freqs.float()                 # (N, N, 2, F)
            ff = torch.cat([phases.sin(), phases.cos()], dim=-1)        # (N, N, 2, 2F)
            ff = ff.flatten(-2)                                        # (N, N, 4F)
        # Projection runs in default precision (autocasts back if AMP is on)
        bias = self.pe_proj(ff).squeeze(-1)                             # (N, N)
        if not self.training:
            self._cached_bias_nn = bias.detach()
        return bias

    def bias_2N(self) -> torch.Tensor:
        """Tile (N, N) → (2N, 2N) for two-view cross-attention with concat A+B."""
        if (not self.training) and self._cached_bias_2n is not None \
                and not os.environ.get("TPB_NO_CACHE"):
            return self._cached_bias_2n
        b = self.bias_NN()
        out = torch.cat(
            [
                torch.cat([b, b], dim=1),
                torch.cat([b, b], dim=1),
            ],
            dim=0,
        )
        if not self.training:
            self._cached_bias_2n = out.detach()
        return out


class TangentWignerBias(nn.Module):
    """S²-TPB — Spherical Steerable Tangent-Pairwise Bias (Peter-Weyl form).

    Drop-in replacement for ``TangentPE2D`` (same ``bias_NN`` / ``bias_2N``
    interface) that produces the pairwise attention bias as a *learnable linear
    combination of real Wigner-D matrix elements of the relative rotation*
    g_ij = R_i^{-1} R_j ∈ SO(3), where R_i is the SAME rotation S²-RoPE assigns
    to token i (north pole → token's sphere position).

    Mathematical basis (Peter-Weyl theorem).
        Any b ∈ L²(SO(3)) expands uniquely on the real Wigner-D matrix elements
            b(g) = Σ_{l=0}^{∞} Σ_{m,m'=-l}^{l} c^l_{mm'} D^l_{mm'}(g).
        {D^l_{mm'}} is a complete orthogonal basis of L²(SO(3)). We truncate at
        l ≤ ``lmax`` (a HIGHER-ORDER APPROXIMATION, *not* an exact finite
        solution — completeness needs l→∞). Coefficients c are learned.

    Relation to TangentPE2D (the planar / flat limit — honest claim).
        Near identity (nearby tokens) g_ij = exp(δ̂_ij) with δ ∈ so(3) ≅ the
        tangent log-map. The l=1 block D^1(g) is the 3×3 rotation matrix, whose
        deviation from I is linear in δ → the low-l Wigner expansion reproduces a
        function on the tangent plane. TangentPE2D (MLP∘Fourier of the 2D tangent
        log-map) is therefore the *flat / curvature-free limit*; S²-TPB carries
        the intrinsic SO(3) curvature via the l≥2 terms.
        NOTE: this is a leading-order limit, NOT a bit-exact truncation
        correspondence (Fourier-on-tangent ≠ low-l-Wigner, plus a query-frame
        gauge difference). We claim only the flat-limit reduction.

    Safety contract (differs from LST's bit-identity).
        All coefficients are ZERO-INITIALISED → b_ij ≡ 0 → plain scaled-dot
        attention fallback (same contract as TangentPE2D's zero-init pe_proj).
        S²-TPB does NOT reduce bit-exactly to TangentPE2D at init — only to
        plain attention.

    Memory: the (N, N, B) basis with B = Σ_{l≤lmax}(2l+1)² is precomputed once
        (depends only on the grid). lmax=1 → B=10 (~168 MB at N=2048, fp32),
        lmax=2 → B=35 (~588 MB). Stored as a non-persistent fp32 buffer.
    """

    def __init__(self, H: int, W: int, lmax: int = 2) -> None:
        super().__init__()
        from sccm.models.rope_wigner import wigner_D_real  # reuse S²-RoPE Wigner code

        self.H, self.W, self.lmax = int(H), int(W), int(lmax)
        N = self.H * self.W

        # ERP grid (identical convention to RoPE2DWigner / TangentPE2D).
        v = torch.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, dtype=torch.float64)
        u = torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, dtype=torch.float64)
        lat = -(math.pi / 2.0) * v
        lon = math.pi * u
        lat_g = lat[:, None].expand(H, W).reshape(-1).numpy()
        lon_g = lon[None, :].expand(H, W).reshape(-1).numpy()

        # Per-token, per-degree real Wigner-D blocks D^l(R_t), then the relative
        # rotation block D^l(g_ij) = D^l(R_i)^T D^l(R_j) (real-orthogonal ⇒ inv=Tᵀ).
        # basis[i, j, ·] concatenates the (2l+1)² matrix elements over l=0..lmax.
        slabs = []
        for l in range(self.lmax + 1):
            d = 2 * l + 1
            Dl = torch.empty(N, d, d, dtype=torch.float64)
            for t in range(N):
                alpha = float(lon_g[t])                 # azimuth = longitude
                beta = math.pi / 2.0 - float(lat_g[t])  # colatitude
                Dl[t] = torch.from_numpy(wigner_D_real(l, alpha, beta, 0.0))
            # D^l(g_ij)_{mn} = Σ_k D^l(R_i)_{km} D^l(R_j)_{kn}
            g = torch.einsum("ikm,jkn->ijmn", Dl, Dl)   # (N, N, d, d)
            slabs.append(g.reshape(N, N, d * d))
            del Dl, g
        basis = torch.cat(slabs, dim=-1).float()         # (N, N, B)

        self.B = basis.shape[-1]
        self.register_buffer("basis", basis, persistent=False)
        # Learnable Peter-Weyl coefficients; zero-init → bias ≡ 0 (safety contract).
        self.coeff = nn.Parameter(torch.zeros(self.B, dtype=torch.float32))

    @property
    def N(self) -> int:
        return self.H * self.W

    def bias_NN(self) -> torch.Tensor:
        """Return (N, N) attention-bias tensor (single view), head-shared."""
        with torch.cuda.amp.autocast(enabled=False):
            b = torch.einsum("ijb,b->ij", self.basis.float(), self.coeff.float())
        return b                                         # (N, N)

    def bias_2N(self) -> torch.Tensor:
        """Tile (N, N) → (2N, 2N) for two-view cross-attention with concat A+B."""
        b = self.bias_NN()
        return torch.cat(
            [
                torch.cat([b, b], dim=1),
                torch.cat([b, b], dim=1),
            ],
            dim=0,
        )

"""
SphereCovisMatcher — unified sphere-metric-aware coarse dense matcher for ERP.

All ERP-specific inductive biases derive from the spherical metric
  ds² = dlat² + cos²(lat) dlon²
and are integrated into ONE module. Each can be toggled independently via
config flags for ablation.

## Five metric-derived components

1. **RoPE-circular attention (N2, exp_006)**
   Integer longitude rotary frequencies → SO(2) 2π-periodicity.
   Latitude uses standard RoFormer base. Zero parameters.

2. **Jacobian covis prior (N1, exp_007)**
   μ = σ(MLP(x) + α·log cos(lat)), α = softplus(α_raw) learnable.
   Compensates ERP projection area element dA = cos(lat) dlat dlon.
   +2 scalars (α_A, α_B).

3. **Spherical Harmonic Positional Aggregation (SHPA, Module @1)**
   Replace Fourier PE `cos(8π · Conv2d(coords))` with Y_l^m(lat, lon)
   real SH basis mixed via learnable linear to dim. Integer m gives
   exact 2π-periodicity on lon. Fixes end-to-end equivariance.

4. **Geodesic Distance Attention Bias (GDAB, Module @2)**
   Attention score bias: A_ij += -β · arccos(n_i · n_j), β = softplus(β_raw)
   learnable. Encourages geodesic locality; β→0 recovers standard attention.
   +1 scalar per block.

5. **Latitude-Adaptive Attention Temperature (LAAT, Module @3)**
   Per-query scale: τ(p) = τ_0 · cos(lat_p)^γ, γ = softplus(γ_raw) learnable.
   Sharper softmax at poles (compressed content), softer at equator.
   Info-theoretic: normalizes effective attention bandwidth by pixel area.
   +1 scalar per block.

All additions are parameter-light and mathematically grounded in the
spherical metric. No feature-capacity restrictions (unlike prior
irrep-based SO(2)-Cov attention, `exp_032`, pck1=0.020).

## Paper thesis

> Dense ERP matching requires sphere-metric-aware attention and gating.
> We propose SphereCovis, a single module that packages up to five
> parameter-light metric-derived priors, each targeting a different
> architectural stage: positional encoding, gating, score bias, temperature,
> and aggregation basis. Together they improve pck@1° on Matterport3D by
> X% over a CoMatch-style planar baseline.
"""
from __future__ import annotations

import math
import torch
# ─── SCCM_POLE_EPS: pole-safety clamp for cos(lat) (Eq.6 LST / Eq.7 LVC).
#     Override via env: SCCM_POLE_EPS=1e-4|1e-3|1e-2 (default 1e-3 matches paper).
import os
_SCCM_POLE_EPS = float(os.environ.get("SCCM_POLE_EPS", "1e-3"))

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import sccm
from sccm.models.sinkhorn_matcher import SinkhornMatcher
from sccm.models.rope_circular import RoPE2DCircular
from sccm.utils.utils_sphere import real_spherical_harmonics


# ──────────────────────────────────────────────────────────────────────
# Helper — ERP grid latitude / direction vectors
# ──────────────────────────────────────────────────────────────────────
def _erp_lat_lon_grid(H: int, W: int, device, dtype=torch.float32):
    """Per-token (lat, lon) in radians on the ERP grid (row-major flat)."""
    v_norm = torch.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, device=device, dtype=dtype)
    u_norm = torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device=device, dtype=dtype)
    lat_row = -(math.pi / 2.0) * v_norm                              # (H,)
    lon_col = math.pi * u_norm                                       # (W,)
    lat = lat_row[:, None].expand(H, W).contiguous().view(H * W)     # (N,)
    lon = lon_col[None, :].expand(H, W).contiguous().view(H * W)     # (N,)
    return lat, lon


def _erp_rays(H: int, W: int, device, dtype=torch.float32) -> torch.Tensor:
    """Per-token unit direction vectors (ray) in repo convention:
        x = cos(lat) sin(lon),  y = sin(lat),  z = cos(lat) cos(lon)
    Returns (N=H*W, 3).
    """
    lat, lon = _erp_lat_lon_grid(H, W, device, dtype=dtype)
    cos_lat = torch.cos(lat)
    x = cos_lat * torch.sin(lon)
    y = torch.sin(lat)
    z = cos_lat * torch.cos(lon)
    return torch.stack([x, y, z], dim=-1)                            # (N, 3)


# ──────────────────────────────────────────────────────────────────────
# SphereGenerativeHead — sphere-parametric generative matching (K-mixture)
# ──────────────────────────────────────────────────────────────────────
class SphereGenerativeHead(nn.Module):
    """Score-free generative matching: each query pixel emits a K-mixture of
    Gaussians on the sphere (tangent plane at each component's μ). The matching
    probability P[i, j] = w_i · Σ_k π_{i,k} · N( log_{μ_{i,k}}(p_j) | 0, Σ_{i,k} ).

    No similarity matrix computed; no softmax over chart candidates.

    Inputs (forward):
        x_A_tok    : (B, N_A, D) refined tokens for view A (from SPA stack)
        x_B_tok    : (B, N_B, D) refined tokens for view B (used as cross-context)
        p_B_sphere : (N_B, 3)    sphere positions of view B's pixel grid (unit vectors)

    Output:
        P : (B, N_A, N_B) matching quasi-probability (non-negative, finite)
    """

    def __init__(
        self,
        dim: int,
        K: int = 4,
        hidden: int = 256,
        cross_heads: int = 4,
        sigma_log_min: float = -5.0,
        sigma_log_max: float = 2.0,
        rho_max_abs: float = 0.95,
        w_init_logit: float = 0.0,
    ) -> None:
        super().__init__()
        if K < 1:
            raise ValueError(f"K (mixture order) must be >= 1, got {K}")
        self.K = int(K)
        self.dim = int(dim)
        self.sigma_log_min = float(sigma_log_min)
        self.sigma_log_max = float(sigma_log_max)
        self.rho_max_abs = float(rho_max_abs)

        # Cross-context: single MHA layer where x_A attends to x_B.
        # This injects view-B info into the per-pixel mixture prediction.
        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=cross_heads, batch_first=True
        )
        self.post_norm = nn.LayerNorm(dim)

        # Per-pixel mixture head: dim → hidden → (7K + 1) raw parameters
        #   for each of K components: μ_raw (3) + Σ_raw (3: log s1, log s2, rho_raw) + π_raw (1)
        #   global: w_raw (1)
        self.out_dim = self.K * 7 + 1
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.out_dim),
        )

        # Smart initialization of the final layer:
        # - WEIGHT: Kaiming default (input-dependent forward from step 0; cross-attn
        #   and MLP first layer get non-zero gradients immediately — critical for
        #   from-scratch convergence).
        # - μ_k bias: spread around equator at K evenly-spaced longitudes (sensible prior)
        # - Σ_k bias: log σ = −2.5 (σ ≈ 0.082 rad ≈ 4.7° — sharp enough for meaningful
        #   matching signal; broad enough to cover candidate region at coarse H=32, W=64)
        # - π_k bias: 0 (uniform after softmax)
        # - w bias: w_init_logit
        # WEIGHT stays at default Kaiming (do NOT zero-init); only structure biases.
        with torch.no_grad():
            bias = self.mlp[-1].bias
            for k in range(self.K):
                ang = 2.0 * math.pi * k / max(self.K, 1)
                # μ_k bias: (sin(ang), 0, cos(ang)) — points along equator at angle 'ang'
                bias[3 * k + 0] = math.sin(ang)
                bias[3 * k + 1] = 0.0
                bias[3 * k + 2] = math.cos(ang)
                # Σ_k bias: log s1 = log s2 = −2.5 (σ ≈ 0.082 rad), rho_raw = 0
                bias[3 * self.K + 3 * k + 0] = -2.5
                bias[3 * self.K + 3 * k + 1] = -2.5
                bias[3 * self.K + 3 * k + 2] = 0.0
                # π_k bias: 0
                bias[6 * self.K + k] = 0.0
            # w bias
            bias[7 * self.K] = float(w_init_logit)

    @staticmethod
    def _local_tangent_basis(mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build orthonormal tangent basis (e1, e2) at each μ on S² via Gram-Schmidt
        with y-axis reference. Yaw-equivariant: reference is along sphere's rotation axis.

        mu: (..., 3) unit vectors
        Returns (e1, e2), each (..., 3), satisfying μ·e1 = μ·e2 = e1·e2 = 0, ‖e1‖=‖e2‖=1.
        """
        ref_y = torch.tensor([0.0, 1.0, 0.0], device=mu.device, dtype=mu.dtype)
        ref_y = ref_y.expand_as(mu)                                                # (..., 3)
        # Project ref_y perpendicular to μ:
        ref_perp_y = ref_y - (ref_y * mu).sum(-1, keepdim=True) * mu               # (..., 3)
        norm_perp_y = ref_perp_y.norm(dim=-1, keepdim=True)                        # (..., 1)
        # Fallback: when μ is near the y-poles, use x-axis as reference
        ref_x = torch.tensor([1.0, 0.0, 0.0], device=mu.device, dtype=mu.dtype)
        ref_x = ref_x.expand_as(mu)
        ref_perp_x = ref_x - (ref_x * mu).sum(-1, keepdim=True) * mu
        near_pole = (norm_perp_y < 1e-3)
        ref_perp = torch.where(near_pole, ref_perp_x, ref_perp_y)
        e1 = F.normalize(ref_perp, dim=-1)                                         # (..., 3)
        e2 = torch.cross(mu, e1, dim=-1)                                           # (..., 3)
        return e1, e2

    def forward(
        self,
        x_A_tok: torch.Tensor,
        x_B_tok: torch.Tensor,
        p_B_sphere: torch.Tensor,
    ) -> torch.Tensor:
        B, N_A, D = x_A_tok.shape
        N_B = p_B_sphere.shape[0]
        K = self.K

        # 1. Cross-context: x_A queries attend to x_B
        q = self.cross_norm_q(x_A_tok)
        kv = self.cross_norm_kv(x_B_tok)
        x_A_ctx, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x_A_ctx = self.post_norm(x_A_tok + x_A_ctx)                                # residual

        # 2. Per-pixel mixture parameters
        raw = self.mlp(x_A_ctx)                                                    # (B, N_A, 7K+1)
        mu_raw = raw[..., : 3 * K].reshape(B, N_A, K, 3)
        Sigma_raw = raw[..., 3 * K : 6 * K].reshape(B, N_A, K, 3)
        pi_raw = raw[..., 6 * K : 7 * K]                                           # (B, N_A, K)
        w_raw = raw[..., 7 * K]                                                    # (B, N_A)

        # Reparameterize
        mu = F.normalize(mu_raw, dim=-1)                                           # (B, N_A, K, 3)
        log_s1 = Sigma_raw[..., 0].clamp(self.sigma_log_min, self.sigma_log_max)
        log_s2 = Sigma_raw[..., 1].clamp(self.sigma_log_min, self.sigma_log_max)
        rho = torch.tanh(Sigma_raw[..., 2]) * self.rho_max_abs
        s1 = log_s1.exp()                                                          # (B, N_A, K)
        s2 = log_s2.exp()
        pi = F.softmax(pi_raw, dim=-1)                                             # (B, N_A, K)
        w = torch.sigmoid(w_raw)                                                   # (B, N_A)

        # 3. Build local tangent basis at each (i, k)
        e1, e2 = self._local_tangent_basis(mu)                                     # each (B, N_A, K, 3)

        # 4. Pairwise (i, k, j) tangent vector components (closed form, no 3D tangent vector materialized)
        # cos_theta[b, i, k, j] = μ_{b,i,k} · p_B[j]
        p_B_exp = p_B_sphere.view(1, 1, 1, N_B, 3).to(mu.dtype)                    # (1, 1, 1, N_B, 3)
        mu_exp = mu.unsqueeze(-2)                                                  # (B, N_A, K, 1, 3)
        cos_theta = (mu_exp * p_B_exp).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)          # (B, N_A, K, N_B)
        theta = torch.arccos(cos_theta)                                            # (B, N_A, K, N_B)
        sin_theta = torch.sin(theta).clamp(min=1e-6)
        scale = theta / sin_theta                                                  # (B, N_A, K, N_B)

        # v1 = scale · (p_B · e1),    v2 = scale · (p_B · e2)   (μ ⊥ e1, e2 ⇒ μ·e1·cos_theta term cancels)
        p_dot_e1 = (e1.unsqueeze(-2) * p_B_exp).sum(-1)                            # (B, N_A, K, N_B)
        p_dot_e2 = (e2.unsqueeze(-2) * p_B_exp).sum(-1)
        v1 = scale * p_dot_e1                                                      # (B, N_A, K, N_B)
        v2 = scale * p_dot_e2

        # 5. Mahalanobis distance + normalization constant
        s1_e = s1.unsqueeze(-1)                                                    # (B, N_A, K, 1)
        s2_e = s2.unsqueeze(-1)
        rho_e = rho.unsqueeze(-1)
        one_m_rho2 = (1.0 - rho_e * rho_e).clamp(min=1e-6)
        # Mahal = (v1²/s1² - 2ρv1v2/(s1s2) + v2²/s2²) / (1 - ρ²)
        mahal = (
            (v1 * v1) / (s1_e * s1_e)
            - 2.0 * rho_e * v1 * v2 / (s1_e * s2_e)
            + (v2 * v2) / (s2_e * s2_e)
        ) / one_m_rho2
        # log_det(Σ) = 2(log s1 + log s2) + log(1 - ρ²)
        log_det = 2.0 * (log_s1 + log_s2) + torch.log(one_m_rho2.squeeze(-1))
        log_det_exp = log_det.unsqueeze(-1)                                        # (B, N_A, K, 1)
        # log Gaussian density: -0.5 mahal - 0.5 log_det - log(2π)
        log_dens = -0.5 * mahal - 0.5 * log_det_exp - math.log(2.0 * math.pi)      # (B, N_A, K, N_B)

        # 6. Mixture: log Σ_k π_k · dens_k = logsumexp_k(log π_k + log dens_k)
        log_pi = torch.log(pi.clamp(min=1e-12)).unsqueeze(-1)                      # (B, N_A, K, 1)
        log_mix = torch.logsumexp(log_pi + log_dens, dim=2)                        # (B, N_A, N_B)

        # 7. Confidence gating (multiplicative in linear, additive in log)
        log_w = torch.log(w.clamp(min=1e-12)).unsqueeze(-1)                        # (B, N_A, 1)
        log_P = log_mix + log_w                                                    # (B, N_A, N_B)
        P = log_P.exp()                                                            # (B, N_A, N_B)
        return P


# ──────────────────────────────────────────────────────────────────────
# SphereCovisHead — single-μ score-bias head with optional Jacobian + SMG geometry input
# ──────────────────────────────────────────────────────────────────────
class SphereCovisHead(nn.Module):
    """Per-query score-bias head (formerly "covis head" — renamed for SMG framing).

    Default mode: μ = σ(MLP(tok) + α·log cos(φ))
    SMG mode (smg_lat_freqs or smg_include_log_cos set):
        μ = σ(MLP([tok, cos(k·φ), sin(k·φ), log cos(φ)]) + α·log cos(φ))
    MLP input augmented with sphere-metric-derived geometry scalars.
    """

    def __init__(
        self,
        dim: int,
        hidden: int = 128,
        init_logit: float = 4.0,
        alpha_init: float = 1.0,
        alpha_learnable: bool = True,
        jacobian_enabled: bool = True,
        smg_lat_freqs: list | None = None,        # e.g., [1, 2, 4] → cos/sin k·φ
        smg_include_log_cos: bool = False,        # append log cos(φ) to MLP input
        # ── ALAC — Adaptive LAC (feature-modulated α(x) per pixel) ──
        # Replaces scalar α with α(x)=softplus(MLP_alpha(tok)). When the
        # MLP_alpha last-layer is zero-init + bias=log(e-1), α(x)≡1.0
        # everywhere and ALAC is bit-equivalent to current LAC at init.
        alac_enabled: bool = False,
        alac_hidden: int = 64,
        alac_alpha_init: float = 1.0,    # target softplus output per pixel at init
        # ── DAC — Depth-Aware Area Correction (extension of LAC) ──
        # Adds a learned per-pixel "depth proxy" term to the logit, on top of
        # the standard LAC area term. The depth proxy is a scalar MLP output
        # interpretable as log of an unobserved depth-like quantity.
        #   ℓ_p = MLP_covis(x_p) + α · log cos φ_p + β · d_proxy(x_p)
        # Safe init: MLP_d zero-init + β=0 → identity to R3 SCCM.
        dac_enabled: bool = False,
        dac_hidden: int = 64,
        dac_beta_init: float = 0.0,      # scalar; zero-init → identity at start
        dac_beta_learnable: bool = True,
        # ── HAAC — Harmonic Area-Aware Correction (LAC generalization) ──
        # Generalizes LAC's fixed α·log cos(φ) with a learnable spectral
        # function of latitude (longitude-invariant spherical harmonics, m=0):
        #   g(φ) = α₀·log cos(φ) + Σ_{k=1..K} (a_k cos(kφ) + b_k sin(kφ))
        # ERP-native (uses only φ), feature-free (no MLP_covis redundancy),
        # tiny (2·K scalars). Zero-init a_k=b_k=0 → identity to LAC at start.
        haac_enabled: bool = False,
        haac_K: int = 4,
        haac_learnable: bool = True,
    ) -> None:
        super().__init__()
        # SMG: compute augmented input dim
        self.smg_lat_freqs = tuple(smg_lat_freqs) if smg_lat_freqs else tuple()
        self.smg_include_log_cos = bool(smg_include_log_cos)
        extra = 2 * len(self.smg_lat_freqs) + (1 if self.smg_include_log_cos else 0)
        self.fc1 = nn.Linear(dim + extra, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        with torch.no_grad():
            self.fc2.weight.zero_()
            self.fc2.bias.fill_(float(init_logit))
        self.jacobian_enabled = bool(jacobian_enabled)
        if self.jacobian_enabled:
            if alpha_init <= 0:
                raise ValueError(f"alpha_init must be positive, got {alpha_init}")
            alpha_raw_init = math.log(math.expm1(float(alpha_init)))
            t = torch.tensor(alpha_raw_init, dtype=torch.float32)
            if alpha_learnable:
                self.alpha_raw = nn.Parameter(t)
            else:
                self.register_buffer("alpha_raw", t)
        else:
            self.alpha_raw = None
        # ── ALAC head ──
        self.alac_enabled = bool(alac_enabled)
        if self.alac_enabled:
            self.alac_mlp = nn.Sequential(
                nn.Linear(dim, alac_hidden),
                nn.GELU(),
                nn.Linear(alac_hidden, 1),
            )
            # Init: last layer w=0, bias=log(expm1(alac_alpha_init))
            # → softplus(bias) = alac_alpha_init (per-pixel α at init).
            nn.init.zeros_(self.alac_mlp[-1].weight)
            bias_val = math.log(math.expm1(float(alac_alpha_init)))
            nn.init.constant_(self.alac_mlp[-1].bias, bias_val)
        else:
            self.alac_mlp = None
        # Diagnostics
        self._last_alac_alpha_mean = 0.0
        self._last_alac_alpha_std = 0.0
        self._last_alac_alpha_min = 0.0
        self._last_alac_alpha_max = 0.0

        # ── DAC head ── (depth proxy MLP + scalar β)
        self.dac_enabled = bool(dac_enabled)
        if self.dac_enabled:
            self.dac_mlp = nn.Sequential(
                nn.Linear(dim, dac_hidden),
                nn.GELU(),
                nn.Linear(dac_hidden, 1),
            )
            # Zero last-layer weight + zero bias → d_proxy ≡ 0 at init
            nn.init.zeros_(self.dac_mlp[-1].weight)
            nn.init.zeros_(self.dac_mlp[-1].bias)
            t = torch.tensor(float(dac_beta_init), dtype=torch.float32)
            if dac_beta_learnable:
                self.dac_beta = nn.Parameter(t)
            else:
                self.register_buffer('dac_beta', t)
        else:
            self.dac_mlp = None
            self.dac_beta = None
        # Diagnostics
        self._last_dac_beta = 0.0
        self._last_dac_dproxy_mean = 0.0
        self._last_dac_dproxy_std = 0.0
        self._last_dac_logit_delta_mean = 0.0

        # ── HAAC head ── Harmonic spectral basis on latitude
        self.haac_enabled = bool(haac_enabled)
        self.haac_K = int(haac_K) if self.haac_enabled else 0
        if self.haac_enabled:
            if self.haac_K <= 0:
                raise ValueError(f"haac_K must be >= 1, got {self.haac_K}")
            self.register_buffer(
                "_haac_k_idx",
                torch.arange(1, self.haac_K + 1, dtype=torch.float32),
            )
            a = torch.zeros(self.haac_K, dtype=torch.float32)
            b = torch.zeros(self.haac_K, dtype=torch.float32)
            if haac_learnable:
                self.haac_a = nn.Parameter(a)
                self.haac_b = nn.Parameter(b)
            else:
                self.register_buffer("haac_a", a)
                self.register_buffer("haac_b", b)
        else:
            self.haac_a = None
            self.haac_b = None
        self._last_haac_a_l1 = 0.0
        self._last_haac_b_l1 = 0.0
        self._last_haac_delta_mean = 0.0
        self._last_haac_delta_std = 0.0

    @property
    def alpha(self):
        return F.softplus(self.alpha_raw) if self.alpha_raw is not None else None

    def forward_logit(self, tok: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        """Return pre-sigmoid logit (= MLP + LAC term). Exposed for GAC
        which needs to inject an additional term into the same logit
        before the sigmoid."""
        if len(self.smg_lat_freqs) > 0 or self.smg_include_log_cos:
            enc_parts = []
            for k in self.smg_lat_freqs:
                enc_parts.append(torch.cos(k * lat).unsqueeze(-1))
                enc_parts.append(torch.sin(k * lat).unsqueeze(-1))
            if self.smg_include_log_cos:
                enc_parts.append(torch.log(torch.cos(lat).clamp(min=_SCCM_POLE_EPS)).unsqueeze(-1))
            enc = torch.cat(enc_parts, dim=-1)                       # (B, N, extra)
            mlp_in = torch.cat([tok, enc.to(tok.dtype)], dim=-1)
        else:
            mlp_in = tok
        h = F.gelu(self.fc1(mlp_in))
        logit = self.fc2(h).squeeze(-1)                              # (B, N)
        if self.jacobian_enabled:
            log_cos_lat = torch.log(torch.cos(lat).clamp(min=_SCCM_POLE_EPS))
            if self.alac_enabled and self.alac_mlp is not None:
                # ALAC: per-pixel α(x) replaces scalar α
                alpha_x = F.softplus(self.alac_mlp(tok).squeeze(-1))   # (B, N)
                logit = logit + alpha_x * log_cos_lat
                with torch.no_grad():
                    self._last_alac_alpha_mean = float(alpha_x.mean().item())
                    self._last_alac_alpha_std  = float(alpha_x.std().item())
                    self._last_alac_alpha_min  = float(alpha_x.min().item())
                    self._last_alac_alpha_max  = float(alpha_x.max().item())
            else:
                # Standard LAC: scalar α
                logit = logit + F.softplus(self.alpha_raw) * log_cos_lat

        # DAC: depth-aware add-on (learned scalar depth proxy per pixel)
        if self.dac_enabled and self.dac_mlp is not None:
            d_proxy = self.dac_mlp(tok).squeeze(-1)             # (B, N)
            beta = self.dac_beta
            logit_dac = beta * d_proxy
            logit = logit + logit_dac
            with torch.no_grad():
                beta_v = beta.detach().item() if isinstance(beta, torch.Tensor) else float(beta)
                self._last_dac_beta = float(beta_v)
                self._last_dac_dproxy_mean = float(d_proxy.mean().item())
                self._last_dac_dproxy_std = float(d_proxy.std().item())
                self._last_dac_logit_delta_mean = float(logit_dac.abs().mean().item())

        # HAAC: harmonic spectral correction (pure latitude function, no features)
        if self.haac_enabled and self.haac_a is not None:
            # lat: (B, N), k_idx: (K,) → k_phi: (B, N, K)
            k_phi = lat.unsqueeze(-1) * self._haac_k_idx.to(lat.dtype)
            cos_k = torch.cos(k_phi)
            sin_k = torch.sin(k_phi)
            a = self.haac_a.to(cos_k.dtype)
            b = self.haac_b.to(sin_k.dtype)
            haac_delta = (cos_k * a).sum(-1) + (sin_k * b).sum(-1)   # (B, N)
            logit = logit + haac_delta
            with torch.no_grad():
                self._last_haac_a_l1 = float(self.haac_a.detach().abs().sum().item())
                self._last_haac_b_l1 = float(self.haac_b.detach().abs().sum().item())
                self._last_haac_delta_mean = float(haac_delta.abs().mean().item())
                self._last_haac_delta_std = float(haac_delta.float().std().item())
        return logit

    def forward(self, tok: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logit(tok, lat))


# ──────────────────────────────────────────────────────────────────────
# SphereAttnBlock — RoPE + optional geodesic bias + optional lat-adaptive temperature
# ──────────────────────────────────────────────────────────────────────
def _manual_attn(q, k, v, *, bias=None, per_query_scale=None) -> torch.Tensor:
    """Manual attention supporting additive bias and per-query scaling.

    q, k, v:            (B, nH, L, d)
    bias:               (L_q, L_k) float additive  (or None)
    per_query_scale:    (L_q,) float multiplying each query's score (or None)

    Equivalent to F.scaled_dot_product_attention with extra flexibility.
    """
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale             # (B, nH, L_q, L_k)
    if per_query_scale is not None:
        # per-query multiplier (=> effective temperature divider)
        # Note: we multiply AFTER the base scale, so final scaling is scale * per_query_scale
        scores = scores * per_query_scale.view(1, 1, -1, 1)
    if bias is not None:
        scores = scores + bias                                         # broadcast over batch, heads
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


class SphereAttnCrossBlock(nn.Module):
    """CrossBlock with RoPE + optional geodesic bias (GDAB) + optional lat-adaptive temperature (LAAT).

    Structure mirrors CrossViewBlockRoPE: concat A+B tokens, self-attn, split.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_ratio: float = 4.0,
        rope: RoPE2DCircular | None = None,
        geodesic_bias: bool = False,
        geodesic_beta_init: float = 0.1,
        lat_adaptive_temp: bool = False,
        lat_gamma_init: float = 0.5,
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

        # GDAB (Module @2)
        self.gdab = bool(geodesic_bias)
        if self.gdab:
            b = float(geodesic_beta_init)
            if b <= 0: raise ValueError("geodesic_beta_init must be > 0")
            self.beta_raw = nn.Parameter(torch.tensor(math.log(math.expm1(b)), dtype=torch.float32))

        # LAAT (Module @3)
        self.laat = bool(lat_adaptive_temp)
        if self.laat:
            g = float(lat_gamma_init)
            if g < 0: raise ValueError("lat_gamma_init must be ≥ 0")
            # Allow γ = 0 (softplus^{-1}(0) = -inf, approximate with large negative)
            g_raw = math.log(math.expm1(max(g, 1e-4)))
            self.gamma_raw = nn.Parameter(torch.tensor(g_raw, dtype=torch.float32))

    def forward(self, x_A, x_B, *, d_geo_NN=None, lat_N=None, tangent_bias_2N=None, cyclic_bias_2N=None, rope_R=None, extra_bias_2N=None, tangent_phase=None):
        """
        x_A, x_B : (B, N, D)
        d_geo_NN : (N, N) geodesic distance matrix on ERP feature grid, or None
        lat_N    : (N,) per-token latitude in radians, or None
        tangent_bias_2N : (2N, 2N) Tangent-Relative PE bias, or None.
            Added to the attention logits — independent of GDAB.
        cyclic_bias_2N : (n_heads, 2N, 2N) Cyclic-W relative position bias, or None.
            Added to attention logits with per-head granularity. A↔A and B↔B
            sub-blocks use cyclic mod-W indexing; A↔B and B↔A sub-blocks zero.
        """
        N = x_A.shape[1]
        x = torch.cat([x_A, x_B], dim=1)                             # (B, 2N, D)

        h = self.norm1(x)
        B, L, D = h.shape
        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        if self.rope is not None:
            q_A, q_B = q[:, :, :N], q[:, :, N:]
            k_A, k_B = k[:, :, :N], k[:, :, N:]
            if rope_R is not None:
                # Rotation-aligned cross attention: B-view positions rotated into
                # A's frame by the estimated relative rotation R (full-coverage).
                q = torch.cat([self.rope.apply(q_A), self.rope.apply(q_B, R=rope_R)], dim=2)
                k = torch.cat([self.rope.apply(k_A), self.rope.apply(k_B, R=rope_R)], dim=2)
            else:
                q = torch.cat([self.rope.apply(q_A), self.rope.apply(q_B)], dim=2)
                k = torch.cat([self.rope.apply(k_A), self.rope.apply(k_B)], dim=2)

        # Build optional attn bias / per-query scale
        bias = None
        per_query_scale = None

        if self.gdab and d_geo_NN is not None:
            beta = F.softplus(self.beta_raw)
            D_NN = d_geo_NN.to(dtype=q.dtype, device=q.device)
            # Tile (N, N) → (2N, 2N). Both A, B sides are ERP grids, so
            # all four (A↔A, A↔B, B↔A, B↔B) sub-blocks use same grid distances.
            D_2N = torch.cat([torch.cat([D_NN, D_NN], dim=1),
                              torch.cat([D_NN, D_NN], dim=1)], dim=0)  # (2N, 2N)
            bias = -beta * D_2N

        if tangent_bias_2N is not None:
            tb = tangent_bias_2N.to(dtype=q.dtype, device=q.device)   # (2N, 2N)
            bias = tb if bias is None else (bias + tb)

        if cyclic_bias_2N is not None:
            # cyclic_bias_2N: (n_heads, 2N, 2N) → broadcast over batch as (1, n_heads, 2N, 2N)
            cb = cyclic_bias_2N.to(dtype=q.dtype, device=q.device).unsqueeze(0)
            if bias is None:
                bias = cb
            elif bias.dim() == 2:
                bias = bias.unsqueeze(0).unsqueeze(0) + cb
            else:
                bias = bias + cb

        if extra_bias_2N is not None:
            # Per-batch additive logit bias (e.g. gated epipolar great-circle bias).
            # (B, 2N, 2N) -> (B, 1, 2N, 2N) to broadcast over heads.
            eb = extra_bias_2N.to(dtype=q.dtype, device=q.device).unsqueeze(1)
            if bias is None:
                bias = eb
            elif bias.dim() == 2:
                bias = bias.unsqueeze(0).unsqueeze(0) + eb
            elif bias.dim() == 3:
                bias = bias.unsqueeze(1) + eb
            else:
                bias = bias + eb

        if tangent_phase is not None:
            # R3-T: content-dependent tangent-phase score (B, nH, 2N, 2N). gate=0 -> no-op.
            ps = tangent_phase.phase_scores_2N(q, k)
            if bias is None:
                bias = ps
            else:
                while bias.dim() < 4:
                    bias = bias.unsqueeze(0)
                bias = bias + ps

        if self.laat and lat_N is not None:
            gamma = F.softplus(self.gamma_raw)
            cos_lat = torch.cos(lat_N).clamp(min=_SCCM_POLE_EPS)                 # (N,)
            # τ(p) = cos(lat)^γ;  scale = 1/τ(p)  (applied multiplicatively to Q·K).
            # Here we pass the MULTIPLIER (= 1/τ), so manual_attn multiplies scores by it.
            scale_N = cos_lat.pow(-gamma)                              # (N,)
            scale_2N = torch.cat([scale_N, scale_N], dim=0).to(dtype=q.dtype, device=q.device)
            per_query_scale = scale_2N

        if bias is None and per_query_scale is None:
            attn_out = F.scaled_dot_product_attention(q, k, v)
        elif per_query_scale is None:
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        else:
            attn_out = _manual_attn(q, k, v, bias=bias, per_query_scale=per_query_scale)

        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        x = x + self.proj(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x[:, :N], x[:, N:]


class SphereAttnSelfBlock(nn.Module):
    """SelfBlock with RoPE + optional GDAB + optional LAAT."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_ratio: float = 4.0,
        rope: RoPE2DCircular | None = None,
        geodesic_bias: bool = False,
        geodesic_beta_init: float = 0.1,
        lat_adaptive_temp: bool = False,
        lat_gamma_init: float = 0.5,
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

        self.gdab = bool(geodesic_bias)
        if self.gdab:
            b = float(geodesic_beta_init)
            if b <= 0: raise ValueError("geodesic_beta_init must be > 0")
            self.beta_raw = nn.Parameter(torch.tensor(math.log(math.expm1(b)), dtype=torch.float32))

        self.laat = bool(lat_adaptive_temp)
        if self.laat:
            g = float(lat_gamma_init)
            if g < 0: raise ValueError("lat_gamma_init must be ≥ 0")
            g_raw = math.log(math.expm1(max(g, 1e-4)))
            self.gamma_raw = nn.Parameter(torch.tensor(g_raw, dtype=torch.float32))

    def forward(self, x, *, d_geo_NN=None, lat_N=None, tangent_bias_NN=None, cyclic_bias_NN=None, tangent_phase=None):
        """
        cyclic_bias_NN : (n_heads, N, N) Cyclic-W relative position bias, or None.
        """
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

        bias = None
        per_query_scale = None

        if self.gdab and d_geo_NN is not None:
            beta = F.softplus(self.beta_raw)
            bias = -beta * d_geo_NN.to(dtype=q.dtype, device=q.device)

        if tangent_bias_NN is not None:
            tb = tangent_bias_NN.to(dtype=q.dtype, device=q.device)   # (N, N)
            bias = tb if bias is None else (bias + tb)

        if cyclic_bias_NN is not None:
            # cyclic_bias_NN: (n_heads, N, N) → broadcast as (1, n_heads, N, N)
            cb = cyclic_bias_NN.to(dtype=q.dtype, device=q.device).unsqueeze(0)
            if bias is None:
                bias = cb
            elif bias.dim() == 2:
                bias = bias.unsqueeze(0).unsqueeze(0) + cb
            else:
                bias = bias + cb

        if tangent_phase is not None:
            ps = tangent_phase.phase_scores_NN(q, k)                  # (B, nH, N, N); gate=0 -> no-op
            if bias is None:
                bias = ps
            else:
                while bias.dim() < 4:
                    bias = bias.unsqueeze(0)
                bias = bias + ps

        if self.laat and lat_N is not None:
            gamma = F.softplus(self.gamma_raw)
            cos_lat = torch.cos(lat_N).clamp(min=_SCCM_POLE_EPS)
            per_query_scale = cos_lat.pow(-gamma).to(dtype=q.dtype, device=q.device)

        if bias is None and per_query_scale is None:
            attn_out = F.scaled_dot_product_attention(q, k, v)
        elif per_query_scale is None:
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        else:
            attn_out = _manual_attn(q, k, v, bias=bias, per_query_scale=per_query_scale)

        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


# ──────────────────────────────────────────────────────────────────────
# SphereCovisMatcher — unified class
# ──────────────────────────────────────────────────────────────────────
class SphereCovisMatcher(SinkhornMatcher):
    """Sphere-metric-aware coarse matcher for ERP.

    Toggles (each corresponds to one metric-derived prior):
        rope_enabled         — RoPE-circular attention (N2)
        jacobian_enabled     — α·log cos(lat) on covis logit (N1)
        sh_pe_enabled        — SH aggregation replacing Fourier PE (@1 SHPA)
        geodesic_bias_enabled — score bias -β·d_geo in attention (@2 GDAB)
        lat_temp_enabled     — per-query scale cos(lat)^γ in attention (@3 LAAT)

    All independent; defaults below correspond to **exp_007 config** (RoPE+Jac).
    """

    def __init__(
        self,
        in_dim: int = 512,
        dim: int = 512,
        n_cross_blocks: int = 4,
        n_heads: int = 8,
        ffn_ratio: float = 4.0,
        temp: float = 0.1,
        sinkhorn_iters: int = 5,
        matching_mode: str = "dual_softmax",
        covis_hidden: int = 128,
        covis_init_logit: float = 4.0,
        # N1: Jacobian
        jacobian_enabled: bool = True,
        jacobian_alpha_init: float = 1.0,
        jacobian_alpha_learnable: bool = True,
        # SMG: geometry-augmented MLP input for score-bias head
        smg_lat_freqs: list | None = None,
        smg_include_log_cos: bool = False,
        # N2: RoPE
        rope_enabled: bool = True,
        rope_gain_random_range=None,   # (lo, hi) -> RoPE gain ~ U[lo,hi] while training; None = off
        rope_gain_eval: float = 1.0,   # deterministic gain at eval time
        rope_H: int = 32,
        rope_W: int = 64,
        rope_lon_int_freq_max: int | None = None,
        rope_lat_theta_base: float = 10000.0,
        rope_lon_geometric: bool = False,   # ablation: geometric (standard RoPE) longitude freqs
        rope_wigner_enabled: bool = False,  # S²-RoPE: Wigner-D SO(3) steerable positional encoding
        # N2b: latitude-adaptive band-limited (anti-aliased) RoPE
        rope_bandlimit_enabled: bool = False,
        rope_bandlimit_scale: float = 1.3,
        rope_bandlimit_temp: float = 2.0,
        # N2c: SO(3)-steerable RoPE (rotate ray-organized feature triplets)
        rope_steerable_enabled: bool = False,
        # @1 SHPA (legacy — replaced by superior σFP)
        sh_pe_enabled: bool = False,
        sh_pe_L_max: int = 4,
        # @1b σFP — Sphere-preserving Factored Fourier PE (fix for SHPA)
        sfp_pe_enabled: bool = False,
        sfp_lon_freqs: list | None = None,     # integer freqs for cos/sin(k·lon)
        sfp_lat_degree: int = 4,               # Chebyshev degree on sin(lat)
        sfp_mode: str = "residual",            # "residual" (v011) | "replace" (v012)
        sfp_init_scale: float = 0.5,           # target output std for replace mode (Kaiming-like)
        # @2 GDAB
        geodesic_bias_enabled: bool = False,
        geodesic_beta_init: float = 0.1,
        # @3 LAAT
        lat_temp_enabled: bool = False,
        lat_gamma_init: float = 0.5,
        # @4 SMK — Spherical Match Kernel (multiplicative Gaussian on similarity S)
        smk_enabled: bool = False,
        smk_sigma_init: float = 1.0,           # kernel width in radians (d_geo ∈ [0, π])
        smk_sigma_learnable: bool = True,
        # @5 Yaw-Cyclic aux loss (training-time only)
        yaw_cyclic_enabled: bool = False,
        yaw_cyclic_lambda: float = 0.05,       # aux loss weight (v011: 0.1 → 0.05)
        yaw_cyclic_warmup_steps: int = 2000,   # v011: delay aux loss until main task stabilizes
        # @6 Cycle-Consistency Covisibility (replaces learned covis MLP)
        cycle_covis_enabled: bool = False,     # if True, μ derived from P_AB · P_BA diagonals
        cycle_covis_jacobian_weight: bool = True,  # multiply cycle μ by cos(φ)^α
        cycle_covis_alpha_init: float = 1.0,
        cycle_covis_alpha_learnable: bool = True,
        # @7 SMA — Sphere-Metric Attention with per-query temperature (replaces covis gate entirely)
        sma_enabled: bool = False,
        sma_temp_hidden: int = 128,            # MLP hidden dim for content-sharpness
        sma_alpha_init: float = 1.0,           # volume-form (cos^α) coefficient
        sma_alpha_learnable: bool = True,
        # @8 MMM — Metric-Only Matcher: NO learned covis module, NO content MLP.
        #   μ = cos(φ)^α with α the only learnable parameter in the covis path.
        #   Dual-softmax gate with pure geometric μ, CoMatch-interface borrowed.
        mmm_enabled: bool = False,
        mmm_alpha_init: float = 1.0,
        mmm_alpha_learnable: bool = True,
        # ★ Tangent-Relative Cross-Attention — ERP-native SCM core contribution.
        #   For every (q, k) pair, attention bias is computed from the log-map
        #   of k's ray onto q's tangent frame. Replaces RoPE's *global* phase
        #   with *query-local* sphere geometry. Compatible with rope_enabled
        #   (additive mode) or alone (replace mode).
        tangent_attn_enabled: bool = False,
        tangent_attn_n_freqs: int = 8,
        tangent_attn_hidden_dim: int = 32,
        tangent_attn_zero_init: bool = True,
        # ── EDM PE (Plan B re-implementation baseline) ─────────────────
        # EDM's 3D Cartesian positional embedding (CVPR'25, Eq. 1 + 9):
        #   chi = cos(W * S + b)  with S = (sin lam cos phi, sin phi, cos lam cos phi).
        # Absolute input-side PE added to per-token features before the attention
        # cascade. Identity-init at step 0 (W zero-init). Disjoint from RoPE/TPB;
        # used to isolate EDM's PE concept in our R1 scaffold.
        edm_pe_enabled: bool = False,
        edm_pe_zero_init: bool = True,
        # E6 — Conformal Tangent (extension of Tangent CA: cos(φ_q)^α scaling)
        tangent_attn_conformal_enabled: bool = False,
        tangent_attn_conformal_alpha_init: float = 0.0,
        # S²-TPB — Spherical Steerable Tangent-Pairwise Bias (Peter-Weyl / Wigner-D).
        #   When enabled, the (N,N) pairwise bias is a learnable linear combination
        #   of real Wigner-D matrix elements of the relative rotation R_i^{-1}R_j,
        #   replacing TangentPE2D's tangent-Fourier MLP (its flat-limit). Same
        #   bias_NN/bias_2N interface → no forward change. Zero-init → plain attn.
        tangent_attn_wigner_enabled: bool = False,
        tangent_attn_wigner_lmax: int = 2,
        # R3-T — Geodesic/Tangent Phase Attention (content-dependent TPB replacement)
        tangent_phase_enabled: bool = False,
        tangent_phase_n_pairs: int = 4,
        tangent_phase_gate_init: float = 0.0,
        tangent_phase_learnable_freqs: bool = False,
        tangent_phase_conformal_enabled: bool = False,
        tangent_phase_conformal_alpha_init: float = 0.0,
        # Geodesic kernel (radial component of spherical relative position)
        geo_kernel_enabled: bool = False,
        geo_kernel_L_max: int = 6,
        # Sphere-uniform (equal-area latitude) substrate for the attention cascade
        sphere_uniform_enabled: bool = False,
        sphere_uniform_lam_init: float = 0.0,
        sphere_uniform_lam_learnable: bool = True,
        sphere_uniform_rope_coords: bool = False,
        # ── SLTP — Spherical Latent Transport Pyramid ──
        #   Builds a low-res equal-area latent transport plan and injects it as a
        #   coarse-to-fine log-prior on the high-res pair score matrix.
        sltp_enabled: bool = False,
        sltp_downsample: int = 2,
        sltp_gamma_init: float = 0.0,
        sltp_gamma_max: float = 2.0,
        sltp_detach_prior: bool = True,
        sltp_use_equal_area: bool = True,
        # Global rotation-aligned coarse matching (full-coverage prior)
        rot_align_enabled: bool = False,
        rot_align_lam_init: float = 0.0,
        rot_align_detach: bool = True,
        # Area-marginal OT — use sphere area as matching measure, not logit bias.
        area_ot_enabled: bool = False,
        area_ot_alpha: float = 1.0,
        area_ot_iters: int = 5,
        area_ot_use_covis: bool = False,
        cache_topk_enabled: bool = False,
        cache_topk_k: int = 4,
        # B3 — gated great-circle epipolar attention bias (pose-conditioned)
        epipolar_enabled: bool = False,
        epipolar_sigma: float = 0.3,
        # A4 — gated parallel-transport of tokens to canonical tangent frame
        pt_transport_enabled: bool = False,
        # D1 — gated SO(2)-covariant attention parallel branch
        so2_cov_enabled: bool = False,
        # ★ FMTA α2 — Cyclic-W relative position bias (seam-cyclic)
        #   Learnable additive attention bias per head with longitude (W)
        #   axis indexed by mod-W. Self-attention sub-blocks (A↔A, B↔B) only.
        seam_cyclic_enabled: bool = False,
        # ★ FMTA α1 — Pole-pool reduction (architectural extension)
        #   Reduces top/bottom pole rows of (H, W) grid to n_super super-tokens.
        #   Applied AFTER attention block cascade, BEFORE similarity.
        pole_pool_enabled: bool = False,
        pole_pool_n_super: int = 1,
        # ★ GAC — Geodesic-Aware Consistency (SCG = LAC + GAC)
        #   Two-pass coarse matching: Pass 1 with LAC-only μ → soft-warp →
        #   sphere geodesic cycle distance d_geo(p, p') → γ·d_geo added
        #   (subtracted) from pre-sigmoid logit → Pass 2 with LAC+GAC μ.
        #   Pass 1 outputs are detached so gradient flows only through
        #   Pass 2's final logit (avoids 2-stage softmax vanishing).
        #   γ_GAC init=0.0 → identity at start, safe warmup automatic.
        gac_enabled: bool = False,
        gac_gamma_init: float = 0.0,
        gac_gamma_learnable: bool = True,
        # ── MGAC — Multiplicative Geodesic-Aware Covisibility (structural variant) ──
        #   3 structural changes vs GAC:
        #     (1) init via inverse-softplus → softplus(raw)≈mgac_gamma_init exactly
        #         (GAC's softplus(0)=0.693 ≠ identity is fixed here)
        #     (2) d_geo form "1-cos" instead of arccos → smooth gradient near
        #         consistency (arccos derivative 1/√(1-x²) explodes near x=1)
        #     (3) multiplicative gate: μ_final = μ_base · exp(-γ·d_geo)
        #         (vs GAC's additive logit shift: μ = σ(logit - γ·d_geo))
        #         → bounded suppression in [0, 1], larger gradient dynamic range
        mgac_enabled: bool = False,
        mgac_gamma_init: float = 0.0,
        mgac_gamma_learnable: bool = True,
        mgac_d_geo_form: str = "1-cos",  # "1-cos" or "arccos"
        # ── GAC-px — Per-pixel learned γ via small MLP ──
        #   Diagnosed bottleneck of GAC: single γ scalar gets *sparse* gradient
        #   (only 1 param updates from cycle signal). GAC-px replaces the scalar
        #   with γ(p) = softplus(MLP(token_features[p])), giving dense gradient
        #   to all features. Same gate logic (additive logit shift), only γ is
        #   feature-adaptive.
        #
        #   logit_shift(p) = -γ(p) · d_geo(p)
        #   γ(p)           = softplus(MLP(x_tok[p]))    # per-pixel from features
        #
        #   Init: MLP last layer weight=0, bias=-10 → γ ≈ 4.5e-5 ≈ 0 (identity).
        #   d_geo form: "1-cos" (smooth gradient, same as MGAC).
        gacpx_enabled: bool = False,
        gacpx_hidden_dim: int = 64,
        gacpx_d_geo_form: str = "1-cos",  # "1-cos" or "arccos"
        # ── GAC-v2 — engineering revision of GAC ──
        #   Integrated fixes over exp011 GAC:
        #     (a) γ init via inverse-softplus  → true identity at γ_init=0
        #         (vs original GAC's softplus(0)=0.693 ≠ identity)
        #     (b) γ linear warmup over `warmup_steps` opt steps
        #         (skips cycle penalty while features mature)
        #     (c) γ upper-bound via tanh squash to gamma_max
        #         (prevents over-suppression once γ grows)
        #     (d) Pass 1 with separate (higher) temperature → smoother P_p1
        #         (less noise in d_geo from over-sharp Pass 1)
        #     (e) d_geo = 1-cos by default (smooth gradient, same as MGAC)
        #     (f) Diagnostics stored as attributes for wandb logging
        #         (_last_gacv2_gamma, _d_geo_mean, _d_geo_max, _warmup_factor)
        gacv2_enabled: bool = False,
        gacv2_gamma_init: float = 0.0,        # target softplus output at init
        gacv2_gamma_max: float = 4.0,         # tanh-squash upper bound for γ
        gacv2_gamma_learnable: bool = True,
        gacv2_warmup_steps: int = 30000,      # linear warmup over this many opt steps
        gacv2_pass1_temp_scale: float = 2.0,  # Pass1 temp = self.temp * this  (smoother)
        gacv2_d_geo_form: str = "1-cos",      # "1-cos" or "arccos"
        # ── GAC-fast — saturation-free γ parameterization ──
        #   Fixes GAC-v2's gradient saturation problem:
        #   - GAC-v2 used softplus(γ_raw) with init γ_raw=-10 → grad = sigmoid(-10) ≈ 4.5e-5
        #     → γ_raw barely moves during training (saturation)
        #   - GAC-fast uses clamp(γ_raw, 0, γ_max) directly → grad = 1 everywhere in [0, γ_max]
        #     init γ_raw = 0 → γ = 0 (true identity) AND full gradient
        #   All other GAC-v2 fixes retained (Pass1 temp, 1-cos d_geo, linear warmup, diagnostics).
        gacfast_enabled: bool = False,
        gacfast_gamma_init: float = 0.0,
        gacfast_gamma_max: float = 4.0,
        gacfast_gamma_learnable: bool = True,
        gacfast_warmup_steps: int = 30000,
        gacfast_pass1_temp_scale: float = 2.0,
        gacfast_d_geo_form: str = "1-cos",
        # ── ALAC — Adaptive LAC (feature-modulated α(x)) ──
        alac_enabled: bool = False,
        alac_hidden: int = 64,
        alac_alpha_init: float = 1.0,
        # ── GAC-fine — cycle gate applied at refiner stage ──
        #   Coarse matcher remains identical to R3 (no gate change at coarse).
        #   d_geo is computed here from the final coarse P and stored as a
        #   detached grid (B, H_coarse, W_coarse) on the matcher.
        #   The Decoder reads it, upsamples to each refiner scale, and applies
        #   -γ_fine[scale] · d_geo as gate on delta_certainty.
        gacfine_enabled: bool = False,
        gacfine_d_geo_form: str = "1-cos",
        # ── BSC — Bidirectional Symmetric Covisibility ──
        #   Reformulates μ as a pair-wise property. After the per-view heads
        #   produce μ_A^(0), μ_B^(0), we refine them with each partner's
        #   covisibility evidence transported via the raw similarity P_raw:
        #     ℓ_A^(1)(p) = ℓ_A^(0)(p) + β · log( Σ_q P_AB(p,q) · μ_B^(0)(q) + ε )
        #     ℓ_B^(1)(q) = ℓ_B^(0)(q) + β · log( Σ_p P_BA(p,q) · μ_A^(0)(p) + ε )
        #   Identity at β=0: bsc_beta_init=0 zero-inits the scalar so the
        #   forward pass is bit-equivalent to the head's output until learned.
        bsc_enabled: bool = False,
        bsc_beta_init: float = 0.0,             # scalar, zero-init → identity
        bsc_beta_learnable: bool = True,
        bsc_partner_detach: bool = True,        # detach μ_partner to break feedback
        bsc_eps: float = 1e-6,
        # ── SHC — Spherical Harmonic Covisibility ──
        #   Replaces the per-pixel covis head with a bandlimited SH expansion.
        #   Predicts SH coefficients c ∈ R^K from the globally-pooled feature,
        #   reconstructs μ via Y @ c. LAC remains complementary on top of ℓ_SH.
        #   K = (L_max + 1)². Identity at init: only DC coefficient set.
        shc_enabled: bool = False,
        shc_L_max: int = 8,
        shc_hidden: int = 128,
        # ── DAC — Depth-Aware Area Correction (pass-through to head) ──
        dac_enabled: bool = False,
        dac_hidden: int = 64,
        dac_beta_init: float = 0.0,
        dac_beta_learnable: bool = True,
        # ── HAAC — Harmonic Area-Aware Correction (pass-through to head) ──
        haac_enabled: bool = False,
        haac_K: int = 4,
        haac_learnable: bool = True,
        # ── SPHM-softmax — Spherical-Prior Harmonic Matching (softmax side) ──
        # Adds a learnable latitude-spectral prior g_s(φ) to BOTH directions of
        # dual-softmax: row-softmax candidate j gets g_s(φ_B[j]); col-softmax
        # candidate i gets g_s(φ_A[i]). Identity-init γ₀=c_k=d_k=0 → R3 dual-softmax.
        #     g_s(φ) = γ₀ · log cos(φ) + Σ_{k=1..K} (c_k cos kφ + d_k sin kφ)
        # γ₀ free scalar (unlike HAAC's α₀ which is softplus-positive); allowed
        # to be negative if the data suggests "down-weight equator pixels".
        sphm_softmax_enabled: bool = False,
        sphm_softmax_K: int = 4,
        sphm_softmax_learnable: bool = True,
        # ── GSM — Geodesic Score Mixing ──
        # Bilateral sphere-geodesic smoothing of S BEFORE dual-softmax:
        #   S' = (1−α) S + α (K_geo @ S @ K_geo^T)
        #   K_geo[i,i'] = softmax(−d_geo(p_i,p_i')² / (2σ²)) on i'-axis
        # Identity-init: sigmoid(α_raw) ≈ 0 → S' = S → R3 bit-equivalent.
        # σ = softplus(σ_raw), init ≈ 0.05 rad (~3°). Both learnable.
        # ERP-native: uses geodesic distance d_geo on the sphere, not chart distance.
        gsm_enabled: bool = False,
        gsm_alpha_init: float = 0.0,          # target sigmoid output at init (0 → identity)
        gsm_sigma_init: float = 0.05,         # bandwidth in radians at init
        gsm_learnable: bool = True,
        # ── SPGM — Sphere-Parametric Generative Matching (K-mixture) ──
        # Paradigm shift: matcher does NOT compute similarity S, does NOT softmax over chart.
        # Each query i emits a K-mixture of sphere Gaussians; P[i,j] = w_i · Σ_k π_k · N(log_μ_k(p_j) | 0, Σ_k).
        spgm_enabled: bool = False,
        spgm_K: int = 4,
        spgm_hidden: int = 256,
        spgm_cross_heads: int = 4,
        spgm_w_init_logit: float = 0.0,
        # ── SCCT — Spherical Cycle-Consistent Transport ──
        #   Pairwise plan reweighting after SCCM:
        #     P'(i,j) ∝ P(i,j) · exp(-γ · d_cycle(i,j))
        #   Unlike GAC, this is not a per-pixel covis logit correction; it changes
        #   the pairwise transport operator itself.
        scct_enabled: bool = False,
        scct_gamma_init: float = 0.0,
        scct_gamma_max: float = 4.0,
        scct_detach_cycle: bool = True,
        # ── PCC — Pose-Conditioned Covisibility ──
        #   Predicts a relative-translation direction t̂ ∈ S² from the pooled
        #   features of both views, then adds a per-pixel score
        #     γ · ray(p) · t̂       (sphere-aware ray-to-direction alignment)
        #   to the covisibility logit. γ scalar zero-init → identity at start.
        #   Sphere-native (no ERP-chart constants), pair-wise (uses both A & B).
        pcc_enabled: bool = False,
        pcc_hidden: int = 128,
        pcc_gamma_init: float = 0.0,
        pcc_gamma_learnable: bool = True,
        pcc_per_view_alignment: bool = True,  # True: t̂_A and t̂_B separately
        # Accept & ignore legacy UOT kwargs (baseline config compatibility)
        uot_lambda1: float = 1.0, uot_lambda2: float = 1.0,
        uot_epsilon: float = 0.05, uot_n_iter: int = 10,
    ) -> None:
        super().__init__(
            in_dim=in_dim, dim=dim,
            n_cross_blocks=n_cross_blocks, n_heads=n_heads, ffn_ratio=ffn_ratio,
            temp=temp, sinkhorn_iters=sinkhorn_iters, matching_mode=matching_mode,
        )

        self.rope_enabled = bool(rope_enabled)
        self.sh_pe_enabled = bool(sh_pe_enabled)
        self.sfp_pe_enabled = bool(sfp_pe_enabled)
        self.gdab_enabled = bool(geodesic_bias_enabled)
        self.laat_enabled = bool(lat_temp_enabled)
        self.smk_enabled = bool(smk_enabled)
        self.yaw_cyclic_enabled = bool(yaw_cyclic_enabled)
        self.yaw_cyclic_lambda = float(yaw_cyclic_lambda)
        self.yaw_cyclic_warmup_steps = int(yaw_cyclic_warmup_steps)
        self.cycle_covis_enabled = bool(cycle_covis_enabled)
        self.cycle_covis_jacobian_weight = bool(cycle_covis_jacobian_weight)
        self.sma_enabled = bool(sma_enabled)
        self.mmm_enabled = bool(mmm_enabled)
        _covis_flags = [self.sma_enabled, self.cycle_covis_enabled, self.mmm_enabled]
        if sum(_covis_flags) > 1:
            raise ValueError("sma / cycle_covis / mmm are mutually exclusive")
        if self.sh_pe_enabled and self.sfp_pe_enabled:
            raise ValueError("Enable either sh_pe or sfp_pe, not both (they replace Fourier PE)")

        head_dim = dim // n_heads

        # ── N2: RoPE ──
        self.rot_align_enabled = bool(rot_align_enabled)
        self.rot_align_detach = bool(rot_align_detach)
        if self.rope_enabled:
            if bool(rope_wigner_enabled):
                from sccm.models.rope_wigner import RoPE2DWigner
                self.rope = RoPE2DWigner(head_dim=head_dim, H=rope_H, W=rope_W)
            elif self.rot_align_enabled:
                from sccm.models.rope_rotatable import RoPE2DRotatable
                self.rope = RoPE2DRotatable(
                    head_dim=head_dim, H=rope_H, W=rope_W,
                    lon_int_freq_max=rope_lon_int_freq_max,
                    lat_theta_base=rope_lat_theta_base,
                )
            elif bool(rope_steerable_enabled):
                from sccm.models.rope_spherical import RoPE2DSteerable
                self.rope = RoPE2DSteerable(
                    head_dim=head_dim, H=rope_H, W=rope_W,
                    lon_int_freq_max=rope_lon_int_freq_max,
                    lat_theta_base=rope_lat_theta_base,
                )
            elif bool(rope_bandlimit_enabled):
                from sccm.models.rope_spherical import RoPE2DBandLimited
                self.rope = RoPE2DBandLimited(
                    head_dim=head_dim, H=rope_H, W=rope_W,
                    lon_int_freq_max=rope_lon_int_freq_max,
                    lat_theta_base=rope_lat_theta_base,
                    bandlimit_scale=rope_bandlimit_scale,
                    bandlimit_temp=rope_bandlimit_temp,
                )
            elif rope_gain_random_range is not None:
                # RoPE strength randomised during training so it stays adjustable
                # at inference (see sccm/models/rope_circular_gain.py).
                from sccm.models.rope_circular_gain import RoPE2DCircularGain
                self.rope = RoPE2DCircularGain(
                    head_dim=head_dim, H=rope_H, W=rope_W,
                    lon_int_freq_max=rope_lon_int_freq_max,
                    lat_theta_base=rope_lat_theta_base,
                    equal_area_latitudes=bool(sphere_uniform_enabled and sphere_uniform_rope_coords),
                    lon_geometric=bool(rope_lon_geometric),
                    gain_random_range=tuple(rope_gain_random_range),
                    eval_gain=float(rope_gain_eval),
                )
                # One gain per model forward (not per apply() call).
                # NOTE: a forward-pre-hook that returns non-None replaces the
                # forward arguments, so this must return None explicitly.
                def _resample_rope_gain(_module, _inputs):
                    _module.rope.resample_gain()
                    return None
                self.register_forward_pre_hook(_resample_rope_gain)
            else:
                self.rope = RoPE2DCircular(
                    head_dim=head_dim, H=rope_H, W=rope_W,
                    lon_int_freq_max=rope_lon_int_freq_max,
                    lat_theta_base=rope_lat_theta_base,
                    equal_area_latitudes=bool(sphere_uniform_enabled and sphere_uniform_rope_coords),
                    lon_geometric=bool(rope_lon_geometric),
                )
        else:
            self.rope = None

        # ── EDM PE (Plan B re-implementation baseline) ──
        self.edm_pe_enabled = bool(edm_pe_enabled)
        if self.edm_pe_enabled:
            from sccm.models.edm_pe import EdmCartesianPE
            self.edm_pe = EdmCartesianPE(
                H=int(rope_H), W=int(rope_W),
                d_model=int(dim),
                zero_init=bool(edm_pe_zero_init),
            )
        else:
            self.edm_pe = None

        # ── ★ Tangent-Relative Cross-Attention PE ──
        # Reuses RoPE's (rope_H, rope_W) since the feature grid is the same.
        self.tangent_attn_enabled = bool(tangent_attn_enabled)
        if self.tangent_attn_enabled:
            if bool(tangent_attn_wigner_enabled):
                # S²-TPB: Peter-Weyl Wigner-D pairwise bias (same interface).
                from sccm.models.tangent_attention import TangentWignerBias
                self.tangent_pe = TangentWignerBias(
                    H=int(rope_H), W=int(rope_W),
                    lmax=int(tangent_attn_wigner_lmax),
                )
            else:
                from sccm.models.tangent_attention import TangentPE2D
                self.tangent_pe = TangentPE2D(
                    H=int(rope_H), W=int(rope_W),
                    n_freqs=int(tangent_attn_n_freqs),
                    hidden_dim=int(tangent_attn_hidden_dim),
                    zero_init=bool(tangent_attn_zero_init),
                    conformal_enabled=bool(tangent_attn_conformal_enabled),
                    conformal_alpha_init=float(tangent_attn_conformal_alpha_init),
                )
        else:
            self.tangent_pe = None

        # R3-T tangent phase attention (content-dependent; gate=0 -> no-op)
        self.tangent_phase_enabled = bool(tangent_phase_enabled)
        if self.tangent_phase_enabled:
            from sccm.models.tangent_phase_attention import TangentPhaseAttention
            self.tangent_phase = TangentPhaseAttention(
                H=int(rope_H), W=int(rope_W),
                n_phase_pairs=int(tangent_phase_n_pairs),
                gate_init=float(tangent_phase_gate_init),
                learnable_freqs=bool(tangent_phase_learnable_freqs),
                conformal_enabled=bool(tangent_phase_conformal_enabled),
                conformal_alpha_init=float(tangent_phase_conformal_alpha_init),
            )
        else:
            self.tangent_phase = None

        # ── Geodesic kernel: radial component of the spherical relative position
        #    (pairs with yaw-rope azimuth). Learnable zonal kernel of n_i·n_j,
        #    coeff init 0 → bias 0 → bit-equivalent to plain yaw-rope. ──
        self.geo_kernel_enabled = bool(geo_kernel_enabled)
        if self.geo_kernel_enabled:
            from sccm.models.geodesic_kernel import GeodesicKernel
            self.geo_kernel = GeodesicKernel(H=int(rope_H), W=int(rope_W),
                                             L_max=int(geo_kernel_L_max))
        else:
            self.geo_kernel = None

        # ── Sphere-uniform substrate: resample attention tokens to equal-area
        #    (sin phi-uniform) latitudes so cos(phi) distortion is eliminated, not
        #    corrected. Gate lam init 0 → identity → bit-equivalent at init. ──
        self.sphere_uniform_enabled = bool(sphere_uniform_enabled)
        if self.sphere_uniform_enabled:
            from sccm.models.sphere_uniform import SphereUniformAdapter
            self.sphere_uniform = SphereUniformAdapter(
                H=int(rope_H),
                W=int(rope_W),
                lam_init=float(sphere_uniform_lam_init),
                lam_learnable=bool(sphere_uniform_lam_learnable),
            )
        else:
            self.sphere_uniform = None

        self.sltp_enabled = bool(sltp_enabled)
        self.sltp_downsample = int(sltp_downsample)
        self.sltp_gamma_max = float(sltp_gamma_max)
        self.sltp_detach_prior = bool(sltp_detach_prior)
        self.sltp_use_equal_area = bool(sltp_use_equal_area)
        if self.sltp_enabled:
            self.sltp_gamma_raw = nn.Parameter(torch.tensor(float(sltp_gamma_init), dtype=torch.float32))
        else:
            self.sltp_gamma_raw = None
        self._last_sltp_gamma = 0.0
        self._last_sltp_prior_std = 0.0

        # Rotation-aligned matching: lam gate (init 0 → identity) + grid rays for Wahba.
        if self.rot_align_enabled:
            self.rot_align_lam = nn.Parameter(torch.tensor(float(rot_align_lam_init)))
            from sccm.utils.utils_sphere import erp_normalized_to_ray as _e2r
            _v = torch.linspace(-1.0 + 1.0 / int(rope_H), 1.0 - 1.0 / int(rope_H), int(rope_H))
            _u = torch.linspace(-1.0 + 1.0 / int(rope_W), 1.0 - 1.0 / int(rope_W), int(rope_W))
            _vv, _uu = torch.meshgrid(_v, _u, indexing="ij")
            self.register_buffer("_rot_align_rays",
                                 _e2r(torch.stack([_uu, _vv], dim=-1).reshape(-1, 2)),
                                 persistent=False)

        self.area_ot_enabled = bool(area_ot_enabled)
        self.area_ot_alpha = float(area_ot_alpha)
        self.area_ot_iters = int(area_ot_iters)
        self.area_ot_use_covis = bool(area_ot_use_covis)
        self.cache_topk_enabled = bool(cache_topk_enabled)
        self.cache_topk_k = int(cache_topk_k)
        self._last_topk_coords = None

        # ── Structural-attention extensions (gated, identity-init → SCCM) ──
        self.epipolar_enabled = bool(epipolar_enabled)
        if self.epipolar_enabled:
            from sccm.models.structural_attention_ext import EpipolarBiasGated
            self.epipolar = EpipolarBiasGated(feat_dim=int(dim), H=int(rope_H),
                                              W=int(rope_W), sigma=float(epipolar_sigma))
        else:
            self.epipolar = None

        self.pt_transport_enabled = bool(pt_transport_enabled)
        if self.pt_transport_enabled:
            from sccm.models.structural_attention_ext import ParallelTransportGated
            self.pt_transport = ParallelTransportGated(feat_dim=int(dim), H=int(rope_H), W=int(rope_W))
        else:
            self.pt_transport = None

        self.so2_cov_enabled = bool(so2_cov_enabled)
        if self.so2_cov_enabled:
            from sccm.models.structural_attention_ext import SO2CovariantGated
            self.so2_cov = SO2CovariantGated(feat_dim=int(dim))
        else:
            self.so2_cov = None

        # ── ★ FMTA α2 — Cyclic-W Relative Position Bias ──
        self.seam_cyclic_enabled = bool(seam_cyclic_enabled)
        if self.seam_cyclic_enabled:
            from sccm.models.erp_axes import CyclicWRelPosBias
            self.cyclic_bias = CyclicWRelPosBias(H=int(rope_H), W=int(rope_W), n_heads=int(n_heads))
        else:
            self.cyclic_bias = None

        # ── ★ FMTA α1 — Pole-pool reduction ──
        self.pole_pool_enabled = bool(pole_pool_enabled)
        self.pole_pool_n_super = int(pole_pool_n_super)
        self._pole_H = int(rope_H)
        self._pole_W = int(rope_W)
        if self.pole_pool_enabled:
            from sccm.models.erp_axes import PolePoolReduction
            self.pole_pool = PolePoolReduction(
                H=int(rope_H), W=int(rope_W),
                dim=int(dim), n_super=int(pole_pool_n_super),
            )
        else:
            self.pole_pool = None

        # ── Attention blocks (RoPE + optional GDAB + LAAT) ──
        self.cross_blocks = nn.ModuleList([
            SphereAttnCrossBlock(
                dim, n_heads=n_heads, ffn_ratio=ffn_ratio,
                rope=self.rope,
                geodesic_bias=self.gdab_enabled, geodesic_beta_init=geodesic_beta_init,
                lat_adaptive_temp=self.laat_enabled, lat_gamma_init=lat_gamma_init,
            ) for _ in range(n_cross_blocks)
        ])
        self.self_blocks = nn.ModuleList([
            SphereAttnSelfBlock(
                dim, n_heads=n_heads, ffn_ratio=ffn_ratio,
                rope=self.rope,
                geodesic_bias=self.gdab_enabled, geodesic_beta_init=geodesic_beta_init,
                lat_adaptive_temp=self.laat_enabled, lat_gamma_init=lat_gamma_init,
            ) for _ in range(n_cross_blocks)
        ])

        # ── N1 + SMG: SphereCovisHead (or SHCovisHead when shc_enabled) ──
        self.shc_enabled = bool(shc_enabled)
        if self.shc_enabled and bool(haac_enabled):
            raise ValueError("shc_enabled and haac_enabled are mutually exclusive (SHC replaces the LAC-based head)")
        if self.shc_enabled:
            from sccm.models.sphere_covis_shc import SHCovisHead
            self.shc_L_max = int(shc_L_max)
            _shc_kwargs = dict(
                dim=dim, L_max=int(shc_L_max), hidden=int(shc_hidden),
                init_logit=covis_init_logit,
                alpha_init=jacobian_alpha_init, alpha_learnable=jacobian_alpha_learnable,
                jacobian_enabled=jacobian_enabled,
            )
            self.covis_head_A = SHCovisHead(**_shc_kwargs)
            self.covis_head_B = SHCovisHead(**_shc_kwargs)
            # Per-(H,W) SH basis cache: key = (H, W, device) → (N, K)
            self._sh_basis_cache: dict = {}
        else:
            self.shc_L_max = 0
            _head_kwargs = dict(
                dim=dim, hidden=covis_hidden, init_logit=covis_init_logit,
                alpha_init=jacobian_alpha_init, alpha_learnable=jacobian_alpha_learnable,
                jacobian_enabled=jacobian_enabled,
                smg_lat_freqs=smg_lat_freqs,
                smg_include_log_cos=smg_include_log_cos,
                alac_enabled=alac_enabled,
                alac_hidden=alac_hidden,
                alac_alpha_init=alac_alpha_init,
                dac_enabled=dac_enabled,
                dac_hidden=dac_hidden,
                dac_beta_init=dac_beta_init,
                dac_beta_learnable=dac_beta_learnable,
                haac_enabled=haac_enabled,
                haac_K=haac_K,
                haac_learnable=haac_learnable,
            )
            self.covis_head_A = SphereCovisHead(**_head_kwargs)
            self.covis_head_B = SphereCovisHead(**_head_kwargs)
            self._sh_basis_cache = {}

        # ── PCC — Pose-Conditioned Covisibility ──
        # Pose head: pooled (A, B) features → translation-direction t̂ ∈ S².
        # Per-pixel logit add-on: γ · (ray_p · t̂).
        self.pcc_enabled = bool(pcc_enabled)
        self.pcc_per_view_alignment = bool(pcc_per_view_alignment)
        if self.pcc_enabled:
            self.pcc_pose_mlp = nn.Sequential(
                nn.Linear(dim * 2, pcc_hidden),
                nn.GELU(),
                nn.Linear(pcc_hidden, 3),
            )
            # Zero last-layer → t̂ ≡ 0; combined with γ=0 → identity at init.
            nn.init.zeros_(self.pcc_pose_mlp[-1].weight)
            nn.init.zeros_(self.pcc_pose_mlp[-1].bias)
            t = torch.tensor(float(pcc_gamma_init), dtype=torch.float32)
            if pcc_gamma_learnable:
                self.pcc_gamma = nn.Parameter(t)
            else:
                self.register_buffer('pcc_gamma', t)
        else:
            self.pcc_pose_mlp = None
            self.pcc_gamma = None
        # Diagnostics
        self._last_pcc_gamma = 0.0
        self._last_pcc_t_norm = 0.0
        self._last_pcc_align_mean = 0.0
        self._last_pcc_logit_delta_mean = 0.0

        # ── BSC — Bidirectional Symmetric Covisibility ──
        # Scalar β multiplier on the partner-evidence term log E[μ_partner].
        # Sign-free param: NO softplus; zero-init at β=0 → identity at start.
        self.bsc_enabled = bool(bsc_enabled)
        self.bsc_partner_detach = bool(bsc_partner_detach)
        self.bsc_eps = float(bsc_eps)
        if self.bsc_enabled:
            t = torch.tensor(float(bsc_beta_init), dtype=torch.float32)
            if bsc_beta_learnable:
                self.bsc_beta = nn.Parameter(t)
            else:
                self.register_buffer("bsc_beta", t)
        else:
            self.bsc_beta = None
        # Diagnostics
        self._last_bsc_beta = 0.0
        self._last_bsc_logE_mu_B_mean = 0.0
        self._last_bsc_logE_mu_A_mean = 0.0
        self._last_bsc_logit_delta_mean = 0.0

        # ── GAC — Geodesic-Aware Consistency (SCG = LAC + GAC) ──
        self.gac_enabled = bool(gac_enabled)
        if self.gac_enabled:
            t = torch.tensor(float(gac_gamma_init), dtype=torch.float32)
            if gac_gamma_learnable:
                self.gac_gamma_raw = nn.Parameter(t)
            else:
                self.register_buffer("gac_gamma_raw", t)
        else:
            self.gac_gamma_raw = None

        # ── MGAC — Multiplicative Geodesic-Aware Covisibility ──
        self.mgac_enabled = bool(mgac_enabled)
        self.mgac_d_geo_form = str(mgac_d_geo_form)
        if self.mgac_enabled:
            target_gamma = float(mgac_gamma_init)
            if target_gamma < 1e-4:
                # softplus(-10) ≈ 4.54e-5 — effectively zero, true identity
                raw_init = -10.0
            else:
                # inverse-softplus: softplus(log(expm1(t))) = t  exactly
                raw_init = math.log(math.expm1(target_gamma))
            t_raw = torch.tensor(raw_init, dtype=torch.float32)
            if mgac_gamma_learnable:
                self.mgac_gamma_raw = nn.Parameter(t_raw)
            else:
                self.register_buffer("mgac_gamma_raw", t_raw)
        else:
            self.mgac_gamma_raw = None

        # ── GAC-v2 — engineering revision ──
        self.gacv2_enabled = bool(gacv2_enabled)
        self.gacv2_gamma_max = float(gacv2_gamma_max)
        self.gacv2_warmup_steps = int(gacv2_warmup_steps)
        self.gacv2_pass1_temp_scale = float(gacv2_pass1_temp_scale)
        self.gacv2_d_geo_form = str(gacv2_d_geo_form)
        if self.gacv2_enabled:
            target_gamma = float(gacv2_gamma_init)
            if target_gamma < 1e-4:
                raw_init = -10.0   # softplus(-10) ≈ 4.5e-5 → effective 0
            else:
                raw_init = math.log(math.expm1(target_gamma))
            t_raw = torch.tensor(raw_init, dtype=torch.float32)
            if gacv2_gamma_learnable:
                self.gacv2_gamma_raw = nn.Parameter(t_raw)
            else:
                self.register_buffer("gacv2_gamma_raw", t_raw)
            # opt-step counter for linear warmup (auto-increment in forward)
            self.register_buffer("_gacv2_step",
                                  torch.tensor(0, dtype=torch.long))
        else:
            self.gacv2_gamma_raw = None
        # Diagnostic attributes (set during forward, read by training loop)
        self._last_gacv2_gamma = 0.0
        self._last_gacv2_gamma_raw = 0.0
        self._last_gacv2_d_geo_mean = 0.0
        self._last_gacv2_d_geo_max = 0.0
        self._last_gacv2_warmup_factor = 0.0

        # ── GAC-fast — saturation-free γ parameterization ──
        self.gacfast_enabled = bool(gacfast_enabled)
        self.gacfast_gamma_max = float(gacfast_gamma_max)
        self.gacfast_warmup_steps = int(gacfast_warmup_steps)
        self.gacfast_pass1_temp_scale = float(gacfast_pass1_temp_scale)
        self.gacfast_d_geo_form = str(gacfast_d_geo_form)
        if self.gacfast_enabled:
            # NO inverse-softplus needed — γ_raw maps directly to γ via clamp.
            # init γ_raw = gamma_init → γ = clamp(init, 0, gamma_max) = init (if 0 ≤ init ≤ max).
            t_raw = torch.tensor(float(gacfast_gamma_init), dtype=torch.float32)
            if gacfast_gamma_learnable:
                self.gacfast_gamma_raw = nn.Parameter(t_raw)
            else:
                self.register_buffer("gacfast_gamma_raw", t_raw)
            self.register_buffer("_gacfast_step",
                                  torch.tensor(0, dtype=torch.long))
        else:
            self.gacfast_gamma_raw = None
        # Diagnostics for wandb
        self._last_gacfast_gamma = 0.0
        self._last_gacfast_gamma_raw = 0.0
        self._last_gacfast_d_geo_mean = 0.0
        self._last_gacfast_d_geo_max = 0.0
        self._last_gacfast_warmup_factor = 0.0

        # ── GAC-fine — compute and expose d_geo grid for Decoder ──
        self.gacfine_enabled = bool(gacfine_enabled)
        self.gacfine_d_geo_form = str(gacfine_d_geo_form)
        # Will be set during forward: detached tensor (B, H_coarse, W_coarse)
        self._gacfine_d_geo_grid = None

        # ── GAC-px — Per-pixel learned γ via small MLP ──
        self.gacpx_enabled = bool(gacpx_enabled)
        self.gacpx_d_geo_form = str(gacpx_d_geo_form)
        if self.gacpx_enabled:
            # Small MLP: dim → hidden → 1.  Zero last-layer weight + bias=-10
            # → softplus(-10) ≈ 4.5e-5 ≈ 0 (true identity start, R3-equivalent).
            self.gacpx_mlp = nn.Sequential(
                nn.Linear(dim, gacpx_hidden_dim),
                nn.GELU(),
                nn.Linear(gacpx_hidden_dim, 1),
            )
            nn.init.zeros_(self.gacpx_mlp[-1].weight)
            nn.init.constant_(self.gacpx_mlp[-1].bias, -10.0)
        else:
            self.gacpx_mlp = None

        # ── @1 SHPA: Spherical Harmonic Positional Aggregation ──
        if self.sh_pe_enabled:
            if not (0 <= sh_pe_L_max <= 4):
                raise ValueError(f"sh_pe_L_max must be in [0, 4], got {sh_pe_L_max}")
            self.sh_pe_L_max = int(sh_pe_L_max)
            n_sh = (sh_pe_L_max + 1) ** 2                               # 25 at L=4
            self.sh_mix = nn.Linear(n_sh, dim, bias=True)               # learnable mix
            # Initialize: near-identity on first `min(dim, n_sh)` channels, rest small-random
            with torch.no_grad():
                self.sh_mix.weight.zero_()
                self.sh_mix.bias.zero_()
                # Tie first n_sh output channels to SH input (near-identity start)
                for i in range(min(dim, n_sh)):
                    self.sh_mix.weight[i, i] = 1.0
        else:
            self.sh_mix = None

        # ── @1b σFP: Sphere-preserving Factored Fourier PE ──
        # Basis: {1} ∪ {cos(k·lon)·T_l(sin(lat)), sin(k·lon)·T_l(sin(lat))} for k ∈ lon_freqs, l ∈ [0..lat_degree]
        # Total channels: 1 + 2 * |lon_freqs| * (lat_degree + 1)
        #
        # Two modes:
        #   "residual" (v011): pe = fourier_pe + sfp_mix(basis), sfp_mix zero-init.
        #                      Warm-start preserved, but σFP redundant subspace of
        #                      Fourier PE → mediocre gain.
        #   "replace"  (v012): pe = sfp_mix(basis) alone (NO Fourier PE). Kaiming
        #                      init scaled to match Fourier PE output std (≈0.5).
        #                      Full commitment to sphere-aware PE.
        self.sfp_mode = str(sfp_mode) if self.sfp_pe_enabled else "residual"
        if self.sfp_mode not in ("residual", "replace"):
            raise ValueError(f"sfp_mode must be 'residual' or 'replace', got {self.sfp_mode}")
        if self.sfp_pe_enabled:
            if sfp_lon_freqs is None:
                sfp_lon_freqs = [1, 2, 4, 8, 16]
            self.sfp_lon_freqs = [int(k) for k in sfp_lon_freqs]
            self.sfp_lat_degree = int(sfp_lat_degree)
            n_sfp = 1 + 2 * len(self.sfp_lon_freqs) * (self.sfp_lat_degree + 1)
            self.sfp_mix = nn.Linear(n_sfp, dim, bias=True)
            with torch.no_grad():
                if self.sfp_mode == "residual":
                    self.sfp_mix.weight.zero_()
                    self.sfp_mix.bias.zero_()
                else:  # replace
                    # Target output std ≈ sfp_init_scale (Fourier PE std is ~0.5).
                    # If weights are orthogonal with basis std ≈ 1, output std ≈ weight_std · sqrt(n_sfp).
                    # We want output_std = sfp_init_scale → weight_std = sfp_init_scale / sqrt(n_sfp).
                    weight_std = float(sfp_init_scale) / math.sqrt(n_sfp)
                    nn.init.normal_(self.sfp_mix.weight, mean=0.0, std=weight_std * math.sqrt(3.0))
                    # Gaussian scaled by √3 gives variance ≈ weight_std² (empirically matches).
                    nn.init.zeros_(self.sfp_mix.bias)
            self.sfp_n_ch = n_sfp
        else:
            self.sfp_mix = None
            self.sfp_n_ch = 0

        # ── @4 SMK: Spherical Match Kernel ──────────────────────────────
        # Multiplicative Gaussian on similarity S: S' = S · exp(-d_geo² / (2σ²))
        # σ = softplus(σ_raw), init ≈ smk_sigma_init rad. d_geo ∈ [0, π].
        if self.smk_enabled:
            s = float(smk_sigma_init)
            if s <= 0: raise ValueError("smk_sigma_init must be > 0")
            sigma_raw_init = math.log(math.expm1(s))
            t = torch.tensor(sigma_raw_init, dtype=torch.float32)
            if smk_sigma_learnable:
                self.smk_sigma_raw = nn.Parameter(t)
            else:
                self.register_buffer("smk_sigma_raw", t)
        else:
            self.smk_sigma_raw = None

        # ── @6 Cycle-Consistency Covisibility — replaces learned per-view MLP covis head ──
        # μ(p) = [P_AB · P_BA]_pp (diagonal of dual-direction assignment composition),
        # optionally multiplied by cos(φ_p)^α (volume form weighting).
        # When enabled, the learned covis heads (covis_head_A, covis_head_B) are REMOVED
        # so the model has no learned covisibility module — covisibility is intrinsic.
        if self.cycle_covis_enabled:
            # Remove learned covis heads — ICCA paper claim: no per-view learned μ
            del self.covis_head_A
            del self.covis_head_B
            self.covis_head_A = None
            self.covis_head_B = None
            if self.cycle_covis_jacobian_weight:
                a = float(cycle_covis_alpha_init)
                if a <= 0: raise ValueError("cycle_covis_alpha_init must be > 0")
                raw = math.log(math.expm1(a))
                t = torch.tensor(raw, dtype=torch.float32)
                if cycle_covis_alpha_learnable:
                    self.cycle_alpha_raw = nn.Parameter(t)
                else:
                    self.register_buffer("cycle_alpha_raw", t)
            else:
                self.cycle_alpha_raw = None
        else:
            self.cycle_alpha_raw = None

        # ── @7 SMA — Sphere-Metric Attention with per-query temperature ─────
        # Replaces covisibility gating entirely: no additive log-bias on S.
        # Instead, per-query temperature τ(p) = exp(-α·log cos(φ_p) + MLP(x_p, lat_enc))
        # scales attention softmax per-query. Flatter at poles, content-modulated.
        # Mathematically distinct from CoMatch's additive gate:
        #   CoMatch gate:  P_row(p,q) ∝ exp(S[p,q] + log μ_A(p) + log μ_B(q))
        #   SMA sharpness: P_row(p,q) ∝ exp(S[p,q] / τ_A(p))
        # The former shifts the row uniformly; the latter RESCALES it, producing
        # different distribution shape (not just translation).
        if self.sma_enabled:
            # Remove any learned covis heads — SMA has no gate
            if self.covis_head_A is not None:
                del self.covis_head_A; self.covis_head_A = None
            if self.covis_head_B is not None:
                del self.covis_head_B; self.covis_head_B = None
            h = int(sma_temp_hidden)
            # MLP: features (dim) + sin(φ), cos(φ) (2) → 1 scalar log-content-temp
            self.sma_temp_mlp = nn.Sequential(
                nn.Linear(dim + 2, h),
                nn.GELU(),
                nn.Linear(h, 1),
            )
            with torch.no_grad():
                # Small-Gaussian init on last layer weight + zero bias.
                # At init, MLP output has small std (~0.03), so τ ≈ cos^α · (1 ± 3%).
                # Close to pure volume-form warm-start, but allows gradient to flow
                # immediately through both MLP layers (full-zero would starve W_1).
                nn.init.normal_(self.sma_temp_mlp[-1].weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.sma_temp_mlp[-1].bias)
            a = float(sma_alpha_init)
            if a <= 0: raise ValueError("sma_alpha_init must be > 0")
            raw = math.log(math.expm1(a))
            t = torch.tensor(raw, dtype=torch.float32)
            if sma_alpha_learnable:
                self.sma_alpha_raw = nn.Parameter(t)
            else:
                self.register_buffer("sma_alpha_raw", t)
        else:
            self.sma_temp_mlp = None
            self.sma_alpha_raw = None

        # ── @8 MMM — Metric-Only Matcher ────────────────────────────────────
        # No learned covisibility module at all. μ = cos(φ)^α, one learnable α.
        # Dual-softmax gate: S_gated = S + α·log cos(φ_p) + α·log cos(φ_q).
        # Sphere metric fully determines the covis signal; nothing borrowed from
        # CoMatch's learned per-view MLP. Cleanest metric-only realization.
        if self.mmm_enabled:
            # Remove learned covis heads
            if self.covis_head_A is not None:
                del self.covis_head_A; self.covis_head_A = None
            if self.covis_head_B is not None:
                del self.covis_head_B; self.covis_head_B = None
            a = float(mmm_alpha_init)
            if a <= 0: raise ValueError("mmm_alpha_init must be > 0")
            raw = math.log(math.expm1(a))
            t = torch.tensor(raw, dtype=torch.float32)
            if mmm_alpha_learnable:
                self.mmm_alpha_raw = nn.Parameter(t)
            else:
                self.register_buffer("mmm_alpha_raw", t)
        else:
            self.mmm_alpha_raw = None

        # ── SPHM-softmax — sphere prior on dual-softmax ────────────────────
        self.sphm_softmax_enabled = bool(sphm_softmax_enabled)
        self.sphm_softmax_K = int(sphm_softmax_K) if self.sphm_softmax_enabled else 0
        if self.sphm_softmax_enabled:
            if self.sphm_softmax_K <= 0:
                raise ValueError(f"sphm_softmax_K must be >= 1, got {self.sphm_softmax_K}")
            self.register_buffer(
                "_sphm_k_idx",
                torch.arange(1, self.sphm_softmax_K + 1, dtype=torch.float32),
            )
            gamma0 = torch.zeros(1, dtype=torch.float32)            # γ₀ free, init 0
            c = torch.zeros(self.sphm_softmax_K, dtype=torch.float32)
            d = torch.zeros(self.sphm_softmax_K, dtype=torch.float32)
            if sphm_softmax_learnable:
                self.sphm_gamma0 = nn.Parameter(gamma0)
                self.sphm_c = nn.Parameter(c)
                self.sphm_d = nn.Parameter(d)
            else:
                self.register_buffer("sphm_gamma0", gamma0)
                self.register_buffer("sphm_c", c)
                self.register_buffer("sphm_d", d)
        else:
            self.sphm_gamma0 = None
            self.sphm_c = None
            self.sphm_d = None
        self._current_lat_N = None
        self._last_sphm_gs_mean = 0.0
        self._last_sphm_gs_std = 0.0
        self._last_sphm_gamma0 = 0.0

        # ── GSM — Geodesic Score Mixing (pre-softmax bilateral smoothing of S) ──
        self.gsm_enabled = bool(gsm_enabled)
        if self.gsm_enabled:
            a_init = float(gsm_alpha_init)
            s_init = float(gsm_sigma_init)
            if not (0.0 <= a_init < 1.0):
                raise ValueError(f"gsm_alpha_init must be in [0, 1), got {a_init}")
            if s_init <= 0:
                raise ValueError(f"gsm_sigma_init must be > 0, got {s_init}")
            # α_raw with sigmoid: target sigmoid(α_raw) = a_init → α_raw = logit(a_init)
            # For a_init=0 use −20 so sigmoid ≈ 2e-9 (machine-precision identity to R3
            # even with O(10) score magnitudes flowing through dual-softmax).
            if a_init <= 0.0:
                alpha_raw_init = -20.0
            else:
                alpha_raw_init = math.log(a_init / (1.0 - a_init))
            sigma_raw_init = math.log(math.expm1(s_init))
            t_a = torch.tensor(alpha_raw_init, dtype=torch.float32)
            t_s = torch.tensor(sigma_raw_init, dtype=torch.float32)
            if gsm_learnable:
                self.gsm_alpha_raw = nn.Parameter(t_a)
                self.gsm_sigma_raw = nn.Parameter(t_s)
            else:
                self.register_buffer("gsm_alpha_raw", t_a)
                self.register_buffer("gsm_sigma_raw", t_s)
        else:
            self.gsm_alpha_raw = None
            self.gsm_sigma_raw = None
        self._last_gsm_alpha = 0.0
        self._last_gsm_sigma = 0.0

        # ── SPGM — Sphere-Parametric Generative Matching head ──
        self.spgm_enabled = bool(spgm_enabled)
        if self.spgm_enabled:
            self.spgm_head = SphereGenerativeHead(
                dim=dim,
                K=int(spgm_K),
                hidden=int(spgm_hidden),
                cross_heads=int(spgm_cross_heads),
                w_init_logit=float(spgm_w_init_logit),
            )
        else:
            self.spgm_head = None

        self.scct_enabled = bool(scct_enabled)
        self.scct_gamma_max = float(scct_gamma_max)
        self.scct_detach_cycle = bool(scct_detach_cycle)
        if self.scct_enabled:
            self.scct_gamma_raw = nn.Parameter(torch.tensor(float(scct_gamma_init), dtype=torch.float32))
        else:
            self.scct_gamma_raw = None
        self._last_scct_gamma = 0.0
        self._last_scct_cycle_mean = 0.0

        # ── Geometry buffers (for GDAB, LAAT, SHPA, Jacobian). Computed per forward
        # based on actual (H, W) of input, but cached at config resolution for speed.
        self._geom_cache: dict = {}

    # ── Geometry helpers with caching ─────────────────────────────────
    def _geom(self, H: int, W: int, device, dtype=torch.float32):
        key = (H, W, str(device), str(dtype))
        if key in self._geom_cache:
            return self._geom_cache[key]
        lat, lon = _erp_lat_lon_grid(H, W, device, dtype=dtype)            # each (N,)
        rays = _erp_rays(H, W, device, dtype=dtype)                        # (N, 3)
        # Pairwise geodesic distance for GDAB
        dot = rays @ rays.transpose(0, 1)                                  # (N, N)
        d_geo = torch.arccos(dot.clamp(-1 + 1e-6, 1 - 1e-6))
        # SH basis for SHPA (N, n_sh)
        sh_basis = None
        if self.sh_pe_enabled:
            sh_basis = real_spherical_harmonics(rays, L_max=self.sh_pe_L_max)  # (N, n_sh)
        geom = {"lat": lat, "lon": lon, "rays": rays, "d_geo": d_geo, "sh_basis": sh_basis}
        self._geom_cache[key] = geom
        return geom

    def _geodesic_score_mix(self, S: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """GSM: bilateral sphere-geodesic smoothing of similarity matrix S.

        S: (B, N_A, N_B) with N_A = N_B = H*W (same ERP grid for both views).
        Returns S' = (1-α) S + α (K_geo @ S @ K_geo^T)
        where K_geo[i, i'] = softmax_{i'}(-d_geo² / (2σ²)) (row-stochastic on i').
        """
        if not self.gsm_enabled or self.gsm_alpha_raw is None:
            return S
        geom = self._geom(H, W, device=S.device, dtype=torch.float32)
        d_geo = geom["d_geo"].to(S.dtype)                                          # (N, N)
        sigma = F.softplus(self.gsm_sigma_raw).to(S.dtype).clamp(min=1e-4)
        # Build row-stochastic K_geo via softmax of -d² / (2σ²) along i' (row sums to 1)
        logits = -0.5 * (d_geo / sigma) ** 2                                       # (N, N)
        K_geo = F.softmax(logits, dim=-1)                                          # (N, N)
        alpha = torch.sigmoid(self.gsm_alpha_raw).to(S.dtype)
        # S' = (1-α) S + α (K_geo @ S @ K_geo^T)
        S_mix = K_geo @ S @ K_geo.transpose(-1, -2)
        out = (1.0 - alpha) * S + alpha * S_mix
        with torch.no_grad():
            self._last_gsm_alpha = float(alpha.detach().item())
            self._last_gsm_sigma = float(sigma.detach().item())
        return out

    def _apply_sltp_prior(self, S: torch.Tensor, x_tok: torch.Tensor, y_tok: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """SLTP: low-res spherical latent transport prior for the high-res plan."""
        if not self.sltp_enabled or self.sltp_gamma_raw is None:
            return S
        ds = max(1, int(self.sltp_downsample))
        if ds == 1 or H < ds or W < ds:
            return S
        H_l = H // ds
        W_l = W // ds
        H_use = H_l * ds
        W_use = W_l * ds
        if H_use != H or W_use != W:
            return S

        B, _, D = x_tok.shape
        x_map = rearrange(x_tok, "b (h w) c -> b c h w", h=H, w=W)
        y_map = rearrange(y_tok, "b (h w) c -> b c h w", h=H, w=W)
        if self.sltp_use_equal_area and self.sphere_uniform is not None:
            x_map = self.sphere_uniform.to_uniform(x_map)
            y_map = self.sphere_uniform.to_uniform(y_map)

        x_low = F.avg_pool2d(x_map, kernel_size=ds, stride=ds)
        y_low = F.avg_pool2d(y_map, kernel_size=ds, stride=ds)
        x_low = rearrange(x_low, "b c h w -> b (h w) c")
        y_low = rearrange(y_low, "b c h w -> b (h w) c")
        x_low = F.normalize(x_low, dim=-1)
        y_low = F.normalize(y_low, dim=-1)
        S_low = torch.bmm(x_low, y_low.transpose(1, 2)) / self.temp
        P_low = F.softmax(S_low, dim=-1) * F.softmax(S_low, dim=-2)
        if self.sltp_detach_prior:
            P_low = P_low.detach()

        h_idx = (torch.arange(H, device=S.device) // ds).clamp(max=H_l - 1)
        w_idx = (torch.arange(W, device=S.device) // ds).clamp(max=W_l - 1)
        low_idx = (h_idx[:, None] * W_l + w_idx[None, :]).reshape(H * W)
        log_prior = torch.log(P_low.clamp(min=1e-12))
        log_prior = log_prior[:, low_idx[:, None], low_idx[None, :]]
        log_prior = log_prior - log_prior.mean(dim=(-2, -1), keepdim=True)

        gamma = torch.clamp(self.sltp_gamma_raw, min=0.0, max=self.sltp_gamma_max).to(S.dtype)
        out = S + gamma * log_prior.to(S.dtype)
        with torch.no_grad():
            self._last_sltp_gamma = float(gamma.detach().item())
            self._last_sltp_prior_std = float(log_prior.float().std().item())
        return out

    def _apply_scct(self, P: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """SCCT: pairwise spherical cycle-consistent transport reweighting."""
        if not self.scct_enabled or self.scct_gamma_raw is None:
            return P
        B = P.shape[0]
        geom = self._geom(H, W, device=P.device, dtype=P.dtype)
        rays = geom["rays"].to(P.dtype)
        rays_b = rays.unsqueeze(0).expand(B, -1, -1).contiguous()

        P_base = P.detach() if self.scct_detach_cycle else P
        P_row = P_base / P_base.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        P_col = P_base / P_base.sum(dim=-2, keepdim=True).clamp(min=1e-8)

        exp_B_at_A = torch.bmm(P_row, rays_b)
        exp_A_at_B = torch.bmm(P_col.transpose(1, 2), rays_b)
        exp_B_at_A = F.normalize(exp_B_at_A, dim=-1)
        exp_A_at_B = F.normalize(exp_A_at_B, dim=-1)

        cos_A = torch.einsum("bik,bjk->bij", rays_b, exp_A_at_B)
        cos_B = torch.einsum("bik,jk->bij", exp_B_at_A, rays)
        cycle_cost = (2.0 - cos_A - cos_B).clamp(min=0.0)
        if self.scct_detach_cycle:
            cycle_cost = cycle_cost.detach()

        gamma = torch.clamp(self.scct_gamma_raw, min=0.0, max=self.scct_gamma_max).to(P.dtype)
        P_scct = P * torch.exp(-gamma * cycle_cost)
        row_mass = P.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        P_scct = P_scct * (row_mass / P_scct.sum(dim=-1, keepdim=True).clamp(min=1e-8))
        with torch.no_grad():
            self._last_scct_gamma = float(gamma.detach().item())
            self._last_scct_cycle_mean = float(cycle_cost.mean().item())
        return P_scct

    def _covis_dual_softmax(self, S, mu_A, mu_B):
        # GSM: bilateral sphere-geodesic smoothing of S before gating + softmax.
        # Identity-init: sigmoid(α_raw)≈0 → no change.
        if self.gsm_enabled and getattr(self, "_current_H", None) is not None:
            S = self._geodesic_score_mix(S, self._current_H, self._current_W)
        log_muA = torch.log(mu_A.clamp(min=1e-6))
        log_muB = torch.log(mu_B.clamp(min=1e-6))
        S_gated = S + log_muA.unsqueeze(-1) + log_muB.unsqueeze(-2)
        if self.sphm_softmax_enabled and self._current_lat_N is not None and self.sphm_gamma0 is not None:
            lat = self._current_lat_N.to(S.dtype)                                          # (N,)
            log_cos = torch.log(torch.cos(lat).clamp(min=_SCCM_POLE_EPS))                  # (N,)
            k_phi = lat.unsqueeze(-1) * self._sphm_k_idx.to(lat.dtype)                     # (N, K)
            g_s = (
                self.sphm_gamma0.to(lat.dtype) * log_cos
                + (torch.cos(k_phi) * self.sphm_c.to(lat.dtype)).sum(-1)
                + (torch.sin(k_phi) * self.sphm_d.to(lat.dtype)).sum(-1)
            )                                                                              # (N,)
            S_row = S_gated + g_s.view(1, 1, -1)        # row-softmax: add g_s on B-candidate axis
            S_col = S_gated + g_s.view(1, -1, 1)        # col-softmax: add g_s on A-candidate axis
            with torch.no_grad():
                self._last_sphm_gs_mean = float(g_s.abs().mean().item())
                self._last_sphm_gs_std = float(g_s.float().std().item())
                self._last_sphm_gamma0 = float(self.sphm_gamma0.detach().item())
            return F.softmax(S_row, dim=-1) * F.softmax(S_col, dim=-2)
        return F.softmax(S_gated, dim=-1) * F.softmax(S_gated, dim=-2)

    def _area_marginal_sinkhorn(self, S: torch.Tensor, mu_A=None, mu_B=None) -> torch.Tensor:
        """Sphere-area OT plan with cos(latitude)^alpha marginals.

        This makes the matching measure itself ERP-native. Optional covis factors
        can modulate the marginals, but the clean ablation keeps them off.
        """
        lat = self._current_lat_N.to(dtype=S.dtype, device=S.device)
        log_area = self.area_ot_alpha * torch.log(torch.cos(lat).clamp(min=_SCCM_POLE_EPS))
        log_a = log_area.unsqueeze(0).expand(S.shape[0], -1)
        log_b = log_area.unsqueeze(0).expand(S.shape[0], -1)
        if self.area_ot_use_covis and mu_A is not None and mu_B is not None:
            log_a = log_a + torch.log(mu_A.clamp(min=1e-6))
            log_b = log_b + torch.log(mu_B.clamp(min=1e-6))
        log_a = log_a - torch.logsumexp(log_a, dim=-1, keepdim=True)
        log_b = log_b - torch.logsumexp(log_b, dim=-1, keepdim=True)
        with torch.no_grad():
            u = torch.zeros_like(log_a)
            v = torch.zeros_like(log_b)
            S_det = S.detach()
            for _ in range(self.area_ot_iters):
                u = log_a - torch.logsumexp(S_det + v.unsqueeze(1), dim=-1)
                v = log_b - torch.logsumexp(S_det + u.unsqueeze(-1), dim=-2)
        return (S + u.detach().unsqueeze(-1) + v.detach().unsqueeze(1)).exp()

    def _get_shc_basis(self, H: int, W: int, device, dtype=torch.float32) -> torch.Tensor:
        """Return cached (N, K) real SH basis for SHCovisHead.

        Uses the matcher's own (lat, lon) convention via _geom(), so the basis
        is consistent with LAC's log cos φ. Theta (colatitude) = π/2 - lat.
        """
        from sccm.models.sphere_covis_shc import compute_real_sh_basis
        key = (H, W, str(device), int(self.shc_L_max))
        if key in self._sh_basis_cache:
            return self._sh_basis_cache[key].to(dtype=dtype)
        geom = self._geom(H, W, device, dtype=torch.float32)
        lat = geom["lat"]                                                # (N,)
        lon = geom["lon"]                                                # (N,)
        theta = (math.pi / 2.0) - lat                                    # colatitude (N,)
        basis = compute_real_sh_basis(theta, lon, self.shc_L_max)        # (N, K)
        self._sh_basis_cache[key] = basis
        return basis.to(dtype=dtype)

    def _apply_smk(self, S: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """SMK: multiply similarity S by Gaussian kernel on geodesic distance.

        S: (B, N_A, N_B) similarity (pre-softmax)
        Returns S · kernel where kernel[i,j] = exp(-d_geo(p_i, p_j)² / (2σ²)).
        σ = softplus(σ_raw), learnable. d_geo uses shared A,B grid.
        """
        geom = self._geom(H, W, device=S.device, dtype=torch.float32)
        d_geo = geom["d_geo"].to(S.dtype)                                # (N, N)
        sigma = F.softplus(self.smk_sigma_raw)
        kernel = torch.exp(-0.5 * (d_geo / sigma.clamp(min=1e-3)) ** 2)  # (N, N)
        return S * kernel.unsqueeze(0)                                   # (B, N, N)

    def _sh_pe(self, B, H, W, device):
        """SHPA: return (B, N, dim) using fixed SH basis + learnable mix."""
        geom = self._geom(H, W, device, dtype=torch.float32)
        sh = geom["sh_basis"]                                           # (N, n_sh)
        pe = self.sh_mix(sh)                                            # (N, dim)
        return pe.unsqueeze(0).expand(B, -1, -1).contiguous()           # (B, N, dim)

    def _sfp_basis(self, H: int, W: int, device, dtype=torch.float32) -> torch.Tensor:
        """σFP: factored basis {1, cos(k·lon)·T_l(sin(lat)), sin(k·lon)·T_l(sin(lat))}.
        Integer k → 2π-periodic in lon. T_l are Chebyshev polynomials on sin(lat) ∈ [-1, 1].
        Returns (N, n_ch).
        """
        geom = self._geom(H, W, device, dtype=dtype)
        lat, lon = geom["lat"], geom["lon"]                              # each (N,)
        s = torch.sin(lat)                                               # (N,) ∈ [-1, 1]
        # Chebyshev T_l(s), l = 0..lat_degree
        T = [torch.ones_like(s), s]
        for l in range(2, self.sfp_lat_degree + 1):
            T.append(2 * s * T[-1] - T[-2])
        T = torch.stack(T[:self.sfp_lat_degree + 1], dim=-1)             # (N, lat_degree+1)
        # For each lon freq k: cos(k·lon), sin(k·lon)
        chans = [torch.ones_like(s).unsqueeze(-1)]                       # constant
        for k in self.sfp_lon_freqs:
            c = torch.cos(k * lon).unsqueeze(-1)                         # (N, 1)
            si = torch.sin(k * lon).unsqueeze(-1)                        # (N, 1)
            chans.append((c * T).view(-1, T.shape[-1]))                  # (N, lat_degree+1)
            chans.append((si * T).view(-1, T.shape[-1]))
        basis = torch.cat(chans, dim=-1)                                 # (N, n_sfp)
        return basis

    def _sfp_pe(self, B, H, W, device):
        """σFP PE correction: returns (B, N, dim) — intended to be ADDED to Fourier PE.
        At init with zero weights/bias, this is a zero tensor (no-op), giving warm-start."""
        basis = self._sfp_basis(H, W, device, dtype=torch.float32)       # (N, n_sfp)
        pe = self.sfp_mix(basis)                                         # (N, dim)
        return pe.unsqueeze(0).expand(B, -1, -1).contiguous()

    # ── Forward ────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        # Precompute geometry (lat, d_geo, sh) once per (H,W,device)
        geom = self._geom(H, W, device=x.device, dtype=torch.float32)
        lat_N = geom["lat"]                                             # (N,)
        self._current_lat_N = lat_N                                     # cached for SPHM-softmax in _covis_dual_softmax
        self._current_H = H                                             # cached for GSM (needs H, W to access d_geo via _geom)
        self._current_W = W
        d_geo_NN = geom["d_geo"] if (self.gdab_enabled) else None       # (N, N) or None

        # Block kwargs only pass geometry if actually needed by blocks
        blk_kw_cross = {}
        blk_kw_self = {}
        if self.gdab_enabled:  blk_kw_cross["d_geo_NN"] = d_geo_NN; blk_kw_self["d_geo_NN"] = d_geo_NN
        if self.laat_enabled:  blk_kw_cross["lat_N"]    = lat_N;    blk_kw_self["lat_N"]    = lat_N

        # ★ Tangent-Relative PE: compute bias once per forward (depends on grid only,
        # but the projection MLP is learnable so it must run each forward).
        if self.tangent_attn_enabled and self.tangent_pe is not None:
            tb_NN = self.tangent_pe.bias_NN()                         # (N, N)
            blk_kw_cross["tangent_bias_2N"] = self.tangent_pe.bias_2N()   # (2N, 2N)
            blk_kw_self["tangent_bias_NN"] = tb_NN

        # R3-T: pass the phase module handle (scores computed from Q/K inside blocks).
        if self.tangent_phase_enabled and self.tangent_phase is not None:
            blk_kw_cross["tangent_phase"] = self.tangent_phase
            blk_kw_self["tangent_phase"] = self.tangent_phase

        # Geodesic kernel bias (radial spherical relative position). Routed through
        # the same additive-bias slot; summed if a tangent bias is also present.
        if self.geo_kernel_enabled and self.geo_kernel is not None:
            gk_2N = self.geo_kernel.bias_2N()                        # (2N, 2N)
            gk_NN = self.geo_kernel.bias_NN()                        # (N, N)
            blk_kw_cross["tangent_bias_2N"] = (
                gk_2N if "tangent_bias_2N" not in blk_kw_cross
                else blk_kw_cross["tangent_bias_2N"] + gk_2N)
            blk_kw_self["tangent_bias_NN"] = (
                gk_NN if "tangent_bias_NN" not in blk_kw_self
                else blk_kw_self["tangent_bias_NN"] + gk_NN)

        # ★ FMTA α2 — Cyclic-W relative position bias: per head bias matrices
        if self.seam_cyclic_enabled and self.cyclic_bias is not None:
            blk_kw_cross["cyclic_bias_2N"] = self.cyclic_bias.get_bias_2N()  # (n_heads, 2N, 2N)
            blk_kw_self["cyclic_bias_NN"] = self.cyclic_bias.get_bias_NN()   # (n_heads, N, N)

        # 1. Project + block cascade
        x_tok = self.input_proj(rearrange(x.float(), "b c h w -> b (h w) c"))
        y_tok = self.input_proj(rearrange(y.float(), "b c h w -> b (h w) c"))
        if self.edm_pe is not None:
            x_tok = self.edm_pe(x_tok)
            y_tok = self.edm_pe(y_tok)
        if self.sphere_uniform_enabled and self.sphere_uniform is not None:
            x_tok = rearrange(self.sphere_uniform.to_uniform(
                rearrange(x_tok, "b (h w) c -> b c h w", h=H, w=W)), "b c h w -> b (h w) c")
            y_tok = rearrange(self.sphere_uniform.to_uniform(
                rearrange(y_tok, "b (h w) c -> b c h w", h=H, w=W)), "b c h w -> b (h w) c")
        def _run_cascade(xt, yt, rope_R):
            kwc = dict(blk_kw_cross)
            if rope_R is not None:
                kwc["rope_R"] = rope_R
            for c_blk, s_blk in zip(self.cross_blocks, self.self_blocks):
                xt, yt = c_blk(xt, yt, **kwc)
                xt = s_blk(xt, **blk_kw_self)
                yt = s_blk(yt, **blk_kw_self)
            return xt, yt

        # A4 — gated parallel transport of tokens to canonical frame (γ=0 → tokens).
        if self.pt_transport_enabled and self.pt_transport is not None:
            x_tok = self.pt_transport(x_tok)
            y_tok = self.pt_transport(y_tok)

        # B3 — gated great-circle epipolar bias (pose from current tokens; β=0 → 0).
        if self.epipolar_enabled and self.epipolar is not None:
            blk_kw_cross["extra_bias_2N"] = self.epipolar(x_tok, y_tok)   # (B, 2N, 2N)

        if self.rot_align_enabled:
            # Pass 1: standard cascade -> soft A->B correspondence -> Wahba R̂ (full-coverage).
            x1, y1 = _run_cascade(x_tok, y_tok, None)
            S1 = torch.bmm(F.normalize(x1, dim=-1), F.normalize(y1, dim=-1).transpose(1, 2)) / self.temp
            P_row = F.softmax(S1, dim=-1)                                  # (B, N, N) A attends B
            rays = self._rot_align_rays.to(dtype=x1.dtype, device=x1.device).unsqueeze(0).expand(B, -1, -1)
            nB_soft = F.normalize(torch.bmm(P_row, rays), dim=-1)          # (B, N, 3) soft target rays
            w = P_row.max(dim=-1).values                                  # (B, N) match confidence
            from sccm.models.rotation_align import solve_wahba, gate_rotation
            R_hat = solve_wahba(rays, nB_soft, w)                         # (B,3,3) A->B
            if self.rot_align_detach:
                R_hat = R_hat.detach()
            R_eff = gate_rotation(R_hat.transpose(1, 2), self.rot_align_lam)  # align B->A, lam-gated
            # Pass 2: rotation-aligned cross attention (B-view rotated into A's frame).
            x_tok, y_tok = _run_cascade(x_tok, y_tok, R_eff)
        else:
            x_tok, y_tok = _run_cascade(x_tok, y_tok, None)

        # D1 — gated SO(2)-covariant attention parallel branch (γ=0 → +0).
        if self.so2_cov_enabled and self.so2_cov is not None:
            x_tok = x_tok + self.so2_cov(x_tok)
            y_tok = y_tok + self.so2_cov(y_tok)

        if self.sphere_uniform_enabled and self.sphere_uniform is not None:
            x_tok = rearrange(self.sphere_uniform.to_erp(
                rearrange(x_tok, "b (h w) c -> b c h w", h=H, w=W)), "b c h w -> b (h w) c")
            y_tok = rearrange(self.sphere_uniform.to_erp(
                rearrange(y_tok, "b (h w) c -> b c h w", h=H, w=W)), "b c h w -> b (h w) c")
        x_tok = self.norm(x_tok)
        y_tok = self.norm(y_tok)

        # ★ FMTA α1 — Pole-row uniformity via super-token aggregation.
        # Reduces top/bottom W tokens to n_super super-tokens via attention
        # pooling, then EXPANDS back across W positions. Net effect:
        # pole rows of x_tok / y_tok become uniform (all W positions share
        # the super-token aggregate). Token graph (H×W) preserved → all
        # downstream PE / similarity / covis pathways unchanged.
        # This realizes "pole token redundancy elimination" as a
        # uniformity constraint.
        if self.pole_pool_enabled and self.pole_pool is not None:
            x_red = self.pole_pool(x_tok)
            x_tok = self.pole_pool.expand_back(x_red)               # (B, H*W, D)
            y_red = self.pole_pool(y_tok)
            y_tok = self.pole_pool.expand_back(y_red)

        lat_batched = lat_N.unsqueeze(0).expand(B, -1).contiguous()     # (B, N)
        self._last_HW = (H, W)

        # 3. Similarity
        x_n = F.normalize(x_tok, dim=-1)
        y_n = F.normalize(y_tok, dim=-1)
        S = torch.bmm(x_n, y_n.transpose(1, 2)) / self.temp
        S = self._apply_sltp_prior(S, x_tok, y_tok, H, W)

        # @4 SMK: multiplicative Gaussian kernel on similarity before softmax
        if self.smk_enabled:
            S = self._apply_smk(S, H, W)

        # 4. Covisibility / Sharpness — three mutually exclusive paths producing P:
        #   (SMA, @7)     per-query temperature multiplicative scaling, no additive gate
        #   (cycle, @6)   cycle-consistency μ + additive log-gate (CoMatch-interface)
        #   (default)     CoMatch-style learned per-view μ + additive log-gate
        if self.sma_enabled:
            # SMA — Sphere-Metric Attention: per-query temperature, NO gate
            lat_enc = torch.stack(
                [torch.sin(lat_batched), torch.cos(lat_batched)], dim=-1
            )                                                     # (B, N, 2)
            input_A = torch.cat([x_tok.float(), lat_enc], dim=-1)
            input_B = torch.cat([y_tok.float(), lat_enc], dim=-1)
            log_content_A = self.sma_temp_mlp(input_A).squeeze(-1)  # (B, N_A)
            log_content_B = self.sma_temp_mlp(input_B).squeeze(-1)

            alpha = F.softplus(self.sma_alpha_raw)
            log_cos = torch.log(torch.cos(lat_batched).clamp(min=_SCCM_POLE_EPS))  # (B, N) ≤ 0
            # log τ = -α·log cos(φ) + content_MLP: positive at poles (flat), 0 at equator
            log_tau_A = -alpha * log_cos + log_content_A            # (B, N_A)
            log_tau_B = -alpha * log_cos + log_content_B            # (B, N_B)

            # Sharpness-scaled dual softmax
            tau_A_inv = torch.exp(-log_tau_A).unsqueeze(-1)          # (B, N_A, 1)
            tau_B_inv = torch.exp(-log_tau_B).unsqueeze(-2)          # (B, 1, N_B)
            S_row = S * tau_A_inv
            P_row = F.softmax(S_row, dim=-1)
            S_col = S * tau_B_inv
            P_col = F.softmax(S_col, dim=-2)
            P = P_row * P_col

            # Proxies for backward-compat ("μ" = sharpness = 1/τ)
            self._last_mu_A = 1.0 / torch.exp(log_tau_A)
            self._last_mu_B = 1.0 / torch.exp(log_tau_B)

        elif self.mmm_enabled:
            # MMM — Metric-Only: μ = cos(φ)^α, one learnable α, no MLP
            alpha = F.softplus(self.mmm_alpha_raw)
            log_cos = torch.log(torch.cos(lat_batched).clamp(min=_SCCM_POLE_EPS))
            log_mu = alpha * log_cos                                  # (B, N)
            mu_A = torch.exp(log_mu)
            mu_B = torch.exp(log_mu)                                  # same lat-dist for both views (shared)
            self._last_mu_A = mu_A
            self._last_mu_B = mu_B
            if self.area_ot_enabled:
                P = self._area_marginal_sinkhorn(S, mu_A, mu_B)
            elif self.matching_mode == "dual_softmax":
                P = self._covis_dual_softmax(S, mu_A, mu_B)
            elif self.matching_mode == "sinkhorn":
                P = self._sinkhorn(S)
            else:
                P = F.softmax(S, dim=-1)
        else:
            if self.cycle_covis_enabled:
                # Cycle-consistency: μ from dual-direction score matrix diagonals
                P_AB_raw = F.softmax(S, dim=-1)
                P_BA_raw = F.softmax(S, dim=-2)
                joint = P_AB_raw * P_BA_raw
                cycle_mu_A = joint.sum(dim=-1).clamp(min=1e-8)
                cycle_mu_B = joint.sum(dim=-2).clamp(min=1e-8)
                mu_A = cycle_mu_A
                mu_B = cycle_mu_B
                if self.cycle_covis_jacobian_weight and self.cycle_alpha_raw is not None:
                    alpha = F.softplus(self.cycle_alpha_raw)
                    cos_lat = torch.cos(lat_batched).clamp(min=_SCCM_POLE_EPS)
                    cos_factor = cos_lat.pow(alpha)
                    mu_A = mu_A * cos_factor
                    mu_B = mu_B * cos_factor
            else:
                # CoMatch-style learned per-view covis head — pre-sigmoid logits
                if self.shc_enabled:
                    sh_basis = self._get_shc_basis(H, W, device=x_tok.device,
                                                    dtype=x_tok.dtype)        # (N, K)
                    logit_A = self.covis_head_A.forward_logit(x_tok, lat_batched, sh_basis_NK=sh_basis)
                    logit_B = self.covis_head_B.forward_logit(y_tok, lat_batched, sh_basis_NK=sh_basis)
                else:
                    logit_A = self.covis_head_A.forward_logit(x_tok, lat_batched)
                    logit_B = self.covis_head_B.forward_logit(y_tok, lat_batched)

                # ── PCC — Pose-Conditioned Covisibility (sphere-native add-on) ──
                # Predict translation direction(s) from pooled features and
                # add γ · ray·t̂ to each per-view logit. γ=0 init → identity.
                if self.pcc_enabled and self.pcc_pose_mlp is not None:
                    rays = self._geom(H, W, device=x_tok.device,
                                       dtype=x_tok.dtype)['rays'].to(x_tok.dtype)  # (N, 3)
                    g_A = x_tok.mean(dim=1)                                    # (B, D)
                    g_B = y_tok.mean(dim=1)
                    if self.pcc_per_view_alignment:
                        # Two complementary directions: one per view perspective.
                        # Inputs are concat(self, partner) so each view gets its
                        # own pose hypothesis. Common backbone for both.
                        joint_A = torch.cat([g_A, g_B], dim=-1)
                        joint_B = torch.cat([g_B, g_A], dim=-1)
                        t_A = self.pcc_pose_mlp(joint_A)                       # (B, 3)
                        t_B = self.pcc_pose_mlp(joint_B)
                        t_A = F.normalize(t_A, dim=-1, eps=1e-8)
                        t_B = F.normalize(t_B, dim=-1, eps=1e-8)
                        align_A = torch.einsum('nk,bk->bn', rays, t_A)         # (B, N)
                        align_B = torch.einsum('nk,bk->bn', rays, t_B)
                    else:
                        joint = torch.cat([g_A, g_B], dim=-1)
                        t = self.pcc_pose_mlp(joint)
                        t = F.normalize(t, dim=-1, eps=1e-8)
                        align_A = torch.einsum('nk,bk->bn', rays, t)
                        align_B = align_A
                    gamma = self.pcc_gamma
                    logit_pcc_A = gamma * align_A
                    logit_pcc_B = gamma * align_B
                    logit_A = logit_A + logit_pcc_A
                    logit_B = logit_B + logit_pcc_B
                    # Diagnostics
                    with torch.no_grad():
                        gamma_v = gamma.detach().item() if isinstance(gamma, torch.Tensor) else float(gamma)
                        self._last_pcc_gamma = float(gamma_v)
                        if self.pcc_per_view_alignment:
                            t_norm_v = float((t_A.norm(dim=-1).mean()
                                              + t_B.norm(dim=-1).mean()).item() * 0.5)
                        else:
                            t_norm_v = float(t.norm(dim=-1).mean().item())
                        self._last_pcc_t_norm = t_norm_v
                        self._last_pcc_align_mean = float(
                            (align_A.mean() + align_B.mean()).item() * 0.5)
                        self._last_pcc_logit_delta_mean = float(
                            (logit_pcc_A.abs().mean() + logit_pcc_B.abs().mean()).item() * 0.5)

                if self.gacfast_enabled and self.gacfast_gamma_raw is not None:
                    # ── GAC-fast: saturation-free γ ──
                    if self.training:
                        self._gacfast_step += 1

                    # Pass 1 smoothing (rescaled S)
                    mu_A_p1 = torch.sigmoid(logit_A)
                    mu_B_p1 = torch.sigmoid(logit_B)
                    S_p1 = S / float(self.gacfast_pass1_temp_scale)
                    P_p1 = self._covis_dual_softmax(S_p1, mu_A_p1, mu_B_p1)

                    # Cycle distance — detached, 1-cos by default
                    with torch.no_grad():
                        rays_3d = geom["rays"].to(P_p1.dtype)
                        rays_b = rays_3d.unsqueeze(0).expand(B, -1, -1).contiguous()
                        P_row = P_p1 / P_p1.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                        P_col = P_p1 / P_p1.sum(dim=-2, keepdim=True).clamp(min=1e-8)

                        ray_back_at_B = torch.bmm(P_col.transpose(1, 2), rays_b)
                        cycle_ray_A = torch.bmm(P_row, ray_back_at_B)
                        cycle_ray_A = F.normalize(cycle_ray_A, dim=-1)
                        cos_A = (rays_b * cycle_ray_A).sum(dim=-1)

                        ray_fwd_at_A = torch.bmm(P_row, rays_b)
                        cycle_ray_B = torch.bmm(P_col.transpose(1, 2), ray_fwd_at_A)
                        cycle_ray_B = F.normalize(cycle_ray_B, dim=-1)
                        cos_B = (rays_b * cycle_ray_B).sum(dim=-1)

                        if self.gacfast_d_geo_form == "1-cos":
                            d_geo_A = (1.0 - cos_A).clamp(min=0.0)
                            d_geo_B = (1.0 - cos_B).clamp(min=0.0)
                        else:
                            cos_A_c = cos_A.clamp(-1 + 1e-6, 1 - 1e-6)
                            cos_B_c = cos_B.clamp(-1 + 1e-6, 1 - 1e-6)
                            d_geo_A = torch.acos(cos_A_c)
                            d_geo_B = torch.acos(cos_B_c)

                    # γ via clamp (NO saturation — gradient = 1 in [0, gamma_max])
                    gamma_bounded = torch.clamp(
                        self.gacfast_gamma_raw, min=0.0, max=self.gacfast_gamma_max
                    )
                    cur_step = float(self._gacfast_step.item())
                    warmup_factor = min(1.0, cur_step / max(self.gacfast_warmup_steps, 1))
                    gamma_eff = warmup_factor * gamma_bounded

                    # Additive logit shift (same form as GAC-v2)
                    logit_A_p2 = logit_A - gamma_eff * d_geo_A
                    logit_B_p2 = logit_B - gamma_eff * d_geo_B
                    mu_A = torch.sigmoid(logit_A_p2)
                    mu_B = torch.sigmoid(logit_B_p2)

                    # Diagnostics
                    self._last_gacfast_gamma = float(gamma_eff.detach().item()) if isinstance(gamma_eff, torch.Tensor) else float(gamma_eff)
                    self._last_gacfast_gamma_raw = float(self.gacfast_gamma_raw.detach().item())
                    self._last_gacfast_d_geo_mean = float(
                        (d_geo_A.mean().item() + d_geo_B.mean().item()) * 0.5
                    )
                    self._last_gacfast_d_geo_max = float(
                        max(d_geo_A.max().item(), d_geo_B.max().item())
                    )
                    self._last_gacfast_warmup_factor = float(warmup_factor)
                elif self.gacv2_enabled and self.gacv2_gamma_raw is not None:
                    # ── GAC-v2: engineering revision of GAC ──
                    # Auto-increment step counter (training mode only)
                    if self.training:
                        self._gacv2_step += 1

                    # Pass 1: smoother dual_softmax with rescaled S (higher temp)
                    # S is already S_raw/self.temp. To use higher temp T':
                    #   S_p1 = S * (self.temp / T') = S / pass1_temp_scale
                    mu_A_p1 = torch.sigmoid(logit_A)
                    mu_B_p1 = torch.sigmoid(logit_B)
                    S_p1 = S / float(self.gacv2_pass1_temp_scale)
                    P_p1 = self._covis_dual_softmax(S_p1, mu_A_p1, mu_B_p1)

                    # Cycle distance — fully detached, 1-cos form for smooth grad
                    with torch.no_grad():
                        rays_3d = geom["rays"].to(P_p1.dtype)
                        rays_b = rays_3d.unsqueeze(0).expand(B, -1, -1).contiguous()
                        P_row = P_p1 / P_p1.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                        P_col = P_p1 / P_p1.sum(dim=-2, keepdim=True).clamp(min=1e-8)

                        ray_back_at_B = torch.bmm(P_col.transpose(1, 2), rays_b)
                        cycle_ray_A = torch.bmm(P_row, ray_back_at_B)
                        cycle_ray_A = F.normalize(cycle_ray_A, dim=-1)
                        cos_A = (rays_b * cycle_ray_A).sum(dim=-1)

                        ray_fwd_at_A = torch.bmm(P_row, rays_b)
                        cycle_ray_B = torch.bmm(P_col.transpose(1, 2), ray_fwd_at_A)
                        cycle_ray_B = F.normalize(cycle_ray_B, dim=-1)
                        cos_B = (rays_b * cycle_ray_B).sum(dim=-1)

                        if self.gacv2_d_geo_form == "1-cos":
                            d_geo_A = (1.0 - cos_A).clamp(min=0.0)
                            d_geo_B = (1.0 - cos_B).clamp(min=0.0)
                        else:
                            cos_A_c = cos_A.clamp(-1 + 1e-6, 1 - 1e-6)
                            cos_B_c = cos_B.clamp(-1 + 1e-6, 1 - 1e-6)
                            d_geo_A = torch.acos(cos_A_c)
                            d_geo_B = torch.acos(cos_B_c)

                    # γ with linear warmup + tanh upper-bound to gamma_max
                    gamma_unbounded = F.softplus(self.gacv2_gamma_raw)
                    gamma_bounded = self.gacv2_gamma_max * torch.tanh(
                        gamma_unbounded / self.gacv2_gamma_max
                    )
                    cur_step = float(self._gacv2_step.item())
                    warmup_factor = min(1.0, cur_step / max(self.gacv2_warmup_steps, 1))
                    gamma_eff = warmup_factor * gamma_bounded

                    # Apply: additive logit shift (same form as original GAC)
                    logit_A_p2 = logit_A - gamma_eff * d_geo_A
                    logit_B_p2 = logit_B - gamma_eff * d_geo_B
                    mu_A = torch.sigmoid(logit_A_p2)
                    mu_B = torch.sigmoid(logit_B_p2)

                    # Diagnostic attributes (Python floats, read by training loop)
                    self._last_gacv2_gamma = float(gamma_eff.detach().item()) if isinstance(gamma_eff, torch.Tensor) else float(gamma_eff)
                    self._last_gacv2_gamma_raw = float(self.gacv2_gamma_raw.detach().item())
                    self._last_gacv2_d_geo_mean = float(
                        (d_geo_A.mean().item() + d_geo_B.mean().item()) * 0.5
                    )
                    self._last_gacv2_d_geo_max = float(
                        max(d_geo_A.max().item(), d_geo_B.max().item())
                    )
                    self._last_gacv2_warmup_factor = float(warmup_factor)
                elif self.gacpx_enabled and self.gacpx_mlp is not None:
                    # ── GAC-px: per-pixel learned γ via MLP ──
                    # Pass 1: LAC-only μ
                    mu_A_p1 = torch.sigmoid(logit_A)
                    mu_B_p1 = torch.sigmoid(logit_B)
                    P_p1 = self._covis_dual_softmax(S, mu_A_p1, mu_B_p1)

                    # Cycle distance — fully detached
                    with torch.no_grad():
                        rays_3d = geom["rays"].to(P_p1.dtype)
                        rays_b = rays_3d.unsqueeze(0).expand(B, -1, -1).contiguous()
                        P_row = P_p1 / P_p1.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                        P_col = P_p1 / P_p1.sum(dim=-2, keepdim=True).clamp(min=1e-8)

                        ray_back_at_B = torch.bmm(P_col.transpose(1, 2), rays_b)
                        cycle_ray_A = torch.bmm(P_row, ray_back_at_B)
                        cycle_ray_A = F.normalize(cycle_ray_A, dim=-1)
                        cos_A = (rays_b * cycle_ray_A).sum(dim=-1)

                        ray_fwd_at_A = torch.bmm(P_row, rays_b)
                        cycle_ray_B = torch.bmm(P_col.transpose(1, 2), ray_fwd_at_A)
                        cycle_ray_B = F.normalize(cycle_ray_B, dim=-1)
                        cos_B = (rays_b * cycle_ray_B).sum(dim=-1)

                        if self.gacpx_d_geo_form == "1-cos":
                            d_geo_A = (1.0 - cos_A).clamp(min=0.0)
                            d_geo_B = (1.0 - cos_B).clamp(min=0.0)
                        else:
                            cos_A_c = cos_A.clamp(-1 + 1e-6, 1 - 1e-6)
                            cos_B_c = cos_B.clamp(-1 + 1e-6, 1 - 1e-6)
                            d_geo_A = torch.acos(cos_A_c)
                            d_geo_B = torch.acos(cos_B_c)

                    # Per-pixel γ from MLP(token_features) — DENSE gradient path:
                    # match loss → P → -γ(p)·d_geo → γ(p) → MLP → x_tok features
                    gamma_A = F.softplus(self.gacpx_mlp(x_tok).squeeze(-1))  # (B, N_A)
                    gamma_B = F.softplus(self.gacpx_mlp(y_tok).squeeze(-1))  # (B, N_B)
                    logit_A_p2 = logit_A - gamma_A * d_geo_A
                    logit_B_p2 = logit_B - gamma_B * d_geo_B
                    mu_A = torch.sigmoid(logit_A_p2)
                    mu_B = torch.sigmoid(logit_B_p2)
                elif self.mgac_enabled and self.mgac_gamma_raw is not None:
                    # ── MGAC: multiplicative gate variant of GAC ──
                    # Pass 1: LAC-only μ_base
                    mu_A_base = torch.sigmoid(logit_A)
                    mu_B_base = torch.sigmoid(logit_B)
                    P_p1 = self._covis_dual_softmax(S, mu_A_base, mu_B_base)

                    # Cycle distance — fully detached (gradient only via γ)
                    with torch.no_grad():
                        rays_3d = geom["rays"].to(P_p1.dtype)                # (N, 3)
                        rays_b = rays_3d.unsqueeze(0).expand(B, -1, -1).contiguous()
                        P_row = P_p1 / P_p1.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                        P_col = P_p1 / P_p1.sum(dim=-2, keepdim=True).clamp(min=1e-8)

                        # A-side cycle: A → B → A in 3D ray space
                        ray_back_at_B = torch.bmm(P_col.transpose(1, 2), rays_b)
                        cycle_ray_A = torch.bmm(P_row, ray_back_at_B)
                        cycle_ray_A = F.normalize(cycle_ray_A, dim=-1)
                        cos_A = (rays_b * cycle_ray_A).sum(dim=-1)

                        # B-side cycle: B → A → B
                        ray_fwd_at_A = torch.bmm(P_row, rays_b)
                        cycle_ray_B = torch.bmm(P_col.transpose(1, 2), ray_fwd_at_A)
                        cycle_ray_B = F.normalize(cycle_ray_B, dim=-1)
                        cos_B = (rays_b * cycle_ray_B).sum(dim=-1)

                        if self.mgac_d_geo_form == "1-cos":
                            # smooth gradient near consistency (cos≈1)
                            d_geo_A = (1.0 - cos_A).clamp(min=0.0)
                            d_geo_B = (1.0 - cos_B).clamp(min=0.0)
                        else:
                            cos_A_c = cos_A.clamp(-1 + 1e-6, 1 - 1e-6)
                            cos_B_c = cos_B.clamp(-1 + 1e-6, 1 - 1e-6)
                            d_geo_A = torch.acos(cos_A_c)
                            d_geo_B = torch.acos(cos_B_c)

                    # Multiplicative gate: μ_final = μ_base · exp(-γ·d_geo)
                    # γ_init via inverse-softplus → softplus(-10) ≈ 4.5e-5 (identity)
                    gamma_mgac = F.softplus(self.mgac_gamma_raw)
                    c_A = torch.exp(-gamma_mgac * d_geo_A)
                    c_B = torch.exp(-gamma_mgac * d_geo_B)
                    mu_A = mu_A_base * c_A
                    mu_B = mu_B_base * c_B
                elif self.gac_enabled and self.gac_gamma_raw is not None:
                    # ── GAC: two-pass coarse matching with geodesic cycle ──
                    # Pass 1: LAC-only μ
                    mu_A_p1 = torch.sigmoid(logit_A)
                    mu_B_p1 = torch.sigmoid(logit_B)
                    P_p1 = self._covis_dual_softmax(S, mu_A_p1, mu_B_p1)

                    # Cycle distance — fully detached (Pass 1 is a feedback signal,
                    # not a gradient path; γ_GAC gets its gradient via Pass 2).
                    with torch.no_grad():
                        rays_3d = geom["rays"].to(P_p1.dtype)        # (N, 3)
                        rays_b = rays_3d.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, N, 3)
                        P_row = P_p1 / P_p1.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                        P_col = P_p1 / P_p1.sum(dim=-2, keepdim=True).clamp(min=1e-8)

                        # A-side cycle: A → B (via P_row) → A (via P_col^T) in 3D ray space
                        ray_back_at_B = torch.bmm(P_col.transpose(1, 2), rays_b)  # (B, N_B, 3)
                        cycle_ray_A = torch.bmm(P_row, ray_back_at_B)            # (B, N_A, 3)
                        cycle_ray_A = F.normalize(cycle_ray_A, dim=-1)
                        cos_A = (rays_b * cycle_ray_A).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
                        d_geo_A = torch.acos(cos_A)                              # (B, N_A) in [0, π]

                        # B-side cycle: B → A (via P_col^T) → B (via P_row^T)
                        ray_fwd_at_A = torch.bmm(P_row, rays_b)                  # (B, N_A, 3)
                        cycle_ray_B = torch.bmm(P_col.transpose(1, 2), ray_fwd_at_A)
                        cycle_ray_B = F.normalize(cycle_ray_B, dim=-1)
                        cos_B = (rays_b * cycle_ray_B).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
                        d_geo_B = torch.acos(cos_B)

                    # Pass 2: add GAC term to logit. γ ≥ 0 (via softplus) ensures
                    # consistency d_geo (large = inconsistent) suppresses gate.
                    gamma_gac = F.softplus(self.gac_gamma_raw)
                    logit_A_p2 = logit_A - gamma_gac * d_geo_A
                    logit_B_p2 = logit_B - gamma_gac * d_geo_B
                    mu_A = torch.sigmoid(logit_A_p2)
                    mu_B = torch.sigmoid(logit_B_p2)
                else:
                    mu_A = torch.sigmoid(logit_A)
                    mu_B = torch.sigmoid(logit_B)

                # ── BSC — Bidirectional Symmetric Covisibility ──
                # Refine μ via partner-evidence transport:
                #   ℓ_A^(1)(p) = ℓ_A(p) + β · log Σ_q P_AB(p,q) · μ_B(q)
                #   ℓ_B^(1)(q) = ℓ_B(q) + β · log Σ_p P_BA(p,q) · μ_A(p)
                # β is scalar (no softplus), zero-init → identity at start.
                # Partner μ optionally detached to break the feedback loop
                # (gradient still flows through the *self* μ via logit_A/logit_B).
                # NOTE: only applied in the default (no-GAC) covis path.
                if self.bsc_enabled and self.bsc_beta is not None:
                    mu_A_0 = mu_A
                    mu_B_0 = mu_B
                    if self.bsc_partner_detach:
                        partner_mu_A = mu_A_0.detach()
                        partner_mu_B = mu_B_0.detach()
                    else:
                        partner_mu_A = mu_A_0
                        partner_mu_B = mu_B_0
                    # Raw matching probabilities from similarity alone (no gating).
                    # P_AB(p,q): each row sums to 1 over q.
                    # P_BA(p,q): each col sums to 1 over p.
                    P_AB = F.softmax(S, dim=-1)
                    P_BA = F.softmax(S, dim=-2)
                    expected_mu_B_at_A = torch.bmm(
                        P_AB, partner_mu_B.unsqueeze(-1)
                    ).squeeze(-1)                                       # (B, N_A)
                    expected_mu_A_at_B = torch.bmm(
                        P_BA.transpose(1, 2), partner_mu_A.unsqueeze(-1)
                    ).squeeze(-1)                                       # (B, N_B)
                    log_eMB = torch.log(expected_mu_B_at_A.clamp(min=self.bsc_eps))
                    log_eMA = torch.log(expected_mu_A_at_B.clamp(min=self.bsc_eps))
                    beta = self.bsc_beta
                    logit_A_bsc = logit_A + beta * log_eMB
                    logit_B_bsc = logit_B + beta * log_eMA
                    mu_A = torch.sigmoid(logit_A_bsc)
                    mu_B = torch.sigmoid(logit_B_bsc)
                    # Diagnostics
                    with torch.no_grad():
                        beta_v = beta.detach().item() if isinstance(beta, torch.Tensor) else float(beta)
                        self._last_bsc_beta = float(beta_v)
                        self._last_bsc_logE_mu_B_mean = float(log_eMB.mean().item())
                        self._last_bsc_logE_mu_A_mean = float(log_eMA.mean().item())
                        delta_mean = (
                            (logit_A_bsc - logit_A).abs().mean()
                            + (logit_B_bsc - logit_B).abs().mean()
                        ) * 0.5
                        self._last_bsc_logit_delta_mean = float(delta_mean.item())

            self._last_mu_A = mu_A
            self._last_mu_B = mu_B

            # 5. Covisibility-gated matching (additive log-gate format)
            if self.area_ot_enabled:
                P = self._area_marginal_sinkhorn(S, mu_A, mu_B)
            elif self.matching_mode == "dual_softmax":
                P = self._covis_dual_softmax(S, mu_A, mu_B)
            elif self.matching_mode == "sinkhorn":
                P = self._sinkhorn(S)
            else:
                P = F.softmax(S, dim=-1)

        # ── SPGM override — replace P with sphere-mixture generative density ──
        # Score-free, softmax-free path. Standard P above is discarded.
        # Wasted compute (S + cascade) acknowledged; clean V1 wiring prioritized.
        if self.spgm_enabled and self.spgm_head is not None:
            geom_spgm = self._geom(H, W, device=x.device, dtype=torch.float32)
            p_B_sphere = geom_spgm["rays"].to(x_tok.dtype)                          # (N, 3)
            P = self.spgm_head(x_tok, y_tok, p_B_sphere)                            # (B, N_A, N_B)
            mu_A = P.sum(dim=-1).clamp(min=1e-8, max=1.0)
            mu_B = P.sum(dim=-2).clamp(min=1e-8, max=1.0)
            self._last_mu_A = mu_A
            self._last_mu_B = mu_B

        if self.scct_enabled:
            P = self._apply_scct(P, H, W)
            mu_A = P.sum(dim=-1).clamp(min=1e-8, max=1.0)
            mu_B = P.sum(dim=-2).clamp(min=1e-8, max=1.0)
            self._last_mu_A = mu_A
            self._last_mu_B = mu_B

        if getattr(self, "_cache_plan", False):
            self._last_plan = P
            self._last_covis_A = mu_A
            self._last_covis_B = mu_B

        if self.cache_topk_enabled:
            k_top = min(int(self.cache_topk_k), P.shape[-1])
            topk_idx = torch.topk(P.detach(), k=k_top, dim=-1).indices          # (B, N, K)
            geom_topk = self._geom(H, W, device=P.device, dtype=P.dtype)
            lat = geom_topk["lat"].to(P.dtype)
            lon = geom_topk["lon"].to(P.dtype)
            coords = torch.stack([lon / math.pi, -lat / (math.pi / 2.0)], dim=-1)  # (N, 2)
            topk_coords = coords.to(P.device)[topk_idx]                         # (B, N, K, 2)
            topk_coords = topk_coords.view(B, H, W, k_top, 2).permute(0, 3, 4, 1, 2).contiguous()
            self._last_topk_coords = topk_coords.detach()                       # (B, K, 2, H, W)
        else:
            self._last_topk_coords = None

        # ── GAC-fine: compute and store d_geo grid for Decoder to consume ──
        if self.gacfine_enabled:
            geom_gf = self._geom(H, W, device=P.device, dtype=torch.float32)
            with torch.no_grad():
                rays_gf = geom_gf["rays"].to(P.dtype)               # (N, 3)
                rays_bf = rays_gf.unsqueeze(0).expand(B, -1, -1).contiguous()
                P_row_gf = P / P.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                P_col_gf = P / P.sum(dim=-2, keepdim=True).clamp(min=1e-8)
                ray_back_at_B = torch.bmm(P_col_gf.transpose(1, 2), rays_bf)
                cycle_ray_A   = torch.bmm(P_row_gf, ray_back_at_B)
                cycle_ray_A   = F.normalize(cycle_ray_A, dim=-1)
                cos_A_gf = (rays_bf * cycle_ray_A).sum(dim=-1)
                if self.gacfine_d_geo_form == "1-cos":
                    d_geo_grid = (1.0 - cos_A_gf).clamp(min=0.0)
                else:
                    cos_clip = cos_A_gf.clamp(-1 + 1e-6, 1 - 1e-6)
                    d_geo_grid = torch.acos(cos_clip)
                # Reshape (B, N) → (B, H, W) for Decoder to upsample
                self._gacfine_d_geo_grid = d_geo_grid.view(B, H, W).detach()
        else:
            self._gacfine_d_geo_grid = None

        # 4. Row-normalize + PE aggregation (Fourier / SHPA / σFP-residual / σFP-replace)
        P_row = P / (P.sum(dim=-1, keepdim=True).clamp(min=1e-8))
        if self.sh_pe_enabled:
            pe = self._sh_pe(B, H, W, x.device)
        elif self.sfp_pe_enabled and self.sfp_mode == "replace":
            # v012: σFP replaces Fourier PE entirely (scale-matched Kaiming init)
            pe = self._sfp_pe(B, H, W, x.device)
        else:
            pe = self._get_fourier_pe(B, H, W, x.device)
            if self.sfp_pe_enabled and self.sfp_mode == "residual":
                # v011: σFP as additive residual (zero-init, warm-start)
                pe = pe + self._sfp_pe(B, H, W, x.device)
        match_emb = torch.bmm(P_row, pe)

        # @5 Yaw-Cyclic Self-Consistency aux loss (training-time only)
        # v011 fix: (a) symmetric gradient (no detach), (b) warmup_steps delay,
        #           (c) reduced λ default 0.1 → 0.05.
        # Second forward pass on yaw-rotated inputs; aux loss enforces:
        #   covis(roll(x)) ≈ roll(covis(x))
        # Both paths share weights — gradient flows through both, so the symmetric
        # aux loss doesn't arbitrarily favor one path (previous detach made rolled
        # path carry all gradient, creating asymmetric training).
        self._last_aux_loss = None
        global_step = getattr(sccm, "GLOBAL_STEP", 0)
        if self.training and self.yaw_cyclic_enabled and global_step >= self.yaw_cyclic_warmup_steps:
            # Shared RNG for A and B (keep pair structure)
            shift = int(torch.randint(1, W, (1,), device=x.device).item())
            x_rolled = torch.roll(x, shifts=shift, dims=-1)                # (B, C, H, W)
            y_rolled = torch.roll(y, shifts=shift, dims=-1)

            # Mini-re-run: input_proj + blocks + covis head on rolled features.
            xr = self.input_proj(rearrange(x_rolled.float(), "b c h w -> b (h w) c"))
            yr = self.input_proj(rearrange(y_rolled.float(), "b c h w -> b (h w) c"))
            for c_blk, s_blk in zip(self.cross_blocks, self.self_blocks):
                xr, yr = c_blk(xr, yr, **blk_kw)
                xr = s_blk(xr, **blk_kw)
                yr = s_blk(yr, **blk_kw)
            xr = self.norm(xr); yr = self.norm(yr)
            if self.shc_enabled:
                sh_basis_yc = self._get_shc_basis(H, W, device=xr.device, dtype=xr.dtype)
                mu_A_rolled = self.covis_head_A(xr, lat_batched, sh_basis_NK=sh_basis_yc)
                mu_B_rolled = self.covis_head_B(yr, lat_batched, sh_basis_NK=sh_basis_yc)
            else:
                mu_A_rolled = self.covis_head_A(xr, lat_batched)               # (B, N)
                mu_B_rolled = self.covis_head_B(yr, lat_batched)

            # Inverse-roll to align spatially
            mu_A_rolled_grid = mu_A_rolled.view(B, H, W)
            mu_B_rolled_grid = mu_B_rolled.view(B, H, W)
            mu_A_rolled_back = torch.roll(mu_A_rolled_grid, shifts=-shift, dims=-1).view(B, N)
            mu_B_rolled_back = torch.roll(mu_B_rolled_grid, shifts=-shift, dims=-1).view(B, N)

            # Symmetric aux loss: NO detach on either side → both paths learn equivariance
            aux_A = F.mse_loss(mu_A_rolled_back, mu_A)
            aux_B = F.mse_loss(mu_B_rolled_back, mu_B)
            self._last_aux_loss = self.yaw_cyclic_lambda * 0.5 * (aux_A + aux_B)

        return rearrange(match_emb, "b (h w) d -> b d h w", h=H, w=W)

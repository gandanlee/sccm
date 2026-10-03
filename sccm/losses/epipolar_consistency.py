"""Spherical-epipolar Sampson residual as an auxiliary loss.

For each source pixel u_A we have the model's predicted warp target u_B
(normalised ERP coords). Convert to unit rays r_A, r_B and impose the
epipolar constraint
    r_A^T E r_B = 0,     E = [t]_× R,
where R, t come from the GT relative pose `T_1to2` (4×4 w2c). The constraint
is depth-independent (any corresponding 3-D point X satisfies it for all
valid depths), so it is a pure geometric regulariser that the matching
loss cannot see.

Loss = Σ_i w_i · Sampson_i(r_A^i, r_B^i; E),
    Sampson = (r_A^T E r_B)² / (||E r_B||² + ||E^T r_A||² + ε)

Weights w_i mix predicted certainty (σ(logit)) with an optional GT-valid
mask. Identical math as Kornia's `sampson_epipolar_distance` but applied to
unit rays (no pinhole intrinsics).

Returns (loss, n_valid) so the caller can log / average.

Gradient stability
- Denominator floor ε = 1e-6 → no div-by-zero.
- `T_1to2` is detached (GT pose).
- All ops differentiable. `r_B` depends on `warp` through the smooth
  `erp_normalized_to_ray` map; no asin singularity (clamps already applied).
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from sccm.utils.utils_sphere import erp_normalized_to_ray


def _skew_sym(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) → (..., 3, 3) skew-symmetric matrix [v]_×."""
    zero = torch.zeros_like(v[..., :1])
    vx, vy, vz = v[..., 0:1], v[..., 1:2], v[..., 2:3]
    row0 = torch.cat([zero, -vz, vy], dim=-1)
    row1 = torch.cat([vz, zero, -vx], dim=-1)
    row2 = torch.cat([-vy, vx, zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def essential_matrix_from_pose(T_1to2: torch.Tensor) -> torch.Tensor:
    """Build (B, 3, 3) E = [t]_× R from (B, 4, 4) world-to-camera transforms.

    Treated as detached — no gradient through pose.
    """
    with torch.no_grad():
        R = T_1to2[:, :3, :3].contiguous()
        t = T_1to2[:, :3, 3].contiguous()
        # Unit-normalise t to kill scene-scale ambiguity (E is up to scalar).
        t_norm = torch.linalg.vector_norm(t, dim=-1, keepdim=True).clamp(min=1e-8)
        t_hat = t / t_norm
        E = _skew_sym(t_hat) @ R
    return E


def epipolar_sampson_residual(
    warp: torch.Tensor,           # (B, 2, H, W) predicted warp in ERP normalized coords
    T_1to2: torch.Tensor,         # (B, 4, 4) GT pose
    certainty_logits: torch.Tensor,  # (B, 1, H, W) predicted certainty logits
    valid_mask: torch.Tensor | None = None,  # (B, H, W) — 1 where GT is valid
    cert_gate: bool = True,
    use_robust: bool = False,
    robust_c: float = 0.01,       # rad — Cauchy scale for IRLS
    eps_denom: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (loss_scalar, n_effective_pixels)."""
    B, _, H, W = warp.shape
    device = warp.device
    dtype = warp.dtype

    # Source pixel grid (broadcast to B).
    u = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=device, dtype=dtype)
    v = torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(v, u, indexing="ij")
    src_uv = torch.stack([gx, gy], dim=-1)         # (H, W, 2)
    src_uv = src_uv.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

    r_A = erp_normalized_to_ray(src_uv)                    # (B, H, W, 3)
    warp_uv = warp.permute(0, 2, 3, 1).contiguous()        # (B, H, W, 2)
    r_B = erp_normalized_to_ray(warp_uv)                   # (B, H, W, 3)

    E = essential_matrix_from_pose(T_1to2.to(dtype))       # (B, 3, 3)
    E_T = E.transpose(-1, -2)                              # (B, 3, 3)

    # E r_B : (B, H, W, 3)
    Er_B = torch.einsum("bij,bhwj->bhwi", E, r_B)
    # E^T r_A : (B, H, W, 3)
    EtrA = torch.einsum("bij,bhwj->bhwi", E_T, r_A)

    # constraint residual c_i = r_A^T E r_B
    c = (r_A * Er_B).sum(dim=-1, keepdim=True)  # (B, H, W, 1)
    num = c.squeeze(-1) ** 2                              # (B, H, W)
    # Tangent-projected denominator (rigorous spherical Sampson):
    # Perturbations of r_A and r_B must lie in T_{r_A}S² and T_{r_B}S².
    # The projections of E r_B and E^T r_A onto these tangent planes are
    # the correct Jacobian magnitudes.
    Er_B_tan = Er_B - c * r_A
    EtrA_tan = EtrA - c * r_B
    den = Er_B_tan.pow(2).sum(dim=-1) + EtrA_tan.pow(2).sum(dim=-1) + eps_denom
    sampson = num / den                                     # (B, H, W) — non-negative

    # Weights: certainty gate (soft) × valid mask (hard).
    w = torch.ones_like(sampson)
    if cert_gate:
        w = w * torch.sigmoid(certainty_logits[:, 0].detach())
    if valid_mask is not None:
        w = w * valid_mask.to(w.dtype)

    if use_robust:
        # Cauchy IRLS reweighting (detached, as in robust M-estimation).
        with torch.no_grad():
            resid = sampson.sqrt().clamp(min=1e-8)
            irls = 1.0 / (1.0 + (resid / robust_c) ** 2)
        w = w * irls

    total_w = w.sum().clamp(min=1e-6)
    loss = (sampson * w).sum() / total_w
    return loss, total_w.detach()


@torch.no_grad()
def _self_test():
    """Sanity: GT-warped rays satisfy constraint, random rays don't."""
    B, H, W = 1, 4, 8
    T = torch.eye(4)[None].clone()
    T[0, :3, 3] = torch.tensor([0.1, 0.0, 0.0])  # small translation
    # Construct GT warp = identity (so r_A = r_B, should give 0 residual
    # only if t × r_A · r_A ≡ 0, which is true: (t × r_A) ⊥ r_A).
    u = torch.linspace(-1 + 1 / W, 1 - 1 / W, W)
    v = torch.linspace(-1 + 1 / H, 1 - 1 / H, H)
    gy, gx = torch.meshgrid(v, u, indexing="ij")
    identity_warp = torch.stack([gx, gy], dim=-1).permute(2, 0, 1)[None]
    cert = torch.zeros(B, 1, H, W)
    loss_id, _ = epipolar_sampson_residual(identity_warp, T, cert, cert_gate=False)
    random_warp = identity_warp + 0.3 * torch.randn_like(identity_warp)
    loss_rand, _ = epipolar_sampson_residual(random_warp, T, cert, cert_gate=False)
    print(f"epipolar: id={loss_id.item():.4e}, rand={loss_rand.item():.4e} (expect id≪rand)")


if __name__ == "__main__":
    _self_test()

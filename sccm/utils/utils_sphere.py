"""
ERP (Equirectangular Projection) geometry utilities for RoMa.

Coordinate convention (matches data/matterport3d/preprocess.py::_py360_dirs_grid):
    lon (u): linspace(-pi, pi, W)     — image left to right
    lat (v): linspace(pi/2, -pi/2, H) — image top to bottom
    direction: x = cos(lat)*sin(lon), y = sin(lat), z = cos(lat)*cos(lon)

Normalized coords [-1, 1] (for grid_sample, align_corners=False):
    u_normalized = lon / pi           — [-1, 1]
    v_normalized = -lat / (pi/2)      — [-1, 1] (top=-1, bottom=+1)

Depth: radial distance from camera center (not z-depth).
"""

import math
import torch
import torch.nn.functional as F


def erp_pixel_to_ray(u: torch.Tensor, v: torch.Tensor,
                     w: float, h: float) -> torch.Tensor:
    """Pixel coords (u, v) -> unit ray direction on sphere.

    Args:
        u, v: (...) float pixel coordinates
        w, h: image width/height
    Returns:
        rays: (..., 3) unit direction vectors
    """
    lon = (u / w - 0.5) * (2 * math.pi)   # [-pi, pi]
    lat = (0.5 - v / h) * math.pi          # [pi/2, -pi/2] top to bottom
    cos_lat = torch.cos(lat)
    x = cos_lat * torch.sin(lon)
    y = torch.sin(lat)
    z = cos_lat * torch.cos(lon)
    return torch.stack([x, y, z], dim=-1)


def erp_ray_to_pixel(ray: torch.Tensor,
                     w: float, h: float) -> torch.Tensor:
    """Unit ray direction -> pixel coords (u, v) in ERP image.

    Args:
        ray: (..., 3) direction vectors (need not be unit)
        w, h: image width/height
    Returns:
        uv: (..., 2) pixel coordinates
    """
    eps = 1e-7
    x, y, z = ray[..., 0], ray[..., 1], ray[..., 2]
    lon = torch.atan2(x, z)                                      # [-pi, pi]
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2 + 1e-12)
    lat = torch.asin(torch.clamp(y / r, -1 + eps, 1 - eps))     # [-pi/2, pi/2]
    u = (lon / (2 * math.pi) + 0.5) * w
    v = (0.5 - lat / math.pi) * h
    return torch.stack([u, v], dim=-1)


def erp_normalized_to_ray(kpts_n: torch.Tensor) -> torch.Tensor:
    """Normalized [-1,1] coords -> unit ray.

    Args:
        kpts_n: (..., 2) where kpts_n[...,0]=u_norm, kpts_n[...,1]=v_norm
    Returns:
        rays: (..., 3)
    """
    lon = math.pi * kpts_n[..., 0]              # [-pi, pi]
    lat = -(math.pi / 2) * kpts_n[..., 1]       # [pi/2, -pi/2]
    cos_lat = torch.cos(lat)
    x = cos_lat * torch.sin(lon)
    y = torch.sin(lat)
    z = cos_lat * torch.cos(lon)
    return torch.stack([x, y, z], dim=-1)


def erp_ray_to_normalized(ray: torch.Tensor) -> torch.Tensor:
    """3D direction -> normalized [-1,1] ERP coords.

    Args:
        ray: (..., 3)
    Returns:
        kpts_n: (..., 2)
    """
    eps = 1e-7
    x, y, z = ray[..., 0], ray[..., 1], ray[..., 2]
    lon = torch.atan2(x, z)
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2 + 1e-12)
    lat = torch.asin(torch.clamp(y / r, -1 + eps, 1 - eps))
    u_n = lon / math.pi
    v_n = -lat / (math.pi / 2)
    return torch.stack([u_n, v_n], dim=-1)


@torch.no_grad()
def warp_kpts_erp(
    kpts0: torch.Tensor,
    depth0: torch.Tensor,
    depth1: torch.Tensor,
    T_0to1: torch.Tensor,
    smooth_mask: bool = False,
    return_relative_depth_error: bool = False,
    depth_interpolation_mode: str = "bilinear",
    relative_depth_error_threshold: float = 0.05,
) -> tuple:
    """Warp keypoints from ERP image 0 to ERP image 1 via depth and pose.

    Algorithm:
        1. Sample radial depth at kpts0
        2. kpts0 -> unit ray via spherical coords
        3. ray * depth -> 3D point in camera 0
        4. R @ pt + t -> 3D point in camera 1
        5. pt -> ray -> ERP pixel in image 1 (with horizontal wrapping)
        6. Depth consistency check

    Args:
        kpts0: (N, L, 2) normalized [-1,1] coordinates
        depth0: (N, H, W) radial depth map of image 0
        depth1: (N, H, W) radial depth map of image 1
        T_0to1: (N, 3, 4) or (N, 4, 4) relative pose transform
        smooth_mask: False for hard mask, float for soft exponential mask
        return_relative_depth_error: if True, return (error, warped_kpts)
        depth_interpolation_mode: grid_sample interpolation mode
        relative_depth_error_threshold: threshold for depth consistency

    Returns:
        valid_mask: (N, L) bool or float
        w_kpts0_norm: (N, L, 2) warped coords in [-1, 1]
    """
    n, h, w = depth0.shape

    # 1. Sample radial depth at kpts0
    kpts0_depth = F.grid_sample(
        depth0[:, None], kpts0[:, :, None],
        mode=depth_interpolation_mode, align_corners=False,
    )[:, 0, :, 0]  # (N, L)

    nonzero_mask = kpts0_depth != 0

    # 2. Normalized coords -> unit ray
    rays = erp_normalized_to_ray(kpts0)  # (N, L, 3)

    # 3. Scale by depth -> 3D points in camera 0
    pts_cam0 = rays * kpts0_depth[..., None]  # (N, L, 3)

    # 4. Rigid transform to camera 1
    R = T_0to1[:, :3, :3]  # (N, 3, 3)
    t = T_0to1[:, :3, 3]   # (N, 3)
    pts_cam1 = (R @ pts_cam0.transpose(2, 1)).transpose(2, 1) + t[:, None, :]  # (N, L, 3)

    # Radial distance in camera 1
    w_depth_computed = pts_cam1.norm(dim=-1)  # (N, L)

    # 5. Project to ERP image 1
    w_kpts0_norm = erp_ray_to_normalized(pts_cam1)  # (N, L, 2)

    # Horizontal wrapping: wrap u into [-1, 1]
    u_wrapped = w_kpts0_norm[..., 0]
    u_wrapped = u_wrapped - 2.0 * torch.round(u_wrapped / 2.0)
    w_kpts0_norm = torch.stack([u_wrapped, w_kpts0_norm[..., 1]], dim=-1)

    # 6. Covisibility: ERP wraps horizontally, only check vertical bounds
    covisible_mask = (
        (w_kpts0_norm[..., 1] >= -1.0) & (w_kpts0_norm[..., 1] <= 1.0)
    )

    # 7. Sample depth1 at warped positions
    w_kpts0_depth_sampled = F.grid_sample(
        depth1[:, None], w_kpts0_norm[:, :, None],
        mode=depth_interpolation_mode, align_corners=False,
    )[:, 0, :, 0]  # (N, L)

    valid_depth1 = w_kpts0_depth_sampled > 1e-6
    relative_depth_error = (
        (w_kpts0_depth_sampled - w_depth_computed) / (w_kpts0_depth_sampled + 1e-8)
    ).abs()

    if return_relative_depth_error:
        return relative_depth_error, w_kpts0_norm

    hard_mask = nonzero_mask & covisible_mask & valid_depth1

    if not smooth_mask:
        valid_mask = hard_mask & (relative_depth_error < relative_depth_error_threshold)
    else:
        consistent_weight = (-relative_depth_error / smooth_mask).exp()
        valid_mask = hard_mask.float() * consistent_weight

    return valid_mask, w_kpts0_norm


def get_gt_warp_erp(
    depth1: torch.Tensor,
    depth2: torch.Tensor,
    T_1to2: torch.Tensor,
    H: int = None,
    W: int = None,
    depth_interpolation_mode: str = "bilinear",
    relative_depth_error_threshold: float = 0.05,
    smooth_mask: bool = False,
    # K1, K2 accepted but ignored (interface compat with get_gt_warp)
    K1=None,
    K2=None,
) -> tuple:
    """Dense GT warp for ERP images. No intrinsics needed.

    Creates a dense grid over image 1, warps every pixel to image 2
    using spherical geometry and depth.

    Args:
        depth1, depth2: (B, H, W) radial depth maps
        T_1to2: (B, 4, 4) relative pose
        H, W: output resolution (if None, use depth1 shape)

    Returns:
        x2: (B, H, W, 2) warped coords in [-1, 1]
        prob: (B, H, W) validity mask (0 or 1)
    """
    if H is None:
        B, H, W = depth1.shape
    else:
        B = depth1.shape[0]

    with torch.no_grad():
        x1_n = torch.meshgrid(
            *[
                torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=depth1.device)
                for n in (B, H, W)
            ],
            indexing="ij",
        )
        x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)

        mask, x2 = warp_kpts_erp(
            x1_n.double(),
            depth1.double(),
            depth2.double(),
            T_0to1=T_1to2.double(),
            depth_interpolation_mode=depth_interpolation_mode,
            relative_depth_error_threshold=relative_depth_error_threshold,
            smooth_mask=smooth_mask,
        )
        prob = mask.float().reshape(B, H, W)
        x2 = x2.reshape(B, H, W, 2)
        return x2, prob


# ─────────────────────────────────────────────────────────────────────────
# Additions for depth-robust experiments (2026-04-15):
#   - tangent_basis_erp: pole-safe orthonormal tangent frame on S²
#   - exp_map_sphere:    geodesic retraction via slerp (numerically stable at |v|→0)
#   - real_spherical_harmonics: ℓ = 0..L real SH basis on rays, smooth on S²
# All functions are differentiable and add no new singularities beyond
# the existing erp_ray_to_normalized(·) (asin clamp already handled).
# ─────────────────────────────────────────────────────────────────────────


def tangent_basis_erp(ray: torch.Tensor) -> torch.Tensor:
    """Pole-safe orthonormal tangent frame T(r) ∈ R^{...,3,2} on S².

    Column 0 = ∂r/∂φ   (latitude direction, always unit norm everywhere including poles)
    Column 1 = r × col0 (longitude-like direction, unit norm; degenerates in direction
                         at poles but still orthonormal to col0 and r)

    For r = (cosφ sinλ, sinφ, cosφ cosλ):
        ∂r/∂φ = (-sinφ sinλ,  cosφ, -sinφ cosλ)   — ||·||=1 for all φ, λ
    The second basis via cross product is well-defined because ray and ∂r/∂φ
    are always linearly independent on S² (they are orthogonal by construction).
    """
    # Extract implicit (φ, λ) from ray components. ray assumed unit.
    x, y, z = ray[..., 0], ray[..., 1], ray[..., 2]
    # φ = asin(y), but we avoid singularity by using trig identities directly.
    # ∂r/∂φ at (φ, λ): need sinφ, cosφ, sinλ, cosλ.
    # We have sinφ = y, cosφ = sqrt(1 - y²), sinλ ∝ x/cosφ, cosλ ∝ z/cosφ.
    # Near pole cosφ → 0 makes (sinλ, cosλ) indeterminate, but ∂r/∂φ itself
    # is still well-defined as limit — we can compute directly via a chart swap.
    # Stable formula: e_phi = (-sinφ · x / cosφ, cosφ, -sinφ · z / cosφ)
    #                     = (-y · x / cosφ, cosφ, -y · z / cosφ)
    # At pole (cosφ → 0) the x, z → 0 too, so x/cosφ, z/cosφ → finite values
    # (sinλ, cosλ respectively). We use clamp to avoid 0/0.
    eps = 1e-7
    cos_phi = torch.sqrt(torch.clamp(1 - y * y, min=eps))
    sin_lam = x / cos_phi
    cos_lam = z / cos_phi
    sin_phi = y
    # e_phi (partial wrt latitude, unit everywhere)
    e_phi = torch.stack([
        -sin_phi * sin_lam,
        cos_phi,
        -sin_phi * cos_lam,
    ], dim=-1)
    # Second basis: r × e_phi (guaranteed unit because r ⊥ e_phi and both unit)
    e_lam = torch.cross(ray, e_phi, dim=-1)
    # Stack columns → (..., 3, 2)
    return torch.stack([e_phi, e_lam], dim=-1)


def exp_map_sphere(ray: torch.Tensor, tangent_vec: torch.Tensor) -> torch.Tensor:
    """Geodesic retraction on S²: r_new = cos(|v|) r + sinc(|v|) v.

    Uses torch.sinc (= sin(πx)/(πx)) for numerical stability at |v|→0.
    Fully differentiable; smooth everywhere; preserves ||r_new||=1.
    """
    norm = torch.linalg.vector_norm(tangent_vec, dim=-1, keepdim=True).clamp(min=0.0)
    # sinc(x) in torch is sin(πx)/(πx); we want sin(|v|)/|v| = sinc(|v|/π)
    sinc_v = torch.sinc(norm / math.pi)
    cos_v = torch.cos(norm)
    out = cos_v * ray + sinc_v * tangent_vec
    # Reproject to unit sphere to kill fp round-off (differentiable).
    out = out / torch.linalg.vector_norm(out, dim=-1, keepdim=True).clamp(min=1e-7)
    return out


def log_map_sphere(ray_p: torch.Tensor, ray_q: torch.Tensor) -> torch.Tensor:
    """Inverse of exp_map_sphere: tangent vector v at p such that exp_p(v) = q.

    v lies in T_p S² (orthogonal to ray_p), with ||v|| = arccos(p·q) (geodesic dist).
    Differentiable everywhere except at antipodes (p·q = -1) — clamped for stability.
    """
    dot = (ray_p * ray_q).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(dot)                          # (..., 1) geodesic distance
    proj = ray_q - dot * ray_p                       # (..., 3) component of q ⊥ p
    proj_norm = torch.linalg.vector_norm(proj, dim=-1, keepdim=True).clamp(min=1e-7)
    v = theta * proj / proj_norm                     # (..., 3) tangent vec, ||v||=theta
    return v


def real_spherical_harmonics(ray: torch.Tensor, L_max: int = 4) -> torch.Tensor:
    """Real spherical harmonics Y_ℓ^m(r) for ℓ = 0..L_max, m = -ℓ..+ℓ.

    Input ray: (..., 3) unit vectors in the convention
        x = cosφ sinλ,  y = sinφ,  z = cosφ cosλ
    (consistent with this repo's ERP utilities).

    Output: (..., (L+1)²) channels, ordered (ℓ=0,m=0), (ℓ=1,m=-1..1), (ℓ=2,m=-2..2), ...

    Uses the standard normalized real SH formula. Smooth, bounded in [-C, +C]
    per channel (C depends on ℓ), fully differentiable. No singularities.
    """
    x, y, z = ray[..., 0], ray[..., 1], ray[..., 2]
    # In this repo convention "up" is +y (latitude). Real SH are typically defined
    # with z as polar axis; swap to preserve standard formulas.
    px, py, pz = x, z, y  # so that pz plays the role of "cos θ" (colatitude cosine)
    channels = []
    norm0 = 1.0 / (2.0 * math.sqrt(math.pi))
    if L_max >= 0:
        Y00 = torch.full_like(pz, norm0)
        channels.append(Y00)
    if L_max >= 1:
        c1 = math.sqrt(3.0 / (4.0 * math.pi))
        channels.extend([c1 * py, c1 * pz, c1 * px])  # (m=-1, 0, +1)
    if L_max >= 2:
        c2m2 = 0.5 * math.sqrt(15.0 / math.pi)
        c2m1 = 0.5 * math.sqrt(15.0 / math.pi)
        c20  = 0.25 * math.sqrt(5.0 / math.pi)
        c2p1 = 0.5 * math.sqrt(15.0 / math.pi)
        c2p2 = 0.25 * math.sqrt(15.0 / math.pi)
        channels.extend([
            c2m2 * px * py,
            c2m1 * py * pz,
            c20 * (3.0 * pz * pz - 1.0),
            c2p1 * px * pz,
            c2p2 * (px * px - py * py),
        ])
    if L_max >= 3:
        c3m3 = 0.25 * math.sqrt(35.0 / (2.0 * math.pi))
        c3m2 = 0.5 * math.sqrt(105.0 / math.pi)
        c3m1 = 0.25 * math.sqrt(21.0 / (2.0 * math.pi))
        c30  = 0.25 * math.sqrt(7.0 / math.pi)
        c3p1 = 0.25 * math.sqrt(21.0 / (2.0 * math.pi))
        c3p2 = 0.25 * math.sqrt(105.0 / math.pi)
        c3p3 = 0.25 * math.sqrt(35.0 / (2.0 * math.pi))
        channels.extend([
            c3m3 * py * (3.0 * px * px - py * py),
            c3m2 * px * py * pz,
            c3m1 * py * (5.0 * pz * pz - 1.0),
            c30 * pz * (5.0 * pz * pz - 3.0),
            c3p1 * px * (5.0 * pz * pz - 1.0),
            c3p2 * pz * (px * px - py * py),
            c3p3 * px * (px * px - 3.0 * py * py),
        ])
    if L_max >= 4:
        c4m4 = 0.75 * math.sqrt(35.0 / math.pi)
        c4m3 = 0.75 * math.sqrt(35.0 / (2.0 * math.pi))
        c4m2 = 0.75 * math.sqrt(5.0 / math.pi)
        c4m1 = 0.75 * math.sqrt(5.0 / (2.0 * math.pi))
        c40  = 3.0 / 16.0 * math.sqrt(1.0 / math.pi)
        c4p1 = 0.75 * math.sqrt(5.0 / (2.0 * math.pi))
        c4p2 = 3.0 / 8.0 * math.sqrt(5.0 / math.pi)
        c4p3 = 0.75 * math.sqrt(35.0 / (2.0 * math.pi))
        c4p4 = 3.0 / 16.0 * math.sqrt(35.0 / math.pi)
        channels.extend([
            c4m4 * px * py * (px * px - py * py),
            c4m3 * py * pz * (3.0 * px * px - py * py),
            c4m2 * px * py * (7.0 * pz * pz - 1.0),
            c4m1 * py * pz * (7.0 * pz * pz - 3.0),
            c40 * (35.0 * pz ** 4 - 30.0 * pz * pz + 3.0),
            c4p1 * px * pz * (7.0 * pz * pz - 3.0),
            c4p2 * (px * px - py * py) * (7.0 * pz * pz - 1.0),
            c4p3 * px * pz * (px * px - 3.0 * py * py),
            c4p4 * (px ** 4 - 6.0 * px * px * py * py + py ** 4),
        ])
    if L_max > 4:
        raise ValueError(f"L_max>4 not implemented (got {L_max}).")
    return torch.stack(channels, dim=-1)


# ─────────────────────────────────────────────────────────────────────────
# Spherical-harmonic projection / inversion helpers on an ERP grid.
#
# Convention (matches `erp_normalized_to_ray`):
#   u ∈ [-1,1] along width  → lon = π·u
#   v ∈ [-1,1] along height → lat = -(π/2)·v,   colat θ = π/2 - lat
# Area element on S²: dω = sin(θ) dθ dφ = cos(lat) · (π/2) · π · du·dv
#                        = (π²/2) · cos(lat) · du·dv
# Per-pixel weight (Δu=2/W, Δv=2/H):   w_pix = 2π² · cos(lat) / (H·W)
#
# Uses the real SH basis already defined via `real_spherical_harmonics`.
# Currently limited to L_max ≤ 4 (25 basis functions). L_max=4 is sufficient
# as a low-pass smoother for refiner velocity fields — spectral decay e^{-20τ}
# at ℓ=4 with τ~0.05 knocks out frequencies above the coarse scale. Extend
# to higher L by adding recurrences below if needed.
#
# All functions are differentiable; they allocate O(H·W·(L+1)²) temporaries.
# ─────────────────────────────────────────────────────────────────────────


def _erp_grid_rays_and_weights(H: int, W: int, device, dtype=torch.float32):
    """Return (rays (H, W, 3), weights (H, W)) for an ERP sample grid.
    Weights integrate to 4π on S².
    """
    u = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=device, dtype=dtype)
    v = torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(v, u, indexing="ij")
    uv = torch.stack([gx, gy], dim=-1)                  # (H, W, 2)
    rays = erp_normalized_to_ray(uv)                    # (H, W, 3)

    # cos(lat) = cos(-v·π/2) = cos(v·π/2)
    cos_lat = torch.cos(v * (math.pi / 2))              # (W,) no wait
    cos_lat = torch.cos(gy * (math.pi / 2))             # (H, W) — use grid
    pixel_w = (2 * math.pi ** 2 / (H * W)) * cos_lat    # (H, W)
    return rays, pixel_w


def sh_project_erp(f_grid: torch.Tensor, L_max: int = 4,
                    rays: torch.Tensor = None, weights: torch.Tensor = None):
    """Project a scalar field sampled on an ERP grid onto real SH basis.

    f_grid: (..., H, W) — arbitrary leading batch/channel dims.
    L_max: SH cutoff (≤ 4 with current `real_spherical_harmonics`).
    rays: precomputed (H, W, 3) unit rays; optional (will be built if None).
    weights: precomputed (H, W) integration weights; optional.

    Returns: (..., K) coefficients where K = (L_max+1)².
    """
    assert f_grid.dim() >= 2
    H, W = f_grid.shape[-2:]
    if rays is None or weights is None:
        rays, weights = _erp_grid_rays_and_weights(H, W, f_grid.device, f_grid.dtype)
    Y = real_spherical_harmonics(rays, L_max=L_max)     # (H, W, K)
    # Weighted inner product along (H, W):
    #   c_k = Σ_{hw} Y_k(r_hw) · f(r_hw) · w(r_hw)
    wf = f_grid * weights                               # (..., H, W)
    # Einsum over the last two dims.
    coeffs = torch.einsum("...hw,hwk->...k", wf, Y)
    return coeffs


def sh_inverse_erp(coeffs: torch.Tensor, H: int, W: int, L_max: int = 4,
                    rays: torch.Tensor = None):
    """Inverse SH: given coefficients, evaluate f(r) on an ERP (H, W) grid."""
    if rays is None:
        rays, _ = _erp_grid_rays_and_weights(H, W, coeffs.device, coeffs.dtype)
    Y = real_spherical_harmonics(rays, L_max=L_max)     # (H, W, K)
    # f(h,w) = Σ_k c_k · Y_k(h,w)
    return torch.einsum("...k,hwk->...hw", coeffs, Y)


def sh_heat_filter_coeffs(coeffs: torch.Tensor, tau: torch.Tensor,
                           L_max: int = 4):
    """Multiply SH coefficients by heat-kernel spectral weights
    e^{-ℓ(ℓ+1)τ/2}. Stable for τ ≥ 0 and L_max moderate.
    """
    device = coeffs.device
    dtype = coeffs.dtype
    # Build a (K,) vector of ℓ(ℓ+1).
    ll = []
    for ell in range(L_max + 1):
        ll.extend([ell * (ell + 1)] * (2 * ell + 1))
    ll = torch.tensor(ll, device=device, dtype=dtype)   # (K,)
    # tau is a scalar tensor; allow autograd through it.
    w = torch.exp(-0.5 * ll * tau.to(dtype))            # (K,)
    return coeffs * w

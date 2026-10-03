from typing import Literal
import torch
import torch.nn.functional as F


def _wrap_normalized_longitude(u: torch.Tensor) -> torch.Tensor:
    return torch.remainder(u + 1.0, 2.0) - 1.0


def _wrap_grid_longitude(grid: torch.Tensor) -> torch.Tensor:
    u = grid[..., 0]
    u_wrapped = torch.where(
        (u < -1.0) | (u > 1.0),
        _wrap_normalized_longitude(u),
        u,
    )
    return torch.stack((u_wrapped, grid[..., 1]), dim=-1)


def local_corr_wrapper(
    feature0: torch.Tensor,
    feature1: torch.Tensor,
    coords: torch.Tensor,
    local_window: torch.Tensor,
    B,
    K,
    c,
    r,
    h,
    w,
    device,
    padding_mode="zeros",
    sample_mode: Literal["bilinear", "nearest"] = "bilinear",
    dtype=torch.float32,
    cyclic_horizontal: bool = False,
):
    import local_corr
    assert padding_mode == "zeros"
    warp = (coords[..., None, :] + local_window[:, None, None]).reshape(B, h * w, K, 2)
    if cyclic_horizontal:
        warp = _wrap_grid_longitude(warp)
    corr = (
        local_corr.local_corr(
            feature0.reshape(B, c, h * w).permute(0, 2, 1).float() / (c**0.5),
            feature1.permute(0, 2, 3, 1).clone().detach().float(),
            warp.clone().detach(),
            mode=sample_mode,
            normalized_coords=True,
        )
        .permute(0, 2, 1)
        .reshape(B, K, h, w)
    )
    return corr, warp


def shitty_native_torch_local_corr(
    feature0,
    feature1,
    warp,
    local_window,
    B,
    K,
    c,
    r,
    h,
    w,
    device,
    padding_mode="zeros",
    sample_mode="bilinear",
    dtype=torch.float32,
    cyclic_horizontal: bool = False,
):
    corr = torch.empty((B, K, h, w), device=device, dtype=dtype)
    for _ in range(B):
        with torch.no_grad():
            local_window_coords = (
                warp[_, :, :, None] + local_window[:, None, None]
            ).reshape(1, h, w * K, 2)
            if cyclic_horizontal:
                local_window_coords = _wrap_grid_longitude(local_window_coords)
            window_feature = F.grid_sample(
                feature1[_ : _ + 1],
                local_window_coords,
                padding_mode=padding_mode,
                align_corners=False,
                mode=sample_mode,  #
            )
            window_feature = window_feature.reshape(c, h, w, K)
        corr[_] = (
            (feature0[_, ..., None] / (c**0.5) * window_feature)
            .sum(dim=0)
            .permute(2, 0, 1)
        )
    return corr, None


def local_correlation(
    feature0: torch.Tensor,  # (B x C x H x W)
    feature1: torch.Tensor,  # (B x C x H x W)
    local_radius: int,
    warp: torch.Tensor,  # (B x 2 x H x W)
    *,
    use_custom_corr: bool,
    padding_mode="zeros",
    sample_mode: Literal["bilinear", "nearest"] = "bilinear",
    cyclic_horizontal: bool = False,
):
    r = local_radius
    K = (2 * r + 1) ** 2
    B, c, h, w = feature0.size()
    warp = warp.permute(0, 2, 3, 1)
    device = feature0.device
    dtype = feature0.dtype
    local_window = torch.meshgrid(
        [
            torch.linspace(
                -2 * local_radius / h, 2 * local_radius / h, 2 * r + 1, device=device
            ),
            torch.linspace(
                -2 * local_radius / w, 2 * local_radius / w, 2 * r + 1, device=device
            ),
        ],
        indexing="ij",
    )
    local_window = (
        torch.stack((local_window[1], local_window[0]), dim=-1)[None]
        .expand(1, 2 * r + 1, 2 * r + 1, 2)
        .reshape(1, K, 2)
    )
    if not use_custom_corr:
        corr, corr_coords = shitty_native_torch_local_corr(
            feature0,
            feature1,
            warp,
            local_window,
            B,
            K,
            c,
            r,
            h,
            w,
            device,
            padding_mode,
            sample_mode,
            dtype,
            cyclic_horizontal,
        )
    else:
        corr, corr_coords = local_corr_wrapper(
            feature0,
            feature1,
            warp,
            local_window,
            B,
            K,
            c,
            r,
            h,
            w,
            device,
            padding_mode,
            sample_mode,
            dtype,
            cyclic_horizontal,
        )
    return corr
"""Tilt augmentation wrapper for ERP training pairs (new module, no core edits).

Applies an independent random SO(3) tilt (pitch about x, roll about z; NOT yaw
about the vertical y) to each view of a pair by resampling image+depth on the
sphere, and conjugates the relative pose accordingly:

    im_A' = resample(im_A, R_A),  im_B' = resample(im_B, R_B)
    T_1to2' = G_B @ T_1to2 @ G_A^{-1},   G = blkdiag(R, 1)

Independent R_A/R_B covers the *asymmetric tilt* case the paper currently lists
as untested. Same ERP ray convention as sccm.utils.utils_sphere.
"""
import math
import torch
import torch.nn.functional as F

from sccm.utils.utils_sphere import erp_normalized_to_ray, erp_ray_to_normalized


def _rot_x(t):
    c, s = math.cos(t), math.sin(t)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)


def _rot_z(t):
    c, s = math.cos(t), math.sin(t)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)


def _grid_for_R(Rmat, H, W):
    """grid for F.grid_sample: output pixel dir n' samples input dir R^T n'."""
    vy = torch.linspace(-1, 1, H)
    ux = torch.linspace(-1, 1, W)
    v, u = torch.meshgrid(vy, ux, indexing="ij")
    kn = torch.stack([u, v], dim=-1)
    n_out = erp_normalized_to_ray(kn)
    n_in = torch.einsum("ij,hwj->hwi", Rmat.t(), n_out)
    return erp_ray_to_normalized(n_in)


def _resample(img, grid, mode):
    x = img.unsqueeze(0)
    y = F.grid_sample(x, grid.unsqueeze(0).to(x.dtype), mode=mode,
                      align_corners=True, padding_mode="border")
    return y.squeeze(0)


class TiltAugDataset(torch.utils.data.Dataset):
    """Wrap an ERP pair dataset; apply random per-view tilt on __getitem__."""

    def __init__(self, base, max_tilt_deg=30.0):
        self.base = base
        self.max = math.radians(float(max_tilt_deg))

    def __len__(self):
        return len(self.base)

    def _sample_R(self):
        # pitch (x) + roll (z), each U(-max, max); no yaw (y) — that is handled by yaw_aug.
        a = (torch.rand(1).item() * 2 - 1) * self.max
        b = (torch.rand(1).item() * 2 - 1) * self.max
        return _rot_x(a) @ _rot_z(b)

    def __getitem__(self, i):
        d = dict(self.base[i])
        RA, RB = self._sample_R(), self._sample_R()
        imA, imB = d["im_A"], d["im_B"]
        H, W = imA.shape[-2], imA.shape[-1]
        gA, gB = _grid_for_R(RA, H, W), _grid_for_R(RB, H, W)
        d["im_A"] = _resample(imA, gA, "bilinear")
        d["im_B"] = _resample(imB, gB, "bilinear")
        for key, g, R in (("im_A_depth", gA, RA), ("im_B_depth", gB, RB)):
            dep = d[key]; sh = dep.shape
            gg = _grid_for_R(R, sh[-2], sh[-1])
            d[key] = _resample(dep.reshape(1, *sh[-2:]).float(), gg, "nearest").reshape(sh)
        GA = torch.eye(4); GA[:3, :3] = RA
        GB = torch.eye(4); GB[:3, :3] = RB
        d["T_1to2"] = (GB @ d["T_1to2"].float() @ torch.inverse(GA))
        return d

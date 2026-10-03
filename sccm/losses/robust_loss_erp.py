"""
ERP-specific loss for RoMa.

Differences from RobustLosses (robust_loss.py):
1. Uses get_gt_warp_erp (spherical geometry, no intrinsics K)
2. Latitude-weighted regression: cos(lat) weighting to reduce polar pixel overfit
3. Angular error metrics (PCK in degrees) instead of pixel-space EPE for logging
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import sccm
import wandb
from sccm.utils.utils_sphere import get_gt_warp_erp, erp_normalized_to_ray


class RobustLossesERP(nn.Module):
    def __init__(
        self,
        robust=False,
        center_coords=False,
        scale_normalize=False,
        ce_weight=0.01,
        local_loss=True,
        local_dist=4.0,
        local_largest_scale=8,
        smooth_mask=False,
        depth_interpolation_mode="bilinear",
        mask_depth_loss=False,
        relative_depth_error_threshold=0.05,
        alpha=1.0,
        c=1e-3,
        epipolar_weight: float = 0.0,
        epipolar_scales=None,          # list of scales; default = [1, 2] when weight > 0
        epipolar_cert_gate: bool = True,
        epipolar_use_robust: bool = False,
        lat_weight_enabled: bool = True,   # cos(lat) per-pixel area weighting (ERP). Off → vanilla RoMa loss form.
    ):
        super().__init__()
        self.robust = robust
        self.center_coords = center_coords
        self.scale_normalize = scale_normalize
        self.ce_weight = ce_weight
        self.local_loss = local_loss
        self.local_dist = local_dist
        self.local_largest_scale = local_largest_scale
        self.smooth_mask = smooth_mask
        self.depth_interpolation_mode = depth_interpolation_mode
        self.mask_depth_loss = mask_depth_loss
        self.relative_depth_error_threshold = relative_depth_error_threshold
        self.avg_overlap = dict()
        self.alpha = alpha
        self.c = c
        # Auxiliary spherical-epipolar Sampson residual (off by default).
        self.epipolar_weight = float(epipolar_weight)
        self.epipolar_scales = list(epipolar_scales) if epipolar_scales else [1, 2]
        self.epipolar_cert_gate = bool(epipolar_cert_gate)
        self.epipolar_use_robust = bool(epipolar_use_robust)
        self.lat_weight_enabled = bool(lat_weight_enabled)

    def gm_cls_loss(self, x2, prob, scale_gm_cls, gm_certainty, scale):
        with torch.no_grad():
            B, C, H, W = scale_gm_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(
                *[torch.linspace(-1 + 1 / cls_res, 1 - 1 / cls_res,
                                 steps=cls_res, device=device)
                  for _ in range(2)],
                indexing="ij",
            )
            G = torch.stack((G[1], G[0]), dim=-1).reshape(C, 2)
            GT = (G[None, :, None, None, :] - x2[:, None]).norm(dim=-1).min(dim=1).indices

        cls_loss = F.cross_entropy(scale_gm_cls, GT, reduction="none")[prob > 0.99]
        certainty_loss = F.binary_cross_entropy_with_logits(gm_certainty[:, 0], prob)
        if not torch.any(cls_loss):
            cls_loss = certainty_loss * 0.0

        losses = {
            f"gm_certainty_loss_{scale}": certainty_loss.mean(),
            f"gm_cls_loss_{scale}": cls_loss.mean(),
        }
        wandb.log(losses, step=sccm.GLOBAL_STEP)
        return losses

    def delta_cls_loss(self, x2, prob, flow_pre_delta, delta_cls, certainty, scale, offset_scale):
        with torch.no_grad():
            B, C, H, W = delta_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(
                *[torch.linspace(-1 + 1 / cls_res, 1 - 1 / cls_res,
                                 steps=cls_res, device=device)
                  for _ in range(2)],
            )
            G = torch.stack((G[1], G[0]), dim=-1).reshape(C, 2) * offset_scale
            GT = (G[None, :, None, None, :] + flow_pre_delta[:, None] - x2[:, None]).norm(dim=-1).min(dim=1).indices

        cls_loss = F.cross_entropy(delta_cls, GT, reduction="none")[prob > 0.99]
        certainty_loss = F.binary_cross_entropy_with_logits(certainty[:, 0], prob)
        if not torch.any(cls_loss):
            cls_loss = certainty_loss * 0.0

        losses = {
            f"delta_certainty_loss_{scale}": certainty_loss.mean(),
            f"delta_cls_loss_{scale}": cls_loss.mean(),
        }
        wandb.log(losses, step=sccm.GLOBAL_STEP)
        return losses

    def regression_loss(self, x2, prob, flow, certainty, scale, eps=1e-8, mode="delta"):
        flow_hw = flow.permute(0, 2, 3, 1)  # (B, H, W, 2)
        epe = (flow_hw - x2).norm(dim=-1)

        # Log angular error metrics at finest scale
        if scale == 1:
            b, _, h, w = flow.shape
            mask = prob > 0.99
            if mask.any():
                ray_pred = erp_normalized_to_ray(flow_hw[mask])
                ray_gt = erp_normalized_to_ray(x2[mask])
                dot = (ray_pred * ray_gt).sum(dim=-1).clamp(-1 + eps, 1 - eps)
                ang_deg = torch.acos(dot) * (180.0 / math.pi)
                wandb.log({
                    "train_pck_1deg": (ang_deg < 1.0).float().mean(),
                    "train_pck_3deg": (ang_deg < 3.0).float().mean(),
                    "train_pck_5deg": (ang_deg < 5.0).float().mean(),
                    "train_mae_deg": ang_deg.mean(),
                }, step=sccm.GLOBAL_STEP)

        ce_loss = F.binary_cross_entropy_with_logits(certainty[:, 0], prob)
        a = self.alpha[scale] if isinstance(self.alpha, dict) else self.alpha
        cs = self.c * scale

        # Latitude weighting: cos(lat) — reduces polar pixel contribution.
        # Toggleable via lat_weight_enabled (default True; set False to recover
        # the vanilla RoMa loss form for ablation against perspective-style
        # baselines).
        x = epe[prob > 0.99]
        reg_loss = cs ** a * ((x / cs) ** 2 + 1) ** (a / 2)
        if self.lat_weight_enabled:
            b, _, h, w = flow.shape
            v_norm = torch.linspace(-1 + 1 / h, 1 - 1 / h, h,
                                    device=flow.device, dtype=flow.dtype)
            lat = -(math.pi / 2) * v_norm  # [pi/2, -pi/2]
            lat_weight = torch.cos(lat)     # 1.0 at equator, ~0 at poles
            lat_weight = lat_weight / (lat_weight.mean() + eps)  # normalize mean=1
            lat_weight = lat_weight[None, :, None].expand(b, h, w)  # (B, H, W)
            lw = lat_weight[prob > 0.99]
            reg_loss = reg_loss * lw  # latitude-weighted

        if not torch.any(reg_loss):
            reg_loss = ce_loss * 0.0

        losses = {
            f"{mode}_certainty_loss_{scale}": ce_loss.mean(),
            f"{mode}_regression_loss_{scale}": reg_loss.mean(),
        }
        wandb.log(losses, step=sccm.GLOBAL_STEP)
        return losses

    def forward(self, corresps, batch):
        scales = list(corresps.keys())
        tot_loss = 0.0
        scale_weights = {1: 1, 2: 1, 4: 1, 8: 1, 16: 1}
        coarse_loss_sum = 0.0  # scales 16, 8
        fine_loss_sum = 0.0    # scales 4, 2, 1

        for scale in scales:
            scale_corresps = corresps[scale]
            scale_certainty = scale_corresps["certainty"]
            flow_pre_delta = scale_corresps.get("flow_pre_delta")
            delta_cls = scale_corresps.get("delta_cls")
            offset_scale = scale_corresps.get("offset_scale")
            scale_gm_cls = scale_corresps.get("gm_cls")
            scale_gm_certainty = scale_corresps.get("gm_certainty")
            flow = scale_corresps["flow"]
            scale_gm_flow = scale_corresps.get("gm_flow")

            if flow_pre_delta is not None:
                flow_pre_delta = rearrange(flow_pre_delta, "b d h w -> b h w d")
                b, h, w, d = flow_pre_delta.shape
            else:
                b, _, h, w = scale_certainty.shape

            # ERP GT warp — no K1/K2 needed
            gt_warp, gt_prob = get_gt_warp_erp(
                batch["im_A_depth"],
                batch["im_B_depth"],
                batch["T_1to2"],
                H=h,
                W=w,
                depth_interpolation_mode=self.depth_interpolation_mode,
                relative_depth_error_threshold=self.relative_depth_error_threshold,
                smooth_mask=self.smooth_mask,
            )
            x2 = gt_warp.float()
            prob = gt_prob

            # Log GT overlap statistics per scale
            valid_ratio = (prob > 0.5).float().mean().item()
            wandb.log({f"gt_valid_ratio_{scale}": valid_ratio}, step=sccm.GLOBAL_STEP)

            if self.local_largest_scale >= scale:
                prob = prob * (
                    F.interpolate(prev_epe[:, None], size=(h, w), mode="nearest-exact")[:, 0]
                    < (2 / 512) * (self.local_dist[scale] * scale)
                )

            scale_loss = 0.0

            # GM (Global Matching / Coarse) loss
            if scale_gm_cls is not None:
                gm_cls_losses = self.gm_cls_loss(x2, prob, scale_gm_cls, scale_gm_certainty, scale)
                gm_loss = (self.ce_weight * gm_cls_losses[f"gm_certainty_loss_{scale}"]
                           + gm_cls_losses[f"gm_cls_loss_{scale}"])
                scale_loss = scale_loss + scale_weights[scale] * gm_loss
            elif scale_gm_flow is not None:
                gm_flow_losses = self.regression_loss(x2, prob, scale_gm_flow, scale_gm_certainty, scale, mode="gm")
                gm_loss = (self.ce_weight * gm_flow_losses[f"gm_certainty_loss_{scale}"]
                           + gm_flow_losses[f"gm_regression_loss_{scale}"])
                scale_loss = scale_loss + scale_weights[scale] * gm_loss

            # Delta (Local Refinement / Fine) loss
            if delta_cls is not None:
                delta_cls_losses = self.delta_cls_loss(x2, prob, flow_pre_delta, delta_cls, scale_certainty, scale, offset_scale)
                delta_loss = (self.ce_weight * delta_cls_losses[f"delta_certainty_loss_{scale}"]
                              + delta_cls_losses[f"delta_cls_loss_{scale}"])
                scale_loss = scale_loss + scale_weights[scale] * delta_loss
            else:
                delta_reg_losses = self.regression_loss(x2, prob, flow, scale_certainty, scale)
                reg_loss = (self.ce_weight * delta_reg_losses[f"delta_certainty_loss_{scale}"]
                            + delta_reg_losses[f"delta_regression_loss_{scale}"])
                scale_loss = scale_loss + scale_weights[scale] * reg_loss

            # Auxiliary: spherical-epipolar Sampson residual (off when weight=0).
            if self.epipolar_weight > 0 and int(scale) in self.epipolar_scales:
                from sccm.losses.epipolar_consistency import epipolar_sampson_residual
                # `flow` is (B, 2, H, W) warp in normalised ERP coords — exactly what
                # the residual expects. `scale_certainty` is logits (B, 1, H, W).
                epi_loss, n_eff = epipolar_sampson_residual(
                    warp=flow,
                    T_1to2=batch["T_1to2"],
                    certainty_logits=scale_certainty,
                    valid_mask=(prob > 0.5),
                    cert_gate=self.epipolar_cert_gate,
                    use_robust=self.epipolar_use_robust,
                )
                scale_loss = scale_loss + self.epipolar_weight * epi_loss
                wandb.log({
                    f"epipolar_loss_{scale}": epi_loss.item(),
                    f"epipolar_n_eff_{scale}": float(n_eff.item()),
                }, step=sccm.GLOBAL_STEP)

            tot_loss = tot_loss + scale_loss

            # Track coarse vs fine
            if scale >= 8:
                coarse_loss_sum = coarse_loss_sum + scale_loss
            else:
                fine_loss_sum = fine_loss_sum + scale_loss

            # Per-scale EPE and total loss
            epe_scale = (flow.permute(0, 2, 3, 1) - x2).norm(dim=-1)
            epe_valid = epe_scale[prob > 0.99]
            wandb.log({
                f"loss/scale_{scale}": scale_loss.item() if torch.is_tensor(scale_loss) else scale_loss,
                f"epe/scale_{scale}": epe_valid.mean().item() if epe_valid.numel() > 0 else 0,
                f"n_valid/scale_{scale}": (prob > 0.99).sum().item(),
            }, step=sccm.GLOBAL_STEP)

            prev_epe = epe_scale.detach()

        # Summary: coarse vs fine, total
        wandb.log({
            "loss/total": tot_loss.item() if torch.is_tensor(tot_loss) else tot_loss,
            "loss/coarse_s16_s8": coarse_loss_sum.item() if torch.is_tensor(coarse_loss_sum) else coarse_loss_sum,
            "loss/fine_s4_s2_s1": fine_loss_sum.item() if torch.is_tensor(fine_loss_sum) else fine_loss_sum,
        }, step=sccm.GLOBAL_STEP)

        return tot_loss

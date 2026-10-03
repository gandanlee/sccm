"""Model builder for the paper configurations.

This is the paper-path extract of the original research codebase's model factory.
The research code carried ~50 config-gated variants; this module keeps only the
paper path. It builds one of four coarse stages, which the ten shipped configs
select between:

  * ``roma_v1_gp``   -- RoMa V1 Gaussian-process coarse stage, retrained on ERP
  * ``chart_naive``  -- cross-attention + dual-softmax scaffold, no sphere prior (R1)
  * ``spa_only``     -- R1 + Spherical Positional Attention          (R2b)
  * ``sccm``         -- R1 + SPA + Area-Aware Covisibility           (R3)

The constructor arguments below were captured from the original factory while it
built each published config, so the resulting modules are parameter-identical to
the ones used for the reported numbers and the released checkpoints load with
``strict=True``. ``tests/test_paper_models.py`` pins both properties.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import yaml

from sccm.models.encoders import CNNandDinov2
from sccm.models.matcher import GP, ConvRefiner, CosKernel, Decoder, RegressionMatcher
from sccm.models.sphere_covis_matcher import SphereCovisMatcher
from sccm.models.transformer import Block, MemEffAttention, TransformerDecoder

# Training/evaluation grids, as used by the research codebase. All published
# runs use "medium"; the RoPE table size is derived from it as (H/14, W/14).
RESOLUTIONS = {
    "low": (336, 672),
    "medium": (448, 896),
    "high": (512, 1024),
}

GP_DIM = 512
FEAT_DIM = 512
DECODER_DIM = GP_DIM + FEAT_DIM
CLS_TO_COORD_RES = 64


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _conv_refiners(amp: bool = True) -> nn.ModuleDict:
    """The five-scale refinement cascade. Identical across all four configs."""
    common = dict(
        kernel_size=5,
        dw=True,
        hidden_blocks=8,
        displacement_emb="linear",
        amp=amp,
        disable_local_corr_grad=True,
        bn_momentum=0.01,
    )
    return nn.ModuleDict(
        {
            "16": ConvRefiner(
                1377, 1377, 3, displacement_emb_dim=128,
                local_corr_radius=7, corr_in_other=True, **common,
            ),
            "8": ConvRefiner(
                1137, 1137, 3, displacement_emb_dim=64,
                local_corr_radius=3, corr_in_other=True, **common,
            ),
            "4": ConvRefiner(
                569, 569, 3, displacement_emb_dim=32,
                local_corr_radius=2, corr_in_other=True, **common,
            ),
            "2": ConvRefiner(144, 144, 3, displacement_emb_dim=16, **common),
            "1": ConvRefiner(24, 24, 3, displacement_emb_dim=6, **common),
        }
    )


def _projections() -> nn.ModuleDict:
    """1x1 projections from the VGG19 feature pyramid to the refiner widths.

    Channel counts (64, 128, 256, 512) are the VGG19 stage outputs; scale "16"
    is fed by the DINOv2 patch features (1024).
    """
    c1, c2, c4, c8 = 64, 128, 256, 512
    return nn.ModuleDict(
        {
            "16": nn.Sequential(nn.Conv2d(1024, 512, 1, 1), nn.BatchNorm2d(512)),
            "8": nn.Sequential(nn.Conv2d(c8, 512, 1, 1), nn.BatchNorm2d(512)),
            "4": nn.Sequential(nn.Conv2d(c4, 256, 1, 1), nn.BatchNorm2d(256)),
            "2": nn.Sequential(nn.Conv2d(c2, 64, 1, 1), nn.BatchNorm2d(64)),
            "1": nn.Sequential(nn.Conv2d(c1, 9, 1, 1), nn.BatchNorm2d(9)),
        }
    )


def _coordinate_decoder(amp: bool = True) -> TransformerDecoder:
    return TransformerDecoder(
        nn.Sequential(
            *[Block(DECODER_DIM, 8, attn_class=MemEffAttention) for _ in range(5)]
        ),
        DECODER_DIM,
        CLS_TO_COORD_RES**2 + 1,
        is_classifier=True,
        amp=amp,
        pos_enc=False,
    )


def _coarse_gp_v1() -> GP:
    """RoMa V1 Gaussian-process coarse matcher."""
    return GP(
        CosKernel,
        T=0.2,
        learn_temperature=False,
        only_attention=False,
        gp_dim=GP_DIM,
        basis="fourier",
        no_cov=True,
    )


def _coarse_sphere_covis(cfg: dict, resolution: str) -> SphereCovisMatcher:
    """Chart-naive / SPA / SCCM coarse matcher.

    A single class with three prior switches; the paper's three ablation rows
    differ only in which switches are on:

        chart-naive : all off
        SPA only    : rope + tangent attention (incl. LST)
        SCCM        : rope + tangent attention + jacobian (AAC / log-area)
    """
    cg = cfg.get("model", {}).get("covis_gated_matching", {})
    sph = cg.get("sphere_covis", {})
    rope = sph.get("rope", {})
    jac = sph.get("jacobian", {})
    tan = sph.get("tangent_attention", {})
    train_h, train_w = RESOLUTIONS.get(resolution, RESOLUTIONS["medium"])

    return SphereCovisMatcher(
        in_dim=FEAT_DIM,
        dim=cg.get("dim", GP_DIM),
        n_cross_blocks=cg.get("n_cross_blocks", 4),
        n_heads=cg.get("n_heads", 8),
        ffn_ratio=cg.get("ffn_ratio", 4),
        temp=cg.get("temp", 0.1),
        sinkhorn_iters=cg.get("sinkhorn_iters", 5),
        matching_mode=cg.get("matching_mode", "dual_softmax"),
        covis_hidden=cg.get("covis_hidden", 128),
        covis_init_logit=cg.get("covis_init_logit", 4.0),
        # --- AAC (Area-Aware Covisibility): pre-sigmoid log-area correction ---
        jacobian_enabled=jac.get("enabled", False),
        jacobian_alpha_init=jac.get("alpha_init", 1.0),
        jacobian_alpha_learnable=jac.get("alpha_learnable", True),
        # --- SPA part 1: yaw-periodic RoPE (integer longitudinal harmonics) ---
        rope_enabled=rope.get("enabled", False),
        rope_H=rope.get("H", train_h // 14),
        rope_W=rope.get("W", train_w // 14),
        rope_lon_int_freq_max=rope.get("lon_int_freq_max", None),
        rope_lat_theta_base=rope.get("lat_theta_base", 10000.0),
        rope_lon_geometric=rope.get("lon_geometric", False),
        rope_gain_random_range=rope.get("gain_random_range", None),
        rope_gain_eval=rope.get("gain_eval", 1.0),
        # --- SPA part 2: tangent-plane bias (+ latitude-scaled tangent) ---
        tangent_attn_enabled=tan.get("enabled", False),
        tangent_attn_n_freqs=tan.get("n_freqs", 8),
        tangent_attn_hidden_dim=tan.get("hidden_dim", 32),
        tangent_attn_zero_init=tan.get("zero_init", True),
        tangent_attn_conformal_enabled=tan.get("conformal_enabled", False),
        tangent_attn_conformal_alpha_init=tan.get("conformal_alpha_init", 0.0),
        # --- input-side absolute PE control (Tab. 2, "ctrl: EDM abs. PE") ---
        edm_pe_enabled=sph.get("edm_pe", {}).get("enabled", False),
        edm_pe_zero_init=sph.get("edm_pe", {}).get("zero_init", True),
    )


def get_model(cfg: dict, resolution: str = "medium", **kwargs) -> RegressionMatcher:
    """Build the matcher described by ``cfg``.

    ``cfg['model']['covis_gated_matching']`` selects the attention-based coarse
    stage (chart-naive / SPA / SCCM); its absence selects the RoMa V1 GP stage.
    """
    warnings.filterwarnings(
        "ignore", category=UserWarning, message="TypedStorage is deprecated"
    )
    model_cfg = cfg.get("model", {})
    pretrained = model_cfg.get("pretrained_backbone", True)
    amp = True

    if model_cfg.get("covis_gated_matching"):
        coarse = _coarse_sphere_covis(cfg, resolution)
    else:
        coarse = _coarse_gp_v1()

    decoder = Decoder(
        _coordinate_decoder(amp),
        nn.ModuleDict({"16": coarse}),   # gps  — coarse matcher
        _projections(),                  # proj — pyramid projections
        _conv_refiners(amp),             # conv_refiner — 5-scale cascade
        detach=True,
        scales=["16", "8", "4", "2", "1"],
    )

    encoder = CNNandDinov2(
        cnn_kwargs=dict(pretrained=pretrained, amp=amp),
        amp=amp,
    )

    h, w = RESOLUTIONS.get(resolution, RESOLUTIONS["medium"])
    return RegressionMatcher(
        encoder,
        decoder,
        h=h,
        w=w,
        attenuate_cert=model_cfg.get("attenuate_cert", False),
    )


def load_checkpoint(model: nn.Module, path: str, strict: bool = True) -> nn.Module:
    """Load a released checkpoint into a model built by :func:`get_model`."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    model.load_state_dict(state, strict=strict)
    return model

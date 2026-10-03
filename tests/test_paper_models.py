"""Regression tests pinning the published models.

These guard the two properties that matter for reproducibility:

  T1  each paper configuration builds a model with the exact parameter count of
      the model used to produce the reported numbers;
  T2  every released checkpoint loads into its configuration with strict=True.

Run with:  pytest tests/ -q          (requires the checkpoints in checkpoints/)
"""

import pathlib

import pytest
import torch

from sccm.build import get_model, load_checkpoint, load_config

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Parameter counts captured from the original research codebase's model factory
# while it built each published configuration.
EXPECTED_PARAMS = {
    "chart_naive": 137_165_330,
    "spa_only": 137_166_420,
    "sccm": 137_166_422,
    "roma_v1_gp": 111_288_336,
}

CONFIGS = {
    "chart_naive": "configs/mp3d/chart_naive.yaml",
    "spa_only": "configs/mp3d/spa_only.yaml",
    "sccm": "configs/mp3d/sccm.yaml",
    "roma_v1_gp": "configs/mp3d/roma_v1_gp.yaml",
}

CHECKPOINTS = [
    ("configs/mp3d/sccm.yaml", "checkpoints/mp3d/sccm.pth"),
    ("configs/holo360d/sccm_ft.yaml", "checkpoints/holo360d/sccm.pth"),
]


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_parameter_count(name):
    model = get_model(load_config(ROOT / CONFIGS[name]))
    n = sum(p.numel() for p in model.parameters())
    assert n == EXPECTED_PARAMS[name], f"{name}: {n} != {EXPECTED_PARAMS[name]}"


@pytest.mark.parametrize("cfg,ckpt", CHECKPOINTS, ids=[c for _, c in CHECKPOINTS])
def test_checkpoint_loads_strict(cfg, ckpt):
    path = ROOT / ckpt
    if not path.exists():
        pytest.skip(f"{ckpt} not downloaded (see docs/CHECKPOINTS.md)")
    model = get_model(load_config(ROOT / cfg))
    load_checkpoint(model, str(path), strict=True)


ALL_CONFIGS = sorted(
    str(p.relative_to(ROOT)) for p in (ROOT / "configs").rglob("*.yaml")
)


@pytest.mark.parametrize("cfg", ALL_CONFIGS, ids=ALL_CONFIGS)
def test_every_shipped_config_builds(cfg):
    """No config in the repo may reference a switch build.py cannot honour."""
    model = get_model(load_config(ROOT / cfg))
    assert sum(p.numel() for p in model.parameters()) > 0


def test_priors_are_switchable():
    """chart-naive < SPA < SCCM in parameter count, by the prior modules only."""
    p = {k: sum(q.numel() for q in get_model(load_config(ROOT / v)).parameters())
         for k, v in CONFIGS.items()}
    assert p["chart_naive"] < p["spa_only"] < p["sccm"]
    assert p["sccm"] - p["spa_only"] == 2, "AAC adds exactly two scalars"

# Reproducing the paper

Every trained row in the paper maps to one config in `configs/` and one command
below; the SCCM rows also have released checkpoints in `checkpoints/`, while the
baseline rows are retrained from their configs (see Training). Numbers are PCK@1° unless stated.

## Setup

```bash
pip install -e .
export SCCM_DATA_ROOT=/path/to/datasets
```

Datasets are not redistributed here. See [Datasets](#datasets) for how each one
is obtained and prepared.

## Metrics

All metrics are **angular**, computed on the unit sphere between the predicted
and ground-truth ray directions. Pixel end-point error is deliberately not
reported: ERP pixel distance is anisotropic in latitude by a factor of cos φ, so
a pixel-space average silently down-weights the poles.

| Metric | Meaning |
|---|---|
| PCK@τ° | fraction of valid pixels whose angular error is below τ |
| MAE° | mean angular error, sensitive to the error tail |
| Med.° | median angular error |

Evaluation covers every pixel with valid ground-truth depth in both views.
Matterport3D test is 15,682 pairs, Stanford2D3D test 8,744, Holo360D test 8,000.

## Table 1(a) — Matterport3D test, in-distribution

```bash
torchrun --nproc_per_node=8 scripts/eval_precision.py \
    --config configs/mp3d/sccm.yaml \
    --checkpoint checkpoints/mp3d/sccm.pth \
    --data_root $SCCM_DATA_ROOT/matterport3d_megadepth \
    --split test --thresholds 0.35,0.5,1,3,5 --dont_log_wandb
```

| Row | Config | Checkpoint | PCK@1° | MAE° |
|---|---|---|---|---|
| RoMa V1 (ERP-retrained) | `configs/mp3d/roma_v1_gp.yaml` | retrain | 0.198 | 6.60 |
| chart-naïve | `configs/mp3d/chart_naive.yaml` | retrain | 0.229 | 5.68 |
| **SCCM** | `configs/mp3d/sccm.yaml` | `checkpoints/mp3d/sccm.pth` | **0.275** | **5.36** |

The remaining rows of Tab. 1(a) are external baselines evaluated from their
authors' released weights, under the same evaluator. See
[External baselines](#external-baselines).

## Table 1(b) — Stanford2D3D test, zero-shot

Same checkpoints, no retraining. Only the data root and split change.

```bash
torchrun --nproc_per_node=8 scripts/eval_precision.py \
    --config configs/mp3d/sccm.yaml \
    --checkpoint checkpoints/mp3d/sccm.pth \
    --data_root $SCCM_DATA_ROOT/stanford2d3d_megadepth \
    --split test --thresholds 0.35,0.5,1,3,5 --dont_log_wandb
```

| Row | PCK@1° | MAE° |
|---|---|---|
| RoMa V1 (ERP-retrained) | 0.167 | 17.80 |
| chart-naïve | 0.179 | 18.86 |
| **SCCM** | **0.229** | **17.09** |

## Table 1(c) and Supp. Sec. P — Holo360D test, outdoor

Each row is initialized from its Matterport3D checkpoint and trained on Holo360D, then evaluated.

```bash
torchrun --nproc_per_node=8 scripts/eval_precision.py \
    --config configs/holo360d/sccm_ft.yaml \
    --checkpoint checkpoints/holo360d/sccm.pth \
    --data_root $SCCM_DATA_ROOT/holo360d_megadepth \
    --split test --thresholds 0.35,0.5,1,3,5 --dont_log_wandb
```

| Row | Config | Checkpoint | PCK@1° | MAE° | Med.° |
|---|---|---|---|---|---|
| RoMa V1 (ERP-retrained) | `configs/holo360d/roma_v1_gp_ft.yaml` | retrain | 0.322 | 5.66 | 2.12 |
| chart-naïve | `configs/holo360d/chart_naive_ft.yaml` | retrain | 0.331 | 5.02 | 1.93 |
| **SCCM** | `configs/holo360d/sccm_ft.yaml` | `checkpoints/holo360d/sccm.pth` | **0.357** | **4.94** | **1.76** |

`configs/holo360d/sccm_ft.yaml` carries one component the two baseline rows do
not: a stochastic gain on the RoPE rotation angles during Holo360D training, fixed to
1.0 at evaluation. The header of that file explains it. It has no counterpart in
the baselines because they have no RoPE, and it is not used anywhere in
Tab. 1(a), Tab. 1(b), or Tab. 2.

## Table 2 — controlled ablation on the fixed scaffold

All seven rows share one training protocol and differ only in which sphere
priors are switched on. Only the SCCM checkpoint (R3) is released; the other rows are reproduced by
retraining from their configs.

| Row | Config | Checkpoint | PCK@1° | MAE° |
|---|---|---|---|---|
| R1 chart-naïve, no PE | `configs/mp3d/chart_naive.yaml` | retrain | 0.229 | 5.68 |
| ctrl: EDM absolute PE | `configs/mp3d/ctrl_edm_pe.yaml` | retrain | 0.224 | 5.97 |
| ctrl: standard RoPE | `configs/mp3d/ctrl_standard_rope.yaml` | retrain | 0.236 | 6.79 |
| R2a + yaw-periodic RoPE | `configs/mp3d/rope_only.yaml` | retrain | 0.267 | 5.87 |
| R2b + tangent bias, full SPA | `configs/mp3d/spa_only.yaml` | retrain | 0.266 | 5.58 |
| **R3 + log-area, = SCCM** | `configs/mp3d/sccm.yaml` | `checkpoints/mp3d/sccm.pth` | **0.275** | **5.36** |

## Supp. Sec. R — training-seed variance

Three seeds per configuration. Only the seed-1 checkpoints (the main-table
files) are released; the other seeds can be retrained with `scripts/train.py --seed <n>`.

| Configuration | seed 1 | seed 2 | seed 3 | s.d. |
|---|---|---|---|---|
| chart-naïve | 0.229 | 0.230 | 0.223 | 0.4 pp |
| SCCM | 0.275 | 0.283 | 0.272 | 0.6 pp |
| margin | +4.6 pp | +5.3 pp | +4.9 pp | |

The per-configuration seed standard deviation, 0.4 to 0.6 pp, is what the paper
compares its margins against. Treat any difference under about 1 pp from a single seed as noise.

## Training

Every Matterport3D row uses one protocol. Eight GPUs, 125,000 optimizer steps,
which is 1M samples, about 22 hours on 8× RTX 4090.

```bash
torchrun --nproc_per_node=8 scripts/train.py \
    --config configs/mp3d/sccm.yaml \
    --data_root $SCCM_DATA_ROOT/matterport3d_megadepth \
    --gpu_batch_size 1 --train_resolution medium \
    --num_steps 125000 --lr_scale 1 \
    --eval_every_steps 2048 --val_max_pairs 999999
```

Holo360D training uses the same command with the Holo360D config and
`--num_steps 25000 --lr_scale 0.1`. The starting checkpoint comes from
`training.init_checkpoint` in the config, as a model-only warm start.

Only the frozen DINOv2-L backbone is pretrained; everything else is randomly
initialised. Resuming is automatic: rerun the same command and the trainer picks
up the model, optimizer, and scheduler state from `training.checkpoint_dir`.

## Datasets

| Dataset | Source | Preparation |
|---|---|---|
| Matterport3D | [official release](https://niessner.github.io/Matterport/) (request form) | see below |
| Stanford2D3D | [official release](http://buildingparser.stanford.edu/dataset.html) | `scripts/data/stanford2d3d/preprocess.py --by_corpus` |
| Holo360D | authors' release | `scripts/data/build_holo360d.py`, then `build_holo360d_postproc.py` |

All three are converted to the same MegaDepth-style layout: ERP images at
1024×512, per-pixel depth, camera poses, and a pair list with a precomputed
overlap score. The ERP loader is `sccm/datasets/megadepth_erp.py`.

**Matterport3D has no preprocessing script in this repository.** It was prepared
with an internal pipeline that we cannot release as-is. The split is the
official 90-scene benchmark split: 61 train scenes, 4,625 val pairs, 15,682 test
pairs. If you need the exact pair lists we used rather than your own, open an
issue and we will publish the scene-info arrays.

Stanford2D3D is split by area, matching the Matterport3D convention: areas 1–4
train, areas 5a and 5b validation, area 6 test. One truncated panorama in area 3
is dropped, leaving 1,412. The `--by_corpus` flag is required.

Holo360D is split scene-disjointly into 5 train, 3 validation, and 4 test
scenes. Test pairs are sampled at 2,000 per test scene with overlap in
[0.3, 0.8].

## External baselines

The remaining rows of Tab. 1 come from the baselines' own released weights,
scored with the same angular evaluator on the same pairs. Nothing was retrained.

| Baseline | Weights | Note |
|---|---|---|
| EDM | authors' `edm_matterport3d` TorchScript checkpoint | run with `scripts/eval_edm.py`; evaluated at its native 320×640, see Supp. Sec. R |
| SphereGlue | authors' SuperPoint variant | sparse matcher, densified for scoring |
| RoMa V1, RoMa V2 | authors' released weights | perspective-trained, applied to ERP unchanged |

## Verifying your install

```bash
pip install pytest && pytest tests/ -q
```

This checks that each configuration builds a model with the exact parameter
count of the one used for the paper, and that every checkpoint present in
`checkpoints/` loads with `strict=True`. It needs no dataset and no GPU.

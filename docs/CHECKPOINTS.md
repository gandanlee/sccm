# Checkpoints

Two SCCM checkpoints cover the main results of the paper. They are hosted on
Hugging Face Hub at **[gandan-lee/sccm](https://huggingface.co/gandan-lee/sccm)**
and are not tracked by git. Checksums are in `docs/checkpoints.md5`.

```bash
pip install huggingface_hub
hf download gandan-lee/sccm --local-dir checkpoints      # both files, 1.1 GB
```

Each file is a `torch.save` dict with the weights under the `model` key. Load
one with

```python
from sccm.build import get_model, load_config, load_checkpoint
model = get_model(load_config("configs/mp3d/sccm.yaml"))
load_checkpoint(model, "checkpoints/mp3d/sccm.pth")      # strict=True
```

The released files contain the model weights only; the Adam optimizer state,
which is needed only to resume training, has been removed. The weights are
tensor-for-tensor identical to the training checkpoints that produced the
paper numbers.

## Indoor — Matterport3D (Tab. 1a, Tab. 1b, Tab. 2)

| File | Config | Paper rows | Size | MD5 |
|---|---|---|---|---|
| `mp3d/sccm.pth` | `configs/mp3d/sccm.yaml` | Tab. 1a/1b **SCCM**, Tab. 2 **R3** | 0.55 GB | `3fbae6188b877111ad568dcaf5449c7f` |

The same checkpoint produces the zero-shot Stanford2D3D numbers (indoor, Tab. 1b) —
no separate training.

## Outdoor — Holo360D training (Tab. 1c, Supp. Sec. P)

Initialized from the Matterport3D checkpoint above and trained under one shared
protocol (25k steps, lr scale 0.1, scene-disjoint 5/3/4 splits).

| File | Config | Paper row | Size | MD5 |
|---|---|---|---|---|
| `holo360d/sccm.pth` | `configs/holo360d/sccm_ft.yaml` | Tab. 1c **SCCM** | 0.55 GB | `2c09f10158d56a542e1e5960dea69dff` |

## Not released

- **Baseline rows (chart-naïve scaffold, ERP-retrained RoMa V1; Tab. 1a–c, Tab. 2 R1)** —
  retrain from their configs (`configs/mp3d/chart_naive.yaml`, `configs/mp3d/roma_v1_gp.yaml`,
  and the `configs/holo360d/*_ft.yaml` counterparts) with `scripts/train.py`; the
  parameter counts are pinned in `tests/`.
- **Training-seed runs (Supp. Sec. R, Tab. 15)** — only the seed-1 runs, which are
  the main-table files above, are released. The other seeds can be retrained with
  `scripts/train.py --seed <n>`.
- **SPA-only (Tab. 2, R2b)** — the checkpoint was not retained; the configuration
  is provided (`configs/mp3d/spa_only.yaml`) and can be retrained with
  `scripts/train.py`.
- **DINOv2-L backbone** — downloaded on first use from the official release; we do
  not redistribute it.
- **External baselines** (EDM, SphereGlue, RoMa V2, LoFTR, DKM, MASt3R, VGGT) — use
  the authors' released weights; see `docs/REPRODUCE.md`.

## Verifying a download

```bash
md5sum -c docs/checkpoints.md5
```

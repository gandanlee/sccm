#!/usr/bin/env python3
"""SCCM training driver (paper path).

This is the slim, public extract of the research training script that produced
the numbers in the paper. It supports exactly the configurations shipped in
``configs/`` — the RoMa V1 GP baseline retrained on ERP, the chart-naive
scaffold, SPA, SCCM, and the small positional-encoding controls — all of which
differ only in the model flags read by :mod:`sccm.build`. Every research branch
that no shipped config activates (alternative backbones, cascade/recurrent
refiners, auxiliary loss wrappers, per-band eval diagnostics, ...) has been
removed; the arithmetic of what remains — learning rates, LR schedule, grad
clipping, seeding, sampler, DDP wrapping, checkpoint/resume — is unchanged.

Example (8 GPUs, the 1M-sample budget used for the ablation table)::

    torchrun --nproc_per_node=8 --master_port=29508 scripts/train.py \\
      --config configs/mp3d/sccm.yaml \\
      --data_root /mnt/datasets/matterport3d_megadepth \\
      --gpu_batch_size 1 \\
      --train_resolution medium \\
      --num_steps 125000 \\
      --lr_scale 1 \\
      --eval_every_steps 2048 \\
      --val_max_pairs 999999 \\
      --wandb_project sccm \\
      --wandb_run_id sccm_mp3d

``training.checkpoint_dir`` and ``training.init_checkpoint`` in the configs are
repo-relative; they are resolved against the repository root (the parent of
``scripts/``) unless they are already absolute paths.
"""

import math
import os
import random
import sys
from argparse import ArgumentParser
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset

import wandb

# Allow `python scripts/train.py` from a checkout without installing the package.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import sccm  # noqa: E402  (must follow the sys.path fix-up)
from sccm.build import RESOLUTIONS, get_model, load_config  # noqa: E402
from sccm.checkpointing import CheckPoint  # noqa: E402
from sccm.datasets.megadepth_erp import ERPBuilder  # noqa: E402
from sccm.losses.robust_loss_erp import RobustLossesERP  # noqa: E402
from sccm.train.train import train_k_steps  # noqa: E402
from sccm.utils.utils_sphere import erp_normalized_to_ray, get_gt_warp_erp  # noqa: E402


def resolve_path(path):
    """Resolve a config path against the repository root when it is relative."""
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def visualize_matches(model, batch, prefix="train", num_vis=4):
    """Generate match visualizations and log to wandb."""
    model.eval()
    with torch.no_grad():
        im_A = batch["im_A"][:num_vis].cuda()
        im_B = batch["im_B"][:num_vis].cuda()
        B = im_A.shape[0]

        matches, certainty = (model.module if hasattr(model, 'module') else model).match(
            im_A, im_B, batched=True,
        )
        # matches: (B, H, W, 4) — [x1_n, y1_n, x2_n, y2_n]
        # certainty: (B, H, W)

        images = []
        for b in range(B):
            # Warp image B to image A using predicted flow
            flow_b = matches[b, :, :, 2:].permute(2, 0, 1).unsqueeze(0)  # (1, 2, H, W)
            im_B_warped = F.grid_sample(
                im_B[b:b+1], flow_b.permute(0, 2, 3, 1),
                mode="bilinear", align_corners=False,
            )
            cert_b = certainty[b].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            # Denormalize images
            mean = torch.tensor([0.485, 0.456, 0.406], device=im_A.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=im_A.device).view(1, 3, 1, 1)
            im_A_vis = (im_A[b:b+1] * std + mean).clamp(0, 1)
            im_B_vis = (im_B[b:b+1] * std + mean).clamp(0, 1)
            warp_vis = (im_B_warped * std + mean).clamp(0, 1)

            # Blend: certainty * warp + (1-cert) * white
            white = torch.ones_like(warp_vis)
            cert_norm = cert_b.clamp(0, 1)
            blend = cert_norm * warp_vis + (1 - cert_norm) * white

            # Concatenate: [im_A | im_B | warp_blend]
            row = torch.cat([im_A_vis, im_B_vis, blend], dim=3)  # (1, 3, H, 3*W)
            row_np = (row[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            images.append(wandb.Image(row_np, caption=f"{prefix}_pair_{b}"))

    wandb.log({f"{prefix}_matches": images}, step=sccm.GLOBAL_STEP)
    model.train(True)


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_val(model, val_dataset, batch_size=2, num_workers=4, max_pairs=256, eps=1e-8,
                 rank=0, world_size=1):
    """DDP-aware evaluation. Each rank processes a strided subset of the val
    pairs, then the per-rank statistics are all_reduced onto rank 0.
    Returns metrics dict on rank 0, empty dict on other ranks. Numerically
    equivalent to single-GPU eval (same indices, just split across ranks)."""
    model.eval()
    inner = model.module if hasattr(model, 'module') else model
    h, w = inner.h_resized, inner.w_resized

    # Deterministic full index list (identical across ranks).
    n = len(val_dataset)
    if n > max_pairs:
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(42))[:max_pairs].tolist()
    else:
        indices = list(range(n))
    # Strided slice for this rank: rank r takes indices[r::world_size].
    rank_indices = indices[rank::world_size]
    val_subset = torch.utils.data.Subset(val_dataset, rank_indices)

    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    all_ang_local = []

    def _angular_error(pred, gt, mask):
        ray_pred = erp_normalized_to_ray(pred[mask])
        ray_gt = erp_normalized_to_ray(gt[mask])
        dot = (ray_pred * ray_gt).sum(dim=-1).clamp(-1 + eps, 1 - eps)
        return torch.acos(dot) * (180.0 / math.pi)

    for batch in val_loader:
        im_A = batch["im_A"].cuda()
        im_B = batch["im_B"].cuda()

        matches, certainty = inner.match(im_A, im_B, batched=True)
        # matches: (B, H, W, 4), certainty: (B, H, W)
        flow_pred = matches[:, :, :, 2:]  # (B, H, W, 2) predicted coords in B

        # GT warp
        gt_warp, gt_prob = get_gt_warp_erp(
            batch["im_A_depth"], batch["im_B_depth"], batch["T_1to2"],
            H=h, W=w,
        )
        gt_warp = gt_warp.float().cuda()
        gt_prob = gt_prob.cuda()

        mask = gt_prob > 0.99
        if not mask.any():
            continue

        ang_deg = _angular_error(flow_pred, gt_warp, mask)
        all_ang_local.append(ang_deg.cpu())

    model.train(True)

    local = torch.cat(all_ang_local) if all_ang_local else torch.empty(0)

    # Memory-safe cross-rank reduction. Gathering every per-pixel error
    # (all_gather_object on `local`) OOMs at full test-set scale (~11 GiB on
    # rank 0 for 15K pairs). Instead reduce to scalar sums (exact pck/mae/n)
    # plus a histogram for the median (exact to bin width = 0.01 deg).
    _MED_BINS = 18000  # 0..180 deg @ 0.01 deg
    rdev = torch.device(f"cuda:{torch.cuda.current_device()}")
    lc = local.to(rdev).float()
    stat = torch.tensor([
        float((lc < 0.35).sum()), float((lc < 0.5).sum()),
        float((lc < 1.0).sum()), float((lc < 3.0).sum()),
        float((lc < 5.0).sum()), float(lc.sum()), float(lc.numel()),
    ], device=rdev, dtype=torch.float64)
    hist = torch.histc(lc, bins=_MED_BINS, min=0.0, max=180.0).double()
    del lc
    if world_size > 1:
        dist.all_reduce(stat, op=dist.ReduceOp.SUM)
        dist.all_reduce(hist, op=dist.ReduceOp.SUM)

    if rank != 0:
        return {}

    n = stat[6].item()
    if n <= 0:
        return {}
    cdf = torch.cumsum(hist, 0)
    med_idx = int(torch.searchsorted(cdf, torch.tensor(n / 2.0, device=cdf.device, dtype=torch.float64)).item())
    med_idx = min(med_idx, _MED_BINS - 1)
    median_deg = (med_idx + 0.5) * (180.0 / _MED_BINS)
    return {
        "val/pck_0p35deg": stat[0].item() / n,
        "val/pck_0p5deg": stat[1].item() / n,
        "val/pck_1deg": stat[2].item() / n,
        "val/pck_3deg": stat[3].item() / n,
        "val/pck_5deg": stat[4].item() / n,
        "val/mae_deg": stat[5].item() / n,
        "val/median_deg": median_deg,
        "val/n_pixels": int(n),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    cfg = load_config(args.config)
    # NCCL collective op timeout: rank-0 evaluates val on the full set (~7-8 min);
    # other ranks block on the next AllReduce. Default 10 min was borderline →
    # extend to 1 hour for safety against per-eval slowdowns.
    dist.init_process_group("nccl", timeout=timedelta(hours=1))
    gpus = int(os.environ["WORLD_SIZE"])
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    sccm.LOCAL_RANK = device_id
    sccm.RANK = rank
    torch.cuda.set_device(device_id)

    # Reproducibility: fix all random seeds
    seed = args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 42)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Rank {rank}] GPU {device_id}, config: {args.config}, seed: {seed + rank}")

    # ── WandB ──
    wandb_cfg = cfg.get("wandb", {})
    wandb_base = wandb_cfg.get("base_url", "http://localhost:8080")
    if "WANDB_BASE_URL" not in os.environ:
        os.environ["WANDB_BASE_URL"] = wandb_base
    wandb_mode = "online" if not args.dont_log_wandb and rank == 0 else "disabled"
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_id,
        id=args.wandb_run_id,
        resume="allow",
        reinit=False,
        mode=wandb_mode,
        settings=wandb.Settings(init_timeout=int(wandb_cfg.get("init_timeout", 180))),
        config={
            "config_file": args.config,
            "resolution": args.train_resolution,
            "gpu_batch_size": args.gpu_batch_size,
            "gpus": gpus,
            "num_steps": args.num_steps,
            "lr_scale": args.lr_scale,
            **cfg,
        },
    )

    # ── Model ──
    resolution = args.train_resolution
    h, w = RESOLUTIONS[resolution]
    model = get_model(cfg, resolution=resolution).to(device_id)

    # ── Steps ──
    batch_size = args.gpu_batch_size
    step_size = gpus * batch_size
    sccm.STEP_SIZE = step_size
    N = step_size * args.num_steps
    k = args.eval_every_steps  # eval every k optimizer steps

    # ── Data ──
    data_cfg = cfg.get("data", {})
    erp = ERPBuilder(data_root=args.data_root)
    erp_train_scenes = erp.build_scenes(
        split="train",
        min_overlap=data_cfg.get("min_overlap", 0.0),
        max_overlap=data_cfg.get("max_overlap", 1.0),
        ht=h, wt=w,
        use_horizontal_flip_aug=data_cfg.get("use_horizontal_flip_aug", True),
        use_yaw_aug=data_cfg.get("use_yaw_aug", True),
    )
    erp_train = ConcatDataset(erp_train_scenes)
    erp_ws = erp.weight_scenes(erp_train, alpha=0.75)

    # Val data for evaluation / visualization
    erp_val_scenes = erp.build_scenes(
        split=args.val_split, ht=h, wt=w,
        use_horizontal_flip_aug=False, use_yaw_aug=False,
    )
    erp_val = ConcatDataset(erp_val_scenes) if erp_val_scenes else None

    print(f"[Rank {rank}] Train: {len(erp_train)} pairs, Val: {len(erp_val) if erp_val else 0} pairs")

    # ── Loss ──
    loss_cfg = cfg.get("loss", {})
    # Convert local_dist keys to int (YAML parses them as strings)
    raw_ld = loss_cfg.get("local_dist", {1: 4, 2: 4, 4: 8, 8: 8})
    local_dist = {int(key): v for key, v in raw_ld.items()}
    depth_loss = RobustLossesERP(
        ce_weight=float(loss_cfg.get("ce_weight", 0.01)),
        local_dist=local_dist,
        local_largest_scale=int(loss_cfg.get("local_largest_scale", 8)),
        depth_interpolation_mode=loss_cfg.get("depth_interpolation_mode", "bilinear"),
        relative_depth_error_threshold=float(loss_cfg.get("relative_depth_error_threshold", 0.05)),
        alpha=float(loss_cfg.get("alpha", 0.5)),
        c=float(loss_cfg.get("c", 1e-4)),
        epipolar_weight=float(loss_cfg.get("epipolar_weight", 0.0)),
        epipolar_scales=loss_cfg.get("epipolar_scales", None),
        epipolar_cert_gate=bool(loss_cfg.get("epipolar_cert_gate", True)),
        epipolar_use_robust=bool(loss_cfg.get("epipolar_use_robust", False)),
        lat_weight_enabled=bool(loss_cfg.get("lat_weight_enabled", True)),
    )

    # Optional warm-start from a different experiment (the holo360d fine-tune
    # configs point at their MP3D checkpoint). Intentionally model-only and
    # non-strict so new heads can be initialized from scratch while the copied
    # backbone/coarse/regression weights are reused.
    train_cfg = cfg.get("training", {})
    init_checkpoint = resolve_path(train_cfg.get("init_checkpoint"))
    if init_checkpoint:
        states = torch.load(init_checkpoint, map_location="cpu")
        ckpt_state = states["model"]
        missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
        if rank == 0:
            # Print a small sample of missing/unexpected so silent random-init
            # is visible at launch time.
            print(
                f"Warm-started model from {init_checkpoint}; "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
            if len(missing) > 0:
                sample_missing = [key for key in missing if "conv_refiner" in key][:8]
                if sample_missing:
                    print(f"  sample missing (conv_refiner): {sample_missing}")
            if len(unexpected) > 0:
                print(f"  sample unexpected: {unexpected[:8]}")

    # ── Optimizer ──
    opt_cfg = cfg.get("optimizer", {})
    lr_scale = args.lr_scale
    enc_lr = float(opt_cfg.get("encoder_lr_base", 5e-6))
    dec_lr = float(opt_cfg.get("decoder_lr_base", 1e-4))

    # LRs are specified per-sample and scaled by the global batch (step_size),
    # normalized to the 8-sample reference batch the base values were tuned on.
    parameters = [
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": step_size * enc_lr * lr_scale / 8},
        {"params": [p for p in model.decoder.parameters() if p.requires_grad], "lr": step_size * dec_lr * lr_scale / 8},
    ]
    optimizer = torch.optim.AdamW(parameters, weight_decay=opt_cfg.get("weight_decay", 0.01))

    sched_cfg = cfg.get("scheduler", {})
    milestone_frac = sched_cfg.get("milestone_fraction", 0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(milestone_frac * args.num_steps)]
    )

    # ── Checkpoint ──
    checkpoint_dir = resolve_path(train_cfg.get("checkpoint_dir", "workspace/checkpoints/"))
    experiment_name = args.wandb_run_id or "sccm"
    global_step = 0
    checkpointer = CheckPoint(checkpoint_dir, experiment_name)
    model, optimizer, lr_scheduler, global_step = checkpointer.load(
        model, optimizer, lr_scheduler, global_step
    )
    # Lower the LR when continuing a finished/resumed run. checkpointer.load()
    # restored the optimizer + MultiStepLR state (their LRs), so a changed
    # --lr_scale alone is ignored. Re-scale the live LRs here instead.
    if args.post_load_lr_mult is not None and global_step > 0:
        m = args.post_load_lr_mult
        for g in optimizer.param_groups:
            g["lr"] *= m
            if "initial_lr" in g:
                g["initial_lr"] *= m
        if hasattr(lr_scheduler, "base_lrs"):
            lr_scheduler.base_lrs = [b * m for b in lr_scheduler.base_lrs]
        if hasattr(lr_scheduler, "_last_lr"):
            lr_scheduler._last_lr = [lr * m for lr in lr_scheduler._last_lr]
        if rank == 0:
            print(f"[post-load-lr] scaled LRs by {m} -> "
                  f"{[round(g['lr'], 12) for g in optimizer.param_groups]}")
    sccm.GLOBAL_STEP = global_step

    # If wandb already has logs at a higher step (from a previous, killed run
    # of the SAME run_id), bump GLOBAL_STEP forward so future logs satisfy
    # wandb's monotonic-step requirement. Otherwise wandb silently drops every
    # log call. Only rank 0 reads the wandb summary; broadcast to the others.
    if rank == 0 and wandb.run is not None:
        wandb_max = int(wandb.run.summary.get("_step", 0) or 0)
        if wandb_max >= sccm.GLOBAL_STEP:
            new_step = wandb_max + step_size
            print(f"[wandb-resume] ckpt step={sccm.GLOBAL_STEP} < wandb._step={wandb_max}; "
                  f"bumping GLOBAL_STEP → {new_step} so logs are monotonic.")
            sccm.GLOBAL_STEP = new_step
    if dist.is_initialized():
        bumped = torch.tensor([sccm.GLOBAL_STEP], dtype=torch.long, device=device_id)
        dist.broadcast(bumped, src=0)
        sccm.GLOBAL_STEP = int(bumped.item())

    # The SphereCovisMatcher coarse stage leaves some parameters unused on a
    # given forward (e.g. the covisibility head when its gate is inactive), so
    # DDP must be told to tolerate that. The V1 GP configs carry no
    # `covis_gated_matching` block and use the cheaper strict mode.
    find_unused = "covis_gated_matching" in cfg.get("model", {})
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=find_unused,
                    gradient_as_bucket_view=True)
    grad_scaler = torch.cuda.amp.GradScaler(
        growth_interval=train_cfg.get("grad_scaler_growth_interval", 1_000_000)
    )
    grad_clip_norm = train_cfg.get("grad_clip_norm", 0.01)
    num_workers = train_cfg.get("num_workers", 8)
    num_vis = wandb_cfg.get("num_vis_samples", 4)

    if args.eval_only:
        if erp_val is not None and len(erp_val) > 0:
            val_metrics = evaluate_val(
                ddp_model, erp_val, batch_size=batch_size,
                num_workers=num_workers, max_pairs=args.val_max_pairs,
                rank=rank, world_size=gpus,
            )
            if rank == 0 and val_metrics:
                wandb.log(val_metrics, step=sccm.GLOBAL_STEP)
                print(f"[EvalOnly] step={sccm.GLOBAL_STEP} " + " ".join(
                    f"{key}={v:.4f}" for key, v in val_metrics.items()
                ))
        dist.barrier()
        dist.destroy_process_group()
        return

    # ── Training loop ──
    for n in range(sccm.GLOBAL_STEP, N, k * step_size):
        sampler_gen = torch.Generator().manual_seed(int(seed) + int(rank) * 100003 + int(n))
        # num_samples per eval-cycle can exceed the dataset size for small datasets
        # (e.g. a few-hundred-pair split < batch_size*k). WeightedRandomSampler with
        # replacement=False then fails in torch.multinomial. Use with-replacement only
        # in that case; large datasets (mp3d/s2d3d) keep the original no-replacement.
        _num_samples = batch_size * k
        _replacement = _num_samples > len(erp_train)
        erp_sampler = torch.utils.data.WeightedRandomSampler(
            erp_ws, num_samples=_num_samples, replacement=_replacement, generator=sampler_gen,
        )
        erp_dataloader = iter(torch.utils.data.DataLoader(
            erp_train, batch_size=batch_size,
            sampler=erp_sampler, num_workers=num_workers,
        ))

        train_k_steps(
            n, k, erp_dataloader, ddp_model, depth_loss, optimizer,
            lr_scheduler, grad_scaler, grad_clip_norm=grad_clip_norm,
        )
        checkpointer.save(model, optimizer, lr_scheduler, sccm.GLOBAL_STEP)

        # ── Visualize matches on train & val (image logging only) ──
        if rank == 0 and wandb_cfg.get("log_images_every_eval", True):
            # Train visualization
            try:
                train_vis_loader = torch.utils.data.DataLoader(
                    erp_train, batch_size=num_vis, shuffle=True, num_workers=2,
                )
                train_batch = next(iter(train_vis_loader))
                visualize_matches(ddp_model, train_batch, prefix="train", num_vis=num_vis)
            except Exception as e:
                print(f"[Warn] Train vis failed: {e}")

            # Val visualization
            if erp_val is not None and len(erp_val) > 0:
                try:
                    val_vis_loader = torch.utils.data.DataLoader(
                        erp_val, batch_size=num_vis, shuffle=True, num_workers=2,
                    )
                    val_batch = next(iter(val_vis_loader))
                    visualize_matches(ddp_model, val_batch, prefix="val", num_vis=num_vis)
                except Exception as e:
                    print(f"[Warn] Val vis failed: {e}")

        # ── Val evaluation (DDP: all ranks participate, rank 0 logs) ──
        if erp_val is not None and len(erp_val) > 0:
            try:
                val_metrics = evaluate_val(
                    ddp_model, erp_val, batch_size=batch_size,
                    num_workers=num_workers, max_pairs=args.val_max_pairs,
                    rank=rank, world_size=gpus,
                )
                if rank == 0 and val_metrics:
                    wandb.log(val_metrics, step=sccm.GLOBAL_STEP)
                    print(f"[Eval] step={sccm.GLOBAL_STEP} " + " ".join(
                        f"{key}={v:.4f}" for key, v in val_metrics.items()))
                    # Save best checkpoint by pck_0p35 (in addition to _latest each cycle).
                    checkpointer.save_best(model, optimizer, lr_scheduler, sccm.GLOBAL_STEP,
                                           val_metrics.get("val/pck_0p35deg"), metric_name="pck_0p35")
            except Exception as e:
                if rank == 0:
                    print(f"[Warn] Val eval failed: {e}")

    if rank == 0:
        print("Training complete.")
        wandb.finish()


def build_parser():
    parser = ArgumentParser(description="SCCM ERP training driver (paper path)")
    parser.add_argument("--config", required=True, help="YAML config file path")
    parser.add_argument("--data_root", default="/mnt/datasets/matterport3d_megadepth")
    parser.add_argument("--train_resolution", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--gpu_batch_size", default=4, type=int)
    parser.add_argument("--num_steps", default=250000, type=int, help="Total optimizer steps")
    parser.add_argument("--lr_scale", default=1.0, type=float, help="LR multiplier")
    parser.add_argument("--seed", default=None, type=int,
                        help="Random seed. Defaults to training.seed in the config (42). "
                             "Rank r uses seed + r.")
    parser.add_argument("--post_load_lr_mult", default=None, type=float,
                        help="After resuming a checkpoint, multiply optimizer LRs and "
                             "scheduler base_lrs by this factor. Use to lower the LR when "
                             "continuing a finished run (e.g. 0.1) — needed because "
                             "checkpointer.load() restores the old optimizer/scheduler LR "
                             "state and would otherwise ignore a changed --lr_scale.")
    parser.add_argument("--eval_every_steps", default=25000, type=int, help="Eval/checkpoint interval")
    parser.add_argument("--val_max_pairs", default=256, type=int,
                        help="Max val pairs per eval call. Use a large value (e.g. 999999) for full val.")
    parser.add_argument("--eval_only", action="store_true",
                        help="Load checkpoint, run validation once, and exit without training or saving.")
    parser.add_argument("--val_split", default="val", choices=["train", "val", "test"],
                        help="Which pair split the eval/val dataset is built from (default: val). "
                             "Use 'test' for held-out test eval.")
    parser.add_argument("--wandb_project", default="sccm")
    parser.add_argument("--wandb_run_id", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--dont_log_wandb", action="store_true")
    parser.add_argument("--debug_mode", action="store_true")
    return parser


if __name__ == "__main__":
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    os.environ["OMP_NUM_THREADS"] = "16"
    if "WANDB_BASE_URL" not in os.environ:
        os.environ["WANDB_BASE_URL"] = "http://localhost:8080"
    torch.backends.cudnn.allow_tf32 = True

    args, _ = build_parser().parse_known_args()
    sccm.DEBUG_MODE = args.debug_mode

    train(args)

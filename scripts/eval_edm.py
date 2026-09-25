"""
Evaluate EDM (Equirectangular Dense Matching) on our Matterport3D val set.

Uses the same val data, same eval seed, same metrics as our experiments.
Logs results to wandb under the same project for direct comparison.

Usage:
    python3 experiments/eval_edm.py \
        --checkpoint submodules/EDM/edm_matterport3d.pth \
        --data_root /mnt/datasets/matterport3d_megadepth \
        --wandb_project roma-erp-v060 \
        --wandb_run_id edm_baseline
"""

import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

import wandb

from sccm.datasets.megadepth_erp import ERPBuilder
from sccm.utils.utils_sphere import get_gt_warp_erp, erp_normalized_to_ray


def _edm_metrics(ang, lat, prefix):
    """Standard + area-weighted (cos-phi) metrics from pooled angular errors."""
    import math as _m
    out = {
        f"{prefix}/pck_1deg": (ang < 1.0).float().mean().item(),
        f"{prefix}/pck_3deg": (ang < 3.0).float().mean().item(),
        f"{prefix}/pck_5deg": (ang < 5.0).float().mean().item(),
        f"{prefix}/mae_deg": ang.mean().item(),
        f"{prefix}/median_deg": ang.median().item(),
        f"{prefix}/n_pixels": ang.numel(),
    }
    if lat is not None and lat.numel() == ang.numel() and ang.numel() > 0:
        w = torch.cos(lat.double() * _m.pi / 180.0).clamp(min=0.0)
        a = ang.double(); W = w.sum().item()
        sa, idx = torch.sort(a); cw = torch.cumsum(w[idx], 0)
        j = int((cw >= W / 2.0).nonzero()[0, 0].item()) if W > 0 else 0
        out.update({
            f"{prefix}/pck_1deg_aw": (w * (a < 1).double()).sum().item() / W,
            f"{prefix}/pck_3deg_aw": (w * (a < 3).double()).sum().item() / W,
            f"{prefix}/pck_5deg_aw": (w * (a < 5).double()).sum().item() / W,
            f"{prefix}/mae_deg_aw": (w * a).sum().item() / W,
            f"{prefix}/median_deg_aw": float(sa[j].item()),
        })
    return out


def evaluate_edm(model, val_dataset, max_pairs=256, device="cuda", eps=1e-8,
                 num_shards=1, shard_id=0, metric_prefix="val"):
    """Evaluate EDM model on val set with angular metrics.

    Args:
        model: EDM TorchScript model
        val_dataset: ConcatDataset of ERPScene
        max_pairs: number of val pairs to evaluate
        device: cuda device
        num_shards/shard_id: split val set across multiple processes
    Returns:
        (dict of metrics, raw angular errors tensor)
    """
    model.eval()

    # Same seed as our experiments for fair comparison
    n = len(val_dataset)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42))[:max_pairs].tolist()
    if num_shards > 1:
        indices = indices[shard_id::num_shards]
    val_subset = torch.utils.data.Subset(val_dataset, indices)

    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=1, shuffle=False, num_workers=4,
    )

    # EDM uses H=320, W=640
    EDM_H, EDM_W = 320, 640

    all_ang = []
    all_lat = []
    for batch in tqdm(val_loader, desc="Evaluating EDM"):
        im_A = batch["im_A"].to(device)  # (1, 3, H_orig, W_orig)
        im_B = batch["im_B"].to(device)

        # Resize to EDM's expected resolution
        im_A_edm = F.interpolate(im_A, size=(EDM_H, EDM_W), mode="bilinear", align_corners=False)
        im_B_edm = F.interpolate(im_B, size=(EDM_H, EDM_W), mode="bilinear", align_corners=False)

        with torch.no_grad():
            warp, certainty = model.match(im_A_edm, im_B_edm, device=device, DK_erp=True)
            # warp: (H, 2*W, 4) — [x1_n, y1_n, x2_n, y2_n] for both directions
            # We need A→B: first W columns, channels 2:4
            flow_pred = warp[:, :EDM_W, 2:]  # (H, W, 2) — predicted coords in B

        # GT warp at EDM resolution
        gt_warp, gt_prob = get_gt_warp_erp(
            batch["im_A_depth"], batch["im_B_depth"], batch["T_1to2"],
            H=EDM_H, W=EDM_W,
        )
        gt_warp = gt_warp.float().to(device)
        gt_prob = gt_prob.to(device)

        mask = gt_prob[0] > 0.99  # (H, W)
        if not mask.any():
            continue

        # Angular error
        ray_pred = erp_normalized_to_ray(flow_pred[mask])
        ray_gt = erp_normalized_to_ray(gt_warp[0][mask])
        dot = (ray_pred * ray_gt).sum(dim=-1).clamp(-1 + eps, 1 - eps)
        ang_deg = torch.acos(dot) * (180.0 / math.pi)
        all_ang.append(ang_deg.cpu())
        _row = torch.arange(EDM_H, device=device).view(EDM_H, 1).expand(EDM_H, EDM_W)
        _lat = ((_row.float() + 0.5) / EDM_H) * 180.0 - 90.0
        all_lat.append(_lat[mask].cpu())

    if not all_ang:
        return {}, torch.empty(0), torch.empty(0)

    all_ang = torch.cat(all_ang)
    all_lat = torch.cat(all_lat)
    metrics = {
        f"{metric_prefix}/pck_0p35deg": (all_ang < 0.35).float().mean().item(),
        f"{metric_prefix}/pck_0p5deg":  (all_ang < 0.5).float().mean().item(),
        f"{metric_prefix}/pck_1deg":    (all_ang < 1.0).float().mean().item(),
        f"{metric_prefix}/pck_3deg":    (all_ang < 3.0).float().mean().item(),
        f"{metric_prefix}/pck_5deg":    (all_ang < 5.0).float().mean().item(),
        f"{metric_prefix}/mae_deg":     all_ang.mean().item(),
        f"{metric_prefix}/median_deg":  all_ang.median().item(),
        f"{metric_prefix}/n_pixels":    all_ang.numel(),
    }
    return metrics, all_ang, all_lat


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", default="submodules/EDM/edm_matterport3d.pth")
    parser.add_argument("--data_root",
                        default=os.environ.get("SCCM_DATA_ROOT", "data/matterport3d_megadepth"),
                        help="dataset root; overrides $SCCM_DATA_ROOT")
    parser.add_argument("--max_pairs", type=int, default=256)
    parser.add_argument("--wandb_project", default="roma-erp-v060")
    parser.add_argument("--wandb_run_id", default="edm_baseline")
    parser.add_argument("--dont_log_wandb", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--shard_dir", default="shards/edm")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate shard outputs and log to wandb (no eval)")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"],
                        help="dataset split — also used as wandb metric prefix")
    parser.add_argument("--metric_prefix", default=None, help="wandb metric prefix (default: same as --split)")
    args = parser.parse_args()
    metric_prefix = args.metric_prefix or args.split  # "val/" or "test/" panel in wandb

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.shard_dir, exist_ok=True)

    # ── Aggregate mode: read shard files, log to wandb ──
    if args.aggregate:
        import glob
        shard_files = sorted(glob.glob(os.path.join(args.shard_dir, "shard_*.pt")))
        if not shard_files:
            print(f"No shard files in {args.shard_dir}")
            return
        loaded = [torch.load(f) for f in shard_files]
        if isinstance(loaded[0], dict):
            all_ang = torch.cat([d["ang"] for d in loaded])
            all_lat = torch.cat([d["lat"] for d in loaded])
        else:
            all_ang = torch.cat(loaded); all_lat = None
        print(f"Aggregated {len(shard_files)} shards → {all_ang.numel()} total pixel angular errors")
        metrics = _edm_metrics(all_ang, all_lat, metric_prefix)
        wandb_mode = "disabled" if args.dont_log_wandb else "online"
        if "WANDB_BASE_URL" not in os.environ:
            os.environ["WANDB_BASE_URL"] = "http://localhost:8080"
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_id,
            id=args.wandb_run_id,
            mode=wandb_mode,
            config={"model": "EDM", "checkpoint": args.checkpoint, "n_shards": len(shard_files)},
        )
        print("\n=== EDM Evaluation Results (aggregated) ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        wandb.log(metrics, step=0)
        wandb.finish()
        return

    # Load EDM model
    print(f"[shard {args.shard_id}/{args.num_shards}] Loading EDM from {args.checkpoint}")
    model = torch.jit.load(args.checkpoint, map_location=device)
    model.eval()
    if device == "cuda":
        model = model.cuda()

    # Load dataset (split = val or test) — same as our experiments
    erp = ERPBuilder(data_root=args.data_root)
    val_scenes = erp.build_scenes(
        split=args.split, ht=320, wt=640,
        use_horizontal_flip_aug=False, use_yaw_aug=False,
    )
    from torch.utils.data import ConcatDataset
    val_dataset = ConcatDataset(val_scenes) if val_scenes else None

    if val_dataset is None or len(val_dataset) == 0:
        print("No val data found!")
        return

    print(f"[shard {args.shard_id}/{args.num_shards}] Val dataset: {len(val_dataset)} pairs")

    # Evaluate (this shard only)
    metrics, all_ang, all_lat = evaluate_edm(
        model, val_dataset, max_pairs=args.max_pairs, device=device,
        num_shards=args.num_shards, shard_id=args.shard_id,
        metric_prefix=metric_prefix,
    )

    # Sharded run: dump raw angular errors + latitudes, skip wandb
    if args.num_shards > 1:
        out_path = os.path.join(args.shard_dir, f"shard_{args.shard_id:02d}.pt")
        torch.save({"ang": all_ang, "lat": all_lat}, out_path)
        print(f"[shard {args.shard_id}] saved {all_ang.numel()} angular errors → {out_path}")
        return

    # Single-process run: log to wandb directly
    wandb_mode = "disabled" if args.dont_log_wandb else "online"
    if "WANDB_BASE_URL" not in os.environ:
        os.environ["WANDB_BASE_URL"] = "http://localhost:8080"
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_id,
        id=args.wandb_run_id,
        mode=wandb_mode,
        config={"model": "EDM", "checkpoint": args.checkpoint},
    )

    if metrics:
        print("\n=== EDM Evaluation Results ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        wandb.log(metrics, step=0)
    else:
        print("No valid metrics computed")

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()

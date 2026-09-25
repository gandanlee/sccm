"""
Evaluate RoMa-ERP checkpoints on Matterport3D val with configurable angular PCK thresholds.

This is intended for post-training precision checks such as PCK@0.35deg and
PCK@0.5deg, without entering the training loop.
"""

import math
import os
import sys
from argparse import ArgumentParser
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from tqdm import tqdm
import wandb

import sccm
from sccm.build import RESOLUTIONS, get_model, load_config
from sccm.datasets.megadepth_erp import ERPBuilder
from sccm.utils.utils_sphere import erp_normalized_to_ray, get_gt_warp_erp


def metric_name_for_threshold(threshold, prefix="val"):
    token = ("%g" % threshold).replace(".", "p")
    return f"{prefix}/pck_{token}deg"


def parse_thresholds(raw):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


@torch.no_grad()
def evaluate_precision(model, val_dataset, thresholds, batch_size, num_workers,
                       max_pairs, rank, world_size, device_id, eps=1e-8,
                       metric_prefix="val", degrade_to=None, eval_grid=None):
    model.eval()
    h, w = model.h_resized, model.w_resized
    # Metric grid: by default the model grid. With --eval_grid, the dataset is
    # built at that (coarser) size and GT/metrics are computed there, mirroring
    # experiments/eval_edm.py exactly — inputs are upsampled to the model's /14
    # grid only because DINOv2 requires it, and the predicted warp is resized
    # back down before scoring. This makes input information, GT resolution and
    # the scored pixel set identical to a baseline evaluated at that size.
    hm, wm = eval_grid if eval_grid is not None else (h, w)

    n = len(val_dataset)
    if n > max_pairs:
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(42))[:max_pairs].tolist()
    else:
        indices = list(range(n))
    rank_indices = indices[rank::world_size]
    val_subset = torch.utils.data.Subset(val_dataset, rank_indices)
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    # ----------- streaming per-rank statistics (NO raw tensor accumulation) -----------
    # Histogram for median: 0..180 deg, 18000 bins → 0.01 deg resolution, ~144 KB per rank.
    HIST_BINS = 18000
    HIST_MAX = 180.0
    BIN_WIDTH = HIST_MAX / HIST_BINS
    thresholds_sorted = sorted(thresholds)
    n_thresh = len(thresholds_sorted)

    local_count = 0
    local_sum = 0.0
    local_thresh_counts = [0] * n_thresh
    local_hist = torch.zeros(HIST_BINS, dtype=torch.int64)
    # ── area-weighted (sphere-area-fair) statistics: weight each ERP pixel by
    #    cos(latitude) to remove the pole over-representation of the ERP grid.
    local_count_aw = 0.0
    local_sum_aw = 0.0
    local_thresh_aw = [0.0] * n_thresh
    local_hist_aw = torch.zeros(HIST_BINS, dtype=torch.float64)
    _lat = ((torch.arange(hm, dtype=torch.float32) + 0.5) / hm) * math.pi - math.pi / 2
    _coslat = torch.cos(_lat).clamp(min=0.0).to(device_id)   # (hm,)

    # ── optional diagnostic: error resolved by |latitude| band (default OFF).
    #    SPA's tangent/LST scaling and AAC's log-cos-phi term are all functions
    #    of latitude, so this localises where a regression actually lives.
    #    Enable with EVAL_LAT_BANDS=1; nothing else in the eval path changes.
    LAT_ON = os.environ.get("EVAL_LAT_BANDS", "") == "1"
    LAT_EDGES = [0, 15, 30, 45, 60, 75, 90]
    LAT_TAIL = 30.0                                       # "catastrophic" threshold
    if LAT_ON:
        _absdeg = _lat.abs() * (180.0 / math.pi)
        _band = torch.bucketize(_absdeg, torch.tensor(LAT_EDGES[1:-1],
                                                      dtype=torch.float32)).to(device_id)
        # per band: [count, sum_ang, count(ang<1deg), count(ang>LAT_TAIL)]
        lat_stats = torch.zeros(len(LAT_EDGES) - 1, 4, dtype=torch.float64, device=device_id)

    # optional per-pair dump (default OFF): EVAL_PAIR_CSV=<path prefix>.
    # Columns are GT-only pair descriptors plus this model's per-pair errors, so
    # two models' CSVs can be joined row-by-row (same deterministic pair order).
    PAIR_CSV = os.environ.get("EVAL_PAIR_CSV", "")
    _pair_rows = []

    # tqdm on rank 0 only; sparse refresh.
    iterator = val_loader
    if rank == 0:
        iterator = tqdm(
            val_loader,
            total=len(val_subset),
            desc=f"{metric_prefix} eval",
            mininterval=5.0,
            file=sys.stdout,
            dynamic_ncols=True,
        )
    for step, batch in enumerate(iterator):
        im_A = batch["im_A"].to(device_id)
        im_B = batch["im_B"].to(device_id)

        if degrade_to is not None:
            # Information-matched eval: round-trip the inputs through a coarser
            # resolution (e.g. EDM's native 320x640) before matching. The model
            # grid, GT warp, and all angular metrics stay unchanged.
            H0, W0 = im_A.shape[-2:]
            im_A = F.interpolate(F.interpolate(im_A, size=degrade_to, mode="bilinear", antialias=True),
                                 size=(H0, W0), mode="bilinear")
            im_B = F.interpolate(F.interpolate(im_B, size=degrade_to, mode="bilinear", antialias=True),
                                 size=(H0, W0), mode="bilinear")

        if eval_grid is not None:
            # dataset is at (hm, wm); upsample only to satisfy the /14 patch grid
            im_A = F.interpolate(im_A, size=(h, w), mode="bilinear", align_corners=False)
            im_B = F.interpolate(im_B, size=(h, w), mode="bilinear", align_corners=False)

        torch.cuda.empty_cache()
        try:
            matches, _ = model.match(im_A, im_B, batched=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f"[rank{rank} step{step}] OOM during match — skipping pair: {e}")
            torch.cuda.empty_cache()
            continue
        flow_pred = matches[:, :, :, 2:]
        if eval_grid is not None:
            # resize predicted warp back to the metric grid (same as eval_edm.py)
            flow_pred = F.interpolate(flow_pred.permute(0, 3, 1, 2).float(),
                                      size=(hm, wm), mode="bilinear",
                                      align_corners=False).permute(0, 2, 3, 1)

        gt_warp, gt_prob = get_gt_warp_erp(
            batch["im_A_depth"], batch["im_B_depth"], batch["T_1to2"],
            H=hm, W=wm,
        )
        gt_warp = gt_warp.float().to(device_id)
        gt_prob = gt_prob.to(device_id)

        mask = gt_prob > 0.99
        if mask.any():
            ray_pred = erp_normalized_to_ray(flow_pred[mask])
            ray_gt = erp_normalized_to_ray(gt_warp[mask])
            dot = (ray_pred * ray_gt).sum(dim=-1).clamp(-1 + eps, 1 - eps)
            ang_deg = torch.acos(dot) * (180.0 / math.pi)
            # ★ statistics-only accumulation (raw tensor NOT kept)
            ang_cpu = ang_deg.cpu()
            local_count += int(ang_cpu.numel())
            local_sum += float(ang_cpu.sum().item())
            for i, t in enumerate(thresholds_sorted):
                local_thresh_counts[i] += int((ang_cpu < t).sum().item())
            # torch.histc: output always has exactly HIST_BINS bins (safe vs bincount overflow).
            local_hist += torch.histc(ang_cpu.float(), bins=HIST_BINS, min=0.0, max=HIST_MAX).to(torch.int64)
            # area-weighted: per-pixel weight = cos(latitude) of that ERP row
            wgt = _coslat.view(1, hm, 1).expand_as(gt_prob)[mask].double().cpu()
            a = ang_cpu.double()
            local_count_aw += float(wgt.sum().item())
            local_sum_aw += float((wgt * a).sum().item())
            for i, t in enumerate(thresholds_sorted):
                local_thresh_aw[i] += float((wgt * (a < t).double()).sum().item())
            bidx = (a / BIN_WIDTH).long().clamp(0, HIST_BINS - 1)
            local_hist_aw.scatter_add_(0, bidx, wgt)
            if PAIR_CSV:
                # one row per pair: GT-only pair descriptors + this model's errors.
                T = batch["T_1to2"][0].double()
                R = T[:3, :3]
                rot = math.degrees(math.acos(
                    float(((torch.diagonal(R).sum() - 1) / 2).clamp(-1, 1))))
                tl = math.degrees(math.acos(
                    float((R @ torch.tensor([0., 1., 0.], dtype=torch.float64))[1].clamp(-1, 1))))
                base = float(T[:3, 3].norm())
                dA = batch["im_A_depth"][0]
                dv = dA[dA > 1e-6]
                med_d = float(dv.median()) if dv.numel() else float("nan")
                ad = ang_deg.double()
                _pair_rows.append((
                    rot, tl, base, float(gt_prob.float().mean()), med_d,
                    base / med_d if med_d > 1e-6 else float("nan"),
                    int(ad.numel()), float(ad.mean()),
                    float((ad < 1.0).double().mean()),
                    float((ad > LAT_TAIL).double().mean())))
            if LAT_ON:
                bsel = _band.view(1, hm, 1).expand_as(gt_prob)[mask]      # (M,)
                ad = ang_deg.double()
                one = torch.ones_like(ad)
                lat_stats[:, 0].scatter_add_(0, bsel, one)
                lat_stats[:, 1].scatter_add_(0, bsel, ad)
                lat_stats[:, 2].scatter_add_(0, bsel, (ad < 1.0).double())
                lat_stats[:, 3].scatter_add_(0, bsel, (ad > LAT_TAIL).double())
                del bsel, ad, one
            del ang_cpu, wgt, a

        del im_A, im_B, matches, flow_pred, gt_warp, gt_prob, mask
        if (step + 1) % 100 == 0:
            torch.cuda.empty_cache()

    if PAIR_CSV:
        import csv
        path = f"{PAIR_CSV}_rank{rank}.csv"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rot_deg", "tilt_deg", "baseline", "covis",
                        "med_depth", "parallax", "n_px", "mae", "pck1", "tail_frac"])
            w.writerows(_pair_rows)
        print(f"[pair-csv] {len(_pair_rows)} rows -> {path}", flush=True)

    if LAT_ON:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(lat_stats, op=dist.ReduceOp.SUM)
        if rank == 0:
            L = lat_stats.cpu().numpy()
            print(f"\n[lat-bands] {metric_prefix}  (supervised px only, "
                  f"tail = err > {LAT_TAIL:g} deg)", flush=True)
            print(f"{'|lat| band':>12} {'px share':>9} {'MAE':>8} {'pck@1':>8} {'tail%':>8}")
            tot = max(L[:, 0].sum(), 1.0)
            for j, (a, b) in enumerate(zip(LAT_EDGES[:-1], LAT_EDGES[1:])):
                c = max(L[j, 0], 1.0)
                print(f"{a:>4}-{b:<3}deg  {L[j,0]/tot:9.4f} {L[j,1]/c:8.3f} "
                      f"{L[j,2]/c:8.4f} {100*L[j,3]/c:8.2f}", flush=True)

    # ----------- aggregate across ranks via a single small all_reduce -----------
    # Packed tensor: [count, sum, *thresh_counts (n_thresh), *hist (HIST_BINS)]
    # Total size: (2 + n_thresh + HIST_BINS) × 8 bytes ≈ 144 KB. No OOM possible.
    block = 2 + n_thresh + HIST_BINS
    flat = torch.zeros(2 * block, dtype=torch.float64)
    flat[0] = float(local_count)
    flat[1] = local_sum
    for i in range(n_thresh):
        flat[2 + i] = float(local_thresh_counts[i])
    flat[2 + n_thresh:block] = local_hist.to(torch.float64)
    flat[block + 0] = local_count_aw
    flat[block + 1] = local_sum_aw
    for i in range(n_thresh):
        flat[block + 2 + i] = float(local_thresh_aw[i])
    flat[block + 2 + n_thresh:] = local_hist_aw
    if world_size > 1:
        flat = flat.to(device_id)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat = flat.cpu()

    if rank != 0:
        return {}

    total_count = int(flat[0].item())
    if total_count == 0:
        return {}

    def _median(hist, total):
        cs = hist.cumsum(0)
        idx = (cs >= total / 2.0).nonzero(as_tuple=True)[0]
        b = int(idx[0].item()) if idx.numel() > 0 else 0
        return float((b + 0.5) * BIN_WIDTH)

    total_sum = float(flat[1].item())
    total_thresh_counts = [float(flat[2 + i].item()) for i in range(n_thresh)]
    total_hist = flat[2 + n_thresh:block]

    metrics = {
        metric_name_for_threshold(t, prefix=metric_prefix): total_thresh_counts[i] / total_count
        for i, t in enumerate(thresholds_sorted)
    }
    metrics.update({
        f"{metric_prefix}/mae_deg": total_sum / total_count,
        f"{metric_prefix}/median_deg": _median(total_hist, total_count),
        f"{metric_prefix}/n_pixels": total_count,
    })

    # ── sphere-area-fair (cos φ-weighted) metrics, reported alongside ──
    cnt_aw = float(flat[block].item())
    if cnt_aw > 0:
        sum_aw = float(flat[block + 1].item())
        thr_aw = [float(flat[block + 2 + i].item()) for i in range(n_thresh)]
        hist_aw = flat[block + 2 + n_thresh:]
        for i, t in enumerate(thresholds_sorted):
            metrics[metric_name_for_threshold(t, prefix=metric_prefix) + "_aw"] = thr_aw[i] / cnt_aw
        metrics[f"{metric_prefix}/mae_deg_aw"] = sum_aw / cnt_aw
        metrics[f"{metric_prefix}/median_deg_aw"] = _median(hist_aw, cnt_aw)
    return metrics


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_root",
                        default=os.environ.get("SCCM_DATA_ROOT", "data/matterport3d_megadepth"),
                        help="dataset root; overrides $SCCM_DATA_ROOT")
    parser.add_argument("--train_resolution", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--gpu_batch_size", default=1, type=int)
    parser.add_argument("--max_pairs", default=999999, type=int)
    parser.add_argument("--thresholds", default="0.35,0.5")
    parser.add_argument("--wandb_project", default="sccm")
    parser.add_argument("--wandb_run_id", default=None,
                        help="run name for logging; defaults to the config basename")
    parser.add_argument("--dont_log_wandb", action="store_true")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"],
                        help="dataset split — also used as wandb metric prefix (val/ vs test/)")
    parser.add_argument("--metric_prefix", default=None,
                        help="wandb metric prefix (default: same as --split)")
    parser.add_argument("--eval_grid", default=None,
                        help="'HxW' (e.g. '320x640') — build the dataset and compute GT/metrics "
                             "at this grid (mirrors eval_edm.py); the model still runs on its own "
                             "/14 grid and its warp is resized back. Default: model grid.")
    parser.add_argument("--degrade_to", default=None,
                        help="'HxW' (e.g. '320x640') — round-trip resize inputs through this "
                             "resolution before matching (information-matched eval; "
                             "model grid, GT and metrics unchanged). Default: off.")
    args = parser.parse_args()
    if args.wandb_run_id is None:
        args.wandb_run_id = os.path.splitext(os.path.basename(args.config))[0]
    metric_prefix = args.metric_prefix or args.split

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl", timeout=timedelta(hours=1))

    device_id = local_rank if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)

    cfg = load_config(args.config)
    h, w = RESOLUTIONS[args.train_resolution]
    # --eval_grid HxW: build the dataset (and thus GT/metrics) at that size,
    # mirroring eval_edm.py. Model still runs on its own /14 grid.
    eval_grid = None
    if getattr(args, "eval_grid", None):
        eh, ew = (int(x) for x in args.eval_grid.lower().split("x"))
        eval_grid = (eh, ew)
    model = get_model(cfg, resolution=args.train_resolution).to(device_id)

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint_dir = cfg.get("training", {}).get("checkpoint_dir")
        checkpoint = os.path.join(checkpoint_dir, f"{args.wandb_run_id}_latest.pth")

    states = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(states["model"])
    # Inference-time LAC strength override (supplement sanity check): set the
    # covisibility-head alpha_raw so that softplus(alpha_raw)=SCCM_ALPHA_OVERRIDE.
    _ao = os.environ.get("SCCM_ALPHA_OVERRIDE")
    if _ao:
        import math as _m
        _raw = _m.log(_m.exp(float(_ao)) - 1.0)
        _cnt = 0
        with torch.no_grad():
            for _n, _p in model.named_parameters():
                if _n.endswith("alpha_raw") and "covis_head" in _n:
                    _p.fill_(_raw); _cnt += 1
        print(f"[alpha-override] set {_cnt} covis_head alpha_raw -> {_raw:.4f} "
              f"(alpha={_ao})", flush=True)

    # Inference-time module knock-out (diagnostic; default OFF).  EVAL_KO is a
    # comma list of:
    #   tpb  zero the tangent pairwise-bias output head -> bias 0 -> plain
    #        attention (the module's documented zero-init fallback)
    #   lst  conformal_alpha -> 0 -> cos(phi)^0 = 1 -> identity
    #   lac  covis alpha_raw -> -30 -> softplus ~ 1e-13 -> log-cos-phi term off
    # Localises which module carries a regression *at inference*; it is not a
    # from-scratch counterfactual, since the model was trained with them on.
    _ko = [t.strip() for t in os.environ.get("EVAL_KO", "").split(",") if t.strip()]
    if _ko:
        hits = {}
        with torch.no_grad():
            for _n, _p in model.named_parameters():
                if "tpb" in _ko and "tangent_pe.pe_proj.2." in _n:
                    _p.zero_(); hits["tpb"] = hits.get("tpb", 0) + 1
                if "lst" in _ko and _n.endswith("conformal_alpha"):
                    _p.zero_(); hits["lst"] = hits.get("lst", 0) + 1
                if "lac" in _ko and _n.endswith("alpha_raw") and "covis_head" in _n:
                    _p.fill_(-30.0); hits["lac"] = hits.get("lac", 0) + 1
        print(f"[knock-out] {_ko} -> patched {hits}", flush=True)

    # Inference-time TPB gain sweep (default OFF): TPB_GAIN=<float> scales the
    # tangent pairwise-bias output head, so gamma=0 reproduces EVAL_KO=tpb and
    # gamma=1 leaves the model untouched.  Localises how much of an OOD gap the
    # fixed, content-independent (N,N) prior accounts for.
    _g = os.environ.get("TPB_GAIN", "")
    if _g != "":
        _g = float(_g); _n_hit = 0
        with torch.no_grad():
            for _n, _p in model.named_parameters():
                if "tangent_pe.pe_proj.2." in _n:
                    _p.mul_(_g); _n_hit += 1
        print(f"[tpb-gain] gamma={_g} -> scaled {_n_hit} tensors", flush=True)

    # Inference-time RoPE gain sweep (default OFF): ROPE_GAIN=<float> scales the
    # RoPE rotation angles theta -> gamma*theta by rebuilding the cos/sin tables.
    # gamma=0 makes every rotation the identity, so q.k reduces to a plain dot
    # product (the module's documented no-PE fallback); gamma=1 is untouched.
    # The rebuild is verified against the stored buffers at gamma=1 before use.
    _rg = os.environ.get("ROPE_GAIN", "")
    if _rg != "":
        import math as _math
        _rg = float(_rg); _hit = 0
        for _n, _mod in model.named_modules():
            if _mod.__class__.__name__ == "RoPE2DCircularGain":
                # gain-aware module: it keeps the raw angles, so just pin the gain.
                _mod.set_gain(_rg); _hit += 1
                continue
            if _mod.__class__.__name__ != "RoPE2DCircular":
                continue
            _H, _W, _hd = _mod.H, _mod.W, _mod.head_dim
            _half = _hd // 2; _np = _half // 2
            _lat_f = 10000.0 ** (-torch.arange(0, _np, dtype=torch.float32) * 2.0 / _half)
            _v = torch.linspace(-1.0 + 1.0 / _H, 1.0 - 1.0 / _H, _H)
            _u = torch.linspace(-1.0 + 1.0 / _W, 1.0 - 1.0 / _W, _W)
            _lat = -(_math.pi / 2.0) * _v
            _lon = _math.pi * _u
            _latg = _lat[:, None].expand(_H, _W).reshape(_H * _W)
            _long = _lon[None, :].expand(_H, _W).reshape(_H * _W)
            # longitude freqs: integers (default) or geometric — pick whichever reproduces the buffer
            _cands = []
            _ints = torch.arange(1, _np + 1, dtype=torch.float32)
            _cands.append(_ints)
            _cands.append(10000.0 ** (-torch.arange(0, _np, dtype=torch.float32) * 2.0 / _half))
            _lon_f = None
            for _c in _cands:
                if torch.allclose((_long[:, None] * _c[None, :]).cos(),
                                  _mod.lon_cos.float().cpu(), atol=1e-4):
                    _lon_f = _c; break
            _ok_lat = torch.allclose((_latg[:, None] * _lat_f[None, :]).cos(),
                                     _mod.lat_cos.float().cpu(), atol=1e-4)
            if _lon_f is None or not _ok_lat:
                raise RuntimeError(f"[rope-gain] table reconstruction mismatch at {_n}")
            _la = _rg * (_latg[:, None] * _lat_f[None, :])
            _lo = _rg * (_long[:, None] * _lon_f[None, :])
            _dev, _dt = _mod.lat_cos.device, _mod.lat_cos.dtype
            _mod.lat_cos.copy_(_la.cos().to(_dev, _dt)); _mod.lat_sin.copy_(_la.sin().to(_dev, _dt))
            _mod.lon_cos.copy_(_lo.cos().to(_dev, _dt)); _mod.lon_sin.copy_(_lo.sin().to(_dev, _dt))
            _hit += 1
        print(f"[rope-gain] gamma={_rg} -> rebuilt {_hit} RoPE table(s)", flush=True)
    step = int(states.get("n", 0))
    model.eval()

    erp = ERPBuilder(data_root=args.data_root)
    val_scenes = erp.build_scenes(
        split=args.split, ht=(eval_grid[0] if eval_grid else h), wt=(eval_grid[1] if eval_grid else w),
        use_horizontal_flip_aug=False, use_yaw_aug=False,
    )
    val_dataset = ConcatDataset(val_scenes)
    thresholds = parse_thresholds(args.thresholds)

    degrade_to = None
    if args.degrade_to:
        dh, dw = (int(x) for x in args.degrade_to.lower().split("x"))
        degrade_to = (dh, dw)
        if rank == 0:
            print(f"[degrade] inputs round-tripped through {dh}x{dw} before matching "
                  f"(model grid {h}x{w}, GT/metrics unchanged)", flush=True)

    metrics = evaluate_precision(
        model, val_dataset, thresholds,
        batch_size=args.gpu_batch_size,
        num_workers=0,  # OOM mitigation — workers inherit CUDA context and bloat per-GPU memory
        max_pairs=args.max_pairs,
        rank=rank, world_size=world_size, device_id=device_id,
        metric_prefix=metric_prefix,
        degrade_to=degrade_to,
        eval_grid=eval_grid,
    )

    if rank == 0:
        # Print metrics first so they are always captured, even if wandb fails.
        print(f"\n=== Precision Evaluation: {args.wandb_run_id} step={step} ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}" if k != "val/n_pixels" else f"  {k}: {v}")
        try:
            wandb_mode = "disabled" if (args.dont_log_wandb or
                                        os.environ.get("WANDB_MODE") in ("disabled", "offline")) else "online"
            if "WANDB_BASE_URL" not in os.environ:
                os.environ["WANDB_BASE_URL"] = cfg.get("wandb", {}).get("base_url", "http://localhost:8080")
            wandb.init(
                project=args.wandb_project, name=args.wandb_run_id, id=args.wandb_run_id,
                mode=wandb_mode,
                config={"config_file": args.config, "checkpoint": checkpoint,
                        "step": step, "thresholds": thresholds, "max_pairs": args.max_pairs},
            )
            wandb.log(metrics, step=step)
            wandb.finish()
        except Exception as e:
            print(f"[wandb logging skipped: {type(e).__name__}: {e}]")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

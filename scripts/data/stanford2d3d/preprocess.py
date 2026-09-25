"""
Stanford 2D-3D-Semantics Dataset → MegaDepth/RoMa scene_info format converter.

Dataset layout:
    <dataset_root>/
        area_1/
            pano/
                rgb/    camera_{uuid}_{room}_frame_equirectangular_domain_rgb.png
                depth/  camera_{uuid}_{room}_frame_equirectangular_domain_depth.png
                pose/   camera_{uuid}_{room}_frame_equirectangular_domain_pose.json
        area_2/ ... area_6/   (usually distributed as .tar; the user must extract them into dataset_root beforehand)

Pose JSON:
    camera_rt_matrix: 3×4 **world-to-cam** matrix (p_cam = R @ p_world + t)
    → append a [0,0,0,1] row → store as 4×4
    Check: -R^T @ t ≈ camera_location (matches the camera_location field)

Depth encoding:
    uint16 PNG (2048×4096), unit: mm (depth_meters = pixel_value / 1000.0)
    pixel_value = 65535: invalid pixel → treated as 0

Split strategy:
    scene_corpus: train/val/test assigned per area (see AREA_SPLIT)
    pair split: the pool of pairs passing the overlap filter is sliced into train/val/test in seed-based order

Defaults matched to the EDM (CVPR 2025) Stanford2D3D evaluation protocol:
    - ERP resolution 640×320
    - overlap = inlier/(H×W), relative depth threshold 0.1; for i<j only the one-way direction A=i→B=j is used
    - test pairs are recommended to have overlap > 50% as in the paper → --min_overlap 0.5 (default)

Output layout (Mat3D/MegaDepth compatible):
    <out_root>/
        prep_scene_info/
            s2d3d_area_1.npy   ...
        area_1/
            images/  {stem}.jpg
            depths/  {stem}.h5

Dependencies:
    pip install h5py tqdm numpy pillow
    with --visualize: matplotlib

Parallelism:
    --num_workers (default 32): for multiple areas, ProcessPool area count = min(n, number of areas).
    For both single and multiple areas, frame processing and pair overlap inside each area run in parallel on a ThreadPool (num_workers; sequential if 1).
"""

import argparse
import importlib.util
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

AREAS = ["area_1", "area_2", "area_3", "area_4", "area_5a", "area_5b", "area_6"]

AREA_SPLIT = {
    # S2D3D used entirely for cross-dataset eval (zero-shot from MP3D-trained models).
    # All 7 areas → test corpus. No train/val split.
    "train": [],
    "val":   [],
    "test":  ["area_1", "area_2", "area_3", "area_4", "area_5a", "area_5b", "area_6"],
}

_AREA_TO_CORPUS = {a: s for s, lst in AREA_SPLIT.items() for a in lst}


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------

def load_pose(pose_path: str) -> np.ndarray:
    """
    pose JSON → world-to-cam 4×4 matrix.

    camera_rt_matrix is a 3×4 world-to-cam matrix.
    Check: -R^T @ t ≈ the camera_location field of the pose JSON.
    """
    with open(pose_path) as f:
        data = json.load(f)
    mat34 = np.array(data["camera_rt_matrix"], dtype=np.float64)  # (3, 4)
    mat44 = np.eye(4, dtype=np.float64)
    mat44[:3, :] = mat34
    return mat44


def load_depth(depth_path: str) -> np.ndarray:
    """
    uint16 PNG → float32 meters.
    Per Stanford 2D-3D-Semantics docs: depth is stored as raw / 512 (NOT /1000).
    Verified empirically: /512 gives mean depth ~2.6m, max ~15m (matches indoor office).
                          /1000 gives ~half values, breaks all geometry-dependent eval.
    65535 (invalid sentinel) → 0.0
    """
    raw = np.array(Image.open(depth_path), dtype=np.uint16)
    depth = raw.astype(np.float32) / 512.0
    depth[raw == 65535] = 0.0
    return depth


def _s2d3_convert_one_frame(task: dict):
    """Save RGB/depth of a single frame + its pose (ThreadPool worker)."""
    idx = task["idx"]
    stem = task["stem"]
    rgb_p = task["rgb_p"]
    depth_p = task["depth_p"]
    pose_p = task["pose_p"]
    img_out = task["img_out"]
    dep_out = task["dep_out"]
    rel_img = task["rel_img"]
    rel_dep = task["rel_dep"]
    K_dummy = task["K_dummy"]
    resize = task["resize"]
    out_h = task["out_h"]
    out_w = task["out_w"]
    try:
        pose = load_pose(pose_p)
    except Exception as e:
        print(f"    [WARN] {stem}: failed to load pose — {e}")
        return None

    if not os.path.exists(img_out):
        try:
            img = Image.open(rgb_p).convert("RGB")
            if resize and (img.height != out_h or img.width != out_w):
                img = img.resize((out_w, out_h), Image.LANCZOS)
            img.save(img_out, quality=95)
        except Exception as e:
            print(f"    [WARN] {stem}: failed to save image — {e}")
            return None

    if not os.path.exists(dep_out):
        try:
            depth_m = load_depth(depth_p)
            if resize and (depth_m.shape[0] != out_h or depth_m.shape[1] != out_w):
                depth_m = np.array(
                    Image.fromarray(depth_m).resize((out_w, out_h), Image.NEAREST),
                    dtype=np.float32,
                )
            with h5py.File(dep_out, "w") as f:
                f.create_dataset("depth", data=depth_m, compression="gzip")
        except Exception as e:
            print(f"    [WARN] {stem}: failed to save depth — {e}")
            return None

    return {
        "idx": idx,
        "rel_img": rel_img,
        "rel_dep": rel_dep,
        "K_dummy": K_dummy,
        "pose": pose,
    }


def _s2d3_pair_overlap(task: dict):
    """EDM overlap of a single (i,j) pair, one-way A=i→B=j (ThreadPool worker)."""
    i, j = task["i"], task["j"]
    depths_mem = task["depths_mem"]
    poses_np = task["poses_np"]
    overlap_threshold = task["overlap_threshold"]
    min_overlap = task["min_overlap"]
    max_overlap = task["max_overlap"]
    max_baseline = task["max_baseline"]
    if max_baseline > 0:
        c_i = -poses_np[i][:3, :3].T @ poses_np[i][:3, 3]
        c_j = -poses_np[j][:3, :3].T @ poses_np[j][:3, 3]
        if float(np.linalg.norm(c_i - c_j)) > max_baseline:
            return None
    T_ij = poses_np[j] @ np.linalg.inv(poses_np[i])
    ov = erp_overlap_edm(
        depths_mem[i], depths_mem[j], T_ij, overlap_threshold
    )
    if min_overlap <= ov <= max_overlap:
        return i, j, float(ov)
    return None


def find_frames(area_dir: str) -> list:
    """Return the list of frames for which pano/{rgb,depth,pose} all exist."""
    rgb_dir   = os.path.join(area_dir, "pano", "rgb")
    depth_dir = os.path.join(area_dir, "pano", "depth")
    pose_dir  = os.path.join(area_dir, "pano", "pose")

    if not os.path.isdir(rgb_dir):
        return []

    frames = []
    for fn in sorted(os.listdir(rgb_dir)):
        if not fn.endswith("_rgb.png"):
            continue
        stem    = fn[: -len("_rgb.png")]
        rgb_p   = os.path.join(rgb_dir,   fn)
        depth_p = os.path.join(depth_dir, stem + "_depth.png")
        pose_p  = os.path.join(pose_dir,  stem + "_pose.json")
        if os.path.exists(depth_p) and os.path.exists(pose_p):
            frames.append((stem, rgb_p, depth_p, pose_p))
    return frames


# ---------------------------------------------------------------------------
# EDM-style ERP depth overlap (w2c poses, same composition as Mat3D/OB3D)
# ---------------------------------------------------------------------------

def erp_overlap_edm(
    depth_A: np.ndarray,
    depth_B: np.ndarray,
    T_AtoB: np.ndarray,
    threshold: float = 0.1,
) -> float:
    """inlier / (H*W), relative depth |d_proj-d_B|/d_B < threshold."""
    H, W = depth_A.shape
    denom = float(H * W)
    if denom <= 0:
        return 0.0

    u = np.arange(W, dtype=np.float32) + 0.5
    v = np.arange(H, dtype=np.float32) + 0.5
    lon = (u / W - 0.5) * 2.0 * np.pi
    lat = (v / H - 0.5) * np.pi
    cos_lat = np.cos(lat)[:, None]
    dirs = np.stack(
        [
            cos_lat * np.sin(lon)[None, :],
            np.sin(lat)[:, None] * np.ones((1, W), dtype=np.float32),
            cos_lat * np.cos(lon)[None, :],
        ],
        axis=-1,
    )

    valid_A = depth_A > 0
    if not np.any(valid_A):
        return 0.0

    d_A = depth_A[valid_A]
    pts_A = dirs[valid_A] * d_A[:, None]
    R, t = T_AtoB[:3, :3], T_AtoB[:3, 3]
    pts_B = (R @ pts_A.T).T + t
    d_proj = np.linalg.norm(pts_B, axis=1)
    valid_proj = d_proj > 1e-4
    pts_B = pts_B[valid_proj]
    d_proj = d_proj[valid_proj]
    pts_n = pts_B / d_proj[:, None]
    lon_B = np.arctan2(pts_n[:, 0], pts_n[:, 2])
    lat_B = np.arcsin(np.clip(pts_n[:, 1], -1.0 + 1e-6, 1.0 - 1e-6))
    u_B = np.clip(((lon_B / (2.0 * np.pi) + 0.5) * W).astype(np.int32), 0, W - 1)
    v_B = np.clip(((lat_B / np.pi + 0.5) * H).astype(np.int32), 0, H - 1)
    d_B = depth_B[v_B, u_B]
    inlier = (d_B > 0) & (np.abs(d_proj - d_B) / (d_B + 1e-6) < threshold)
    return float(inlier.sum()) / denom


def _erp_dirs_grid(out_h: int, out_w: int) -> np.ndarray:
    u_arr = np.arange(out_w, dtype=np.float32) + 0.5
    v_arr = np.arange(out_h, dtype=np.float32) + 0.5
    lon_grid = (u_arr / out_w - 0.5) * 2.0 * np.pi
    lat_grid = (v_arr / out_h - 0.5) * np.pi
    cos_lat = np.cos(lat_grid)[:, None]
    return np.stack(
        [
            cos_lat * np.sin(lon_grid)[None, :],
            np.sin(lat_grid)[:, None] * np.ones((1, out_w), dtype=np.float32),
            cos_lat * np.cos(lon_grid)[None, :],
        ],
        axis=-1,
    )


def _edge_magnitude_rgb(img_rgb: np.ndarray) -> np.ndarray:
    x = img_rgb.astype(np.float32)
    g = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
    m = np.sqrt(gx * gx + gy * gy)
    m = m / (float(np.percentile(m, 99.5)) + 1e-6)
    return np.clip(m, 0.0, 1.0)


def _dense_reproj_err_and_edge_overlay(
    depth_A, depth_B, dirs_grid, R, t, img_A, img_B, out_h, out_w,
):
    ya, xa = np.where(depth_A > 0)
    if len(ya) == 0:
        return None, None, 0.0
    d_A = depth_A[ya, xa]
    pts_A = dirs_grid[ya, xa] * d_A[:, None]
    pts_B = (R @ pts_A.T).T + t
    d_proj = np.linalg.norm(pts_B, axis=1)
    m0 = d_proj > 1e-4
    if not np.any(m0):
        return None, None, 0.0
    pts_Bn = pts_B[m0] / d_proj[m0, None]
    lon_B = np.arctan2(pts_Bn[:, 0], pts_Bn[:, 2])
    lat_B = np.arcsin(np.clip(pts_Bn[:, 1], -1.0 + 1e-6, 1.0 - 1e-6))
    xs_B = (lon_B / (2.0 * np.pi) + 0.5) * out_w
    ys_B = (lat_B / np.pi + 0.5) * out_h
    m1 = (xs_B >= 0) & (xs_B < out_w) & (ys_B >= 0) & (ys_B < out_h)
    if not np.any(m1):
        return None, None, 0.0
    ya1 = ya[m0][m1]
    xa1 = xa[m0][m1]
    xb = np.clip(xs_B[m1].astype(np.int32), 0, out_w - 1)
    yb = np.clip(ys_B[m1].astype(np.int32), 0, out_h - 1)
    d_p = d_proj[m0][m1]
    d_b = depth_B[yb, xb]
    m2 = d_b > 0
    if not np.any(m2):
        return None, None, 0.0
    rel = np.abs(d_p[m2] - d_b[m2]) / (d_b[m2] + 1e-6)
    yb2 = yb[m2]
    xb2 = xb[m2]
    sum_err = np.zeros((out_h, out_w), dtype=np.float64)
    cnt = np.zeros((out_h, out_w), dtype=np.int32)
    np.add.at(sum_err, (yb2, xb2), rel)
    np.add.at(cnt, (yb2, xb2), 1)
    mean_err = np.full((out_h, out_w), np.nan, dtype=np.float64)
    valid_c = cnt > 0
    mean_err[valid_c] = sum_err[valid_c] / cnt[valid_c].astype(np.float64)
    inlier_frac = float((rel < 0.1).sum()) / float(rel.size)
    mag_A = _edge_magnitude_rgb(img_A)
    edg_proj = np.zeros((out_h, out_w), dtype=np.float32)
    mag_vals = mag_A[ya1[m2], xa1[m2]]
    np.maximum.at(edg_proj, (yb2, xb2), mag_vals)
    ov = img_B.astype(np.float32) / 255.0
    green = np.zeros_like(ov)
    green[:, :, 1] = np.clip(edg_proj / (np.percentile(edg_proj, 99.5) + 1e-6), 0, 1)
    blend = np.clip(0.65 * ov + 0.35 * green, 0, 1)
    return mean_err, blend, inlier_frac


def visualize_s2d3d_area(
    area_name: str,
    out_root: str,
    image_paths,
    depth_paths,
    poses: np.ndarray,
    pairs_for_vis: list,
    vis_n_pairs: int = 3,
    n_kpts: int = 300,
    seed: int = 0,
):
    """
    RGB, depth, pair reprojection and dense_geom under <out_root>/vis/s2d3d_<area>/.
    Iterates over the pair list until vis_n_pairs of them have been saved successfully (trying only the first N would risk too few outputs).
    """
    if importlib.util.find_spec("matplotlib") is None:
        print("  [VIS SKIP] matplotlib required: pip install matplotlib")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from PIL import Image

    tag = f"s2d3d_{area_name}"
    vis_dir = os.path.join(out_root, "vis", tag)
    os.makedirs(vis_dir, exist_ok=True)

    paths_img = [str(p) for p in list(image_paths)]
    paths_dep = [str(p) for p in list(depth_paths)]
    n = len(paths_img)
    if n == 0 or not pairs_for_vis or len(paths_dep) != n:
        return

    def _depth_at(idx: int):
        p = os.path.join(out_root, paths_dep[idx])
        if not os.path.isfile(p):
            return None
        with h5py.File(p, "r") as f:
            return f["depth"][:].astype(np.float32)

    fi0 = int(pairs_for_vis[0][0])
    depth0 = _depth_at(fi0)
    if depth0 is None:
        print("  [VIS SKIP] failed to load depth")
        return
    out_h, out_w = int(depth0.shape[0]), int(depth0.shape[1])
    dirs_grid = _erp_dirs_grid(out_h, out_w)

    img_path0 = os.path.join(out_root, paths_img[fi0])
    img0 = (
        np.array(Image.open(img_path0))
        if os.path.exists(img_path0)
        else np.zeros((out_h, out_w, 3), dtype=np.uint8)
    )
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    axes[0].imshow(img0)
    axes[0].set_title(f"ERP RGB: {paths_img[fi0].split('/')[-1]}")
    axes[0].axis("off")
    depth_vis = np.where(depth0 > 0, depth0, np.nan)
    vmax = (
        float(np.nanpercentile(depth_vis[~np.isnan(depth_vis)], 95))
        if np.any(~np.isnan(depth_vis))
        else 10.0
    )
    im = axes[1].imshow(depth_vis, cmap="plasma", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=axes[1], label="depth (m)", fraction=0.03)
    axes[1].set_title("ERP depth (camera-local)")
    axes[1].axis("off")
    fig.suptitle(f"{tag} — RGB/depth")
    out_p = os.path.join(vis_dir, f"00_rgb_depth_{fi0:03d}.jpg")
    plt.savefig(out_p, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [VIS] {out_p}")

    poses_np = np.asarray(poses, dtype=np.float64)

    pair_rows = []
    if isinstance(pairs_for_vis, np.ndarray) and pairs_for_vis.ndim == 2:
        for r in pairs_for_vis:
            pair_rows.append((int(r[0]), int(r[1])))
    else:
        for r in pairs_for_vis:
            pair_rows.append((int(r[0]), int(r[1])))

    saved = 0
    for fi, fj in pair_rows:
        if saved >= vis_n_pairs:
            break
        depth_A = _depth_at(fi)
        depth_B = _depth_at(fj)
        if depth_A is None or depth_B is None:
            print(f"  [VIS SKIP] ({fi},{fj}) depth file missing or failed to load")
            continue
        if depth_A.shape != (out_h, out_w) or depth_B.shape != (out_h, out_w):
            print(f"  [VIS SKIP] ({fi},{fj}) depth resolution mismatch")
            continue

        p_a = os.path.join(out_root, paths_img[fi])
        p_b = os.path.join(out_root, paths_img[fj])
        img_A = (
            np.array(Image.open(p_a))
            if os.path.exists(p_a)
            else np.zeros((out_h, out_w, 3), dtype=np.uint8)
        )
        img_B = (
            np.array(Image.open(p_b))
            if os.path.exists(p_b)
            else np.zeros((out_h, out_w, 3), dtype=np.uint8)
        )

        T_AtoB = poses_np[fj] @ np.linalg.inv(poses_np[fi])
        R, t = T_AtoB[:3, :3], T_AtoB[:3, 3]

        valid_ys, valid_xs = np.where(depth_A > 0)
        if len(valid_ys) == 0:
            print(f"  [VIS SKIP] ({fi},{fj}) no valid depth pixels in A")
            continue
        rng = np.random.default_rng(int(seed) + 42 + saved)
        sel = rng.choice(len(valid_ys), size=min(n_kpts, len(valid_ys)), replace=False)
        ys_A, xs_A = valid_ys[sel], valid_xs[sel]

        d_A = depth_A[ys_A, xs_A]
        pts_A = dirs_grid[ys_A, xs_A] * d_A[:, None]
        pts_B = (R @ pts_A.T).T + t
        d_proj = np.linalg.norm(pts_B, axis=1)
        ok = d_proj > 1e-4
        pts_Bn = pts_B[ok] / d_proj[ok, None]
        lon_B = np.arctan2(pts_Bn[:, 0], pts_Bn[:, 2])
        lat_B = np.arcsin(np.clip(pts_Bn[:, 1], -1.0 + 1e-6, 1.0 - 1e-6))
        xs_B = (lon_B / (2.0 * np.pi) + 0.5) * out_w
        ys_B = (lat_B / np.pi + 0.5) * out_h

        xs_Av = xs_A[ok].astype(np.float32)
        ys_Av = ys_A[ok].astype(np.float32)
        in_range = (xs_B >= 0) & (xs_B < out_w) & (ys_B >= 0) & (ys_B < out_h)
        xs_Av, ys_Av = xs_Av[in_range], ys_Av[in_range]
        xs_B = xs_B[in_range].astype(np.float32)
        ys_B = ys_B[in_range].astype(np.float32)

        xs_Bi = np.clip(xs_B.astype(np.int32), 0, out_w - 1)
        ys_Bi = np.clip(ys_B.astype(np.int32), 0, out_h - 1)
        d_B_gt = depth_B[ys_Bi, xs_Bi]
        d_proj_ok = d_proj[ok][in_range]
        inlier = (d_B_gt > 0) & (np.abs(d_proj_ok - d_B_gt) / (d_B_gt + 1e-6) < 0.1)
        inlier_ratio = float(inlier.sum()) / max(len(inlier), 1)

        if len(xs_Av) == 0:
            print(f"  [VIS SKIP] ({fi},{fj}) sampled reprojections fall outside the B ERP range")
            continue

        colors = cm.hsv(np.linspace(0, 1, len(xs_Av)))
        fig, axes = plt.subplots(1, 2, figsize=(22, 5))
        axes[0].imshow(img_A)
        axes[1].imshow(img_B)
        for xi, yi, c in zip(xs_Av, ys_Av, colors):
            axes[0].plot(xi, yi, "o", color=c, markersize=2, markeredgewidth=0)
        for xj, yj, c, inl in zip(xs_B, ys_B, colors, inlier):
            m = "o" if inl else "x"
            axes[1].plot(
                xj, yj, m, color=c, markersize=2 if inl else 3,
                markeredgewidth=0 if inl else 0.5,
            )
        axes[0].set_title(f"A: {paths_img[fi].split('/')[-1]}", fontsize=9)
        axes[1].set_title(f"B: {paths_img[fj].split('/')[-1]}", fontsize=9)
        fig.suptitle(f"{tag} pair {saved} — {len(xs_Av)} reprojections  inlier {inlier_ratio:.1%}")
        plt.tight_layout()
        out_path = os.path.join(vis_dir, f"pair_{saved:02d}_{fi:03d}_{fj:03d}.jpg")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  [VIS] {out_path}  (inlier {inlier_ratio:.1%})")

        mean_err, blend, frac_dense = _dense_reproj_err_and_edge_overlay(
            depth_A, depth_B, dirs_grid, R, t, img_A, img_B, out_h, out_w
        )
        if mean_err is not None:
            fig2, axes2 = plt.subplots(1, 3, figsize=(26, 5))
            axes2[0].imshow(img_B)
            axes2[0].set_title(f"B: {paths_img[fj].split('/')[-1]}", fontsize=9)
            axes2[0].axis("off")
            err_cap = float(np.nanpercentile(mean_err, 95)) if np.any(np.isfinite(mean_err)) else 0.3
            err_cap = max(err_cap, 0.05)
            im_e = axes2[1].imshow(mean_err, cmap="inferno", vmin=0.0, vmax=min(err_cap, 1.0))
            plt.colorbar(im_e, ax=axes2[1], fraction=0.03, label="|d_proj - d_B| / d_B")
            axes2[1].set_title(
                f"dense mean rel. depth err (pix rel<10%: {frac_dense:.1%})",
                fontsize=9,
            )
            axes2[1].axis("off")
            axes2[2].imshow(blend)
            axes2[2].set_title("A gradient edges -> B (green blend)", fontsize=9)
            axes2[2].axis("off")
            fig2.suptitle(f"{tag} pair {saved} dense geom (fi={fi}, fj={fj})")
            plt.tight_layout()
            out_ex = os.path.join(vis_dir, f"pair_{saved:02d}_{fi:03d}_{fj:03d}_dense_geom.jpg")
            plt.savefig(out_ex, dpi=100, bbox_inches="tight")
            plt.close(fig2)
            print(f"  [VIS] {out_ex}  (dense rel<10%: {frac_dense:.1%})")

        saved += 1


# ---------------------------------------------------------------------------
# Single-area conversion
# ---------------------------------------------------------------------------

def convert_area(
    area_name: str,
    dataset_root: str,
    out_root: str,
    out_h: int = 320,
    out_w: int = 640,
    min_overlap: float = 0.5,
    max_overlap: float = 1.0,
    max_pairs: int = 100_000,
    max_baseline: float = 0.0,
    overlap_threshold: float = 0.1,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    by_corpus: bool = False,
    seed: int = 0,
    resize: bool = True,
    visualize: bool = False,
    vis_n_pairs: int = 3,
    vis_n_kpts: int = 300,
    num_workers: int = 32,
):
    area_dir = os.path.join(dataset_root, area_name)
    if not os.path.isdir(area_dir):
        print(f"  [SKIP] {area_name}: directory missing (needs to be extracted)")
        return

    frames = find_frames(area_dir)
    if not frames:
        print(f"  [SKIP] {area_name}: no frames")
        return

    print(f"  {area_name}: found {len(frames)} frames")

    random.seed(seed)
    np.random.seed(seed % (2**32))

    out_img_dir = os.path.join(out_root, area_name, "images")
    out_dep_dir = os.path.join(out_root, area_name, "depths")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_dep_dir, exist_ok=True)

    # ERP dummy intrinsics (K is unused by the spherical loss; stored only for format compatibility)
    f_dummy = out_w / (2.0 * np.pi)
    K_dummy = np.array(
        [[f_dummy, 0, out_w / 2.0], [0, f_dummy, out_h / 2.0], [0, 0, 1.0]],
        dtype=np.float64,
    )

    image_paths, depth_paths, intrinsics_list, poses_list = [], [], [], []

    frame_tasks = []
    for idx, (stem, rgb_p, depth_p, pose_p) in enumerate(frames):
        img_out = os.path.join(out_img_dir, f"{stem}.jpg")
        dep_out = os.path.join(out_dep_dir, f"{stem}.h5")
        rel_img = os.path.join(area_name, "images", f"{stem}.jpg")
        rel_dep = os.path.join(area_name, "depths", f"{stem}.h5")
        frame_tasks.append(
            {
                "idx": idx,
                "stem": stem,
                "rgb_p": rgb_p,
                "depth_p": depth_p,
                "pose_p": pose_p,
                "img_out": img_out,
                "dep_out": dep_out,
                "rel_img": rel_img,
                "rel_dep": rel_dep,
                "K_dummy": K_dummy,
                "resize": resize,
                "out_h": out_h,
                "out_w": out_w,
            }
        )

    frame_results = []
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futs = {ex.submit(_s2d3_convert_one_frame, t): t["idx"] for t in frame_tasks}
            for f in tqdm(
                as_completed(futs),
                total=len(futs),
                desc=f"[{area_name}] frames",
                unit="fr",
                leave=False,
            ):
                r = f.result()
                if r is not None:
                    frame_results.append(r)
    else:
        for t in tqdm(
            frame_tasks,
            desc=f"[{area_name}] frames",
            unit="fr",
            leave=False,
        ):
            r = _s2d3_convert_one_frame(t)
            if r is not None:
                frame_results.append(r)

    frame_results.sort(key=lambda x: x["idx"])
    for r in frame_results:
        image_paths.append(r["rel_img"])
        depth_paths.append(r["rel_dep"])
        intrinsics_list.append(r["K_dummy"])
        poses_list.append(r["pose"])

    n = len(image_paths)
    if n < 2:
        print(f"  [SKIP] {area_name}: {n} valid frames (at least 2 required)")
        return

    depths_mem: list = []
    for i in range(n):
        hp = os.path.join(out_root, depth_paths[i])
        with h5py.File(hp, "r") as f:
            depths_mem.append(f["depth"][:].astype(np.float32))
    poses_np = np.asarray(poses_list, dtype=np.float64)

    # --- full pair pool (EDM overlap: one-way i→j) ---
    print(f"  [{area_name}] generating pairs (N={n}, EDM depth overlap)...")
    pair_cands = [(i, j) for i in range(n) for j in range(i + 1, n)]
    base_task = {
        "depths_mem": depths_mem,
        "poses_np": poses_np,
        "overlap_threshold": overlap_threshold,
        "min_overlap": min_overlap,
        "max_overlap": max_overlap,
        "max_baseline": max_baseline,
    }
    all_pairs, all_overlaps = [], []
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futs = {
                ex.submit(_s2d3_pair_overlap, {**base_task, "i": i, "j": j}): (i, j)
                for i, j in pair_cands
            }
            for f in tqdm(
                as_completed(futs),
                total=len(futs),
                desc=f"[{area_name}] pair overlap",
                unit="pair",
                leave=False,
            ):
                res = f.result()
                if res is not None:
                    i, j, ov = res
                    all_pairs.append([i, j])
                    all_overlaps.append(ov)
    else:
        for i, j in tqdm(
            pair_cands,
            desc=f"[{area_name}] pair overlap",
            unit="pair",
            leave=False,
        ):
            res = _s2d3_pair_overlap({**base_task, "i": i, "j": j})
            if res is not None:
                ii, jj, ov = res
                all_pairs.append([ii, jj])
                all_overlaps.append(ov)

    order = sorted(range(len(all_pairs)), key=lambda k: (all_pairs[k][0], all_pairs[k][1]))
    all_pairs = [all_pairs[k] for k in order]
    all_overlaps = [all_overlaps[k] for k in order]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_pairs))
    if len(perm) > max_pairs:
        perm = perm[:max_pairs]

    all_pairs    = [all_pairs[k]    for k in perm]
    all_overlaps = [all_overlaps[k] for k in perm]
    total = len(all_pairs)

    # --- train / val / test split ---
    corpus = _AREA_TO_CORPUS.get(area_name, "train")
    if by_corpus:
        # Assign ALL pairs to a single split according to the area's corpus
        # (one train/val/test definition per area; same convention as matterport3d)
        pairs_train    = all_pairs    if corpus == "train" else []
        overlaps_train = all_overlaps if corpus == "train" else []
        pairs_val      = all_pairs    if corpus == "val"   else []
        overlaps_val   = all_overlaps if corpus == "val"   else []
        pairs_test     = all_pairs    if corpus == "test"  else []
        overlaps_test  = all_overlaps if corpus == "test"  else []
    else:
        # legacy: ratio-based split inside the area
        n_test  = max(1, int(total * test_ratio))
        n_val   = max(1, int(total * val_ratio))
        n_train = total - n_val - n_test
        pairs_train    = all_pairs[:n_train]
        overlaps_train = all_overlaps[:n_train]
        pairs_val      = all_pairs[n_train : n_train + n_val]
        overlaps_val   = all_overlaps[n_train : n_train + n_val]
        pairs_test     = all_pairs[n_train + n_val :]
        overlaps_test  = all_overlaps[n_train + n_val :]

    print(f"    [{corpus}] train={len(pairs_train)}, val={len(pairs_val)}, test={len(pairs_test)}")

    def _arr2d(lst):
        return np.array(lst, dtype=np.int64) if lst else np.zeros((0, 2), dtype=np.int64)

    def _arr1d(lst):
        return np.array(lst, dtype=np.float64) if lst else np.zeros(0, dtype=np.float64)

    # ERP cam-frame convention harmonization with sccm runtime (utils_sphere.py:
    # lat = (0.5 - v/H)*pi, top of image = lat=+pi/2 → cam +y points UP).
    # S2D3D's camera_rt_matrix stores poses in a cam frame with +y DOWN (matching the
    # internal preprocess overlap dirs grid, lat = (v/H - 0.5)*pi). Apply a y-flip
    # (diag(1,-1,1)) on the left of each w2c pose so the stored convention matches
    # the runtime ERP cam frame (cf. matterport3d/preprocess.py:_GL_CAM_TO_PY360_DIR).
    _y_flip_4 = np.diag([1.0, -1.0, 1.0, 1.0])
    poses_runtime = [_y_flip_4 @ p for p in poses_list]

    scene_info = {
        "scene":          area_name,
        "scene_corpus":   _AREA_TO_CORPUS.get(area_name, "train"),
        "image_paths":    np.array(image_paths),
        "depth_paths":    np.array(depth_paths),
        "intrinsics":     np.array(intrinsics_list),
        "poses":          np.array(poses_runtime),
        "pairs_train":    _arr2d(pairs_train),
        "overlaps_train": _arr1d(overlaps_train),
        "pairs_val":      _arr2d(pairs_val),
        "overlaps_val":   _arr1d(overlaps_val),
        "pairs_test":     _arr2d(pairs_test),
        "overlaps_test":  _arr1d(overlaps_test),
    }

    prep_dir = os.path.join(out_root, "prep_scene_info")
    os.makedirs(prep_dir, exist_ok=True)
    npy_path = os.path.join(prep_dir, f"s2d3d_{area_name}.npy")
    np.save(npy_path, scene_info)
    print(f"  → {npy_path}  ({n} frames)")

    if visualize:
        vis_pairs = (
            pairs_train
            if len(pairs_train) > 0
            else (pairs_val if len(pairs_val) > 0 else pairs_test)
        )
        if len(vis_pairs) > 0:
            visualize_s2d3d_area(
                area_name=area_name,
                out_root=out_root,
                image_paths=image_paths,
                depth_paths=depth_paths,
                poses=scene_info["poses"],
                pairs_for_vis=list(vis_pairs),
                vis_n_pairs=vis_n_pairs,
                n_kpts=vis_n_kpts,
                seed=seed,
            )
        else:
            print("  [VIS SKIP] no pairs to visualize")


def _process_s2d3d_area(payload):
    """Per-area conversion for ProcessPoolExecutor (picklable)."""
    area_name, dataset_root, out_root, kwargs = payload
    convert_area(
        area_name=area_name,
        dataset_root=dataset_root,
        out_root=out_root,
        **kwargs,
    )
    return area_name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stanford 2D-3D-Semantics → MegaDepth/RoMa format converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # single area_1 conversion — EDM evaluation protocol (640×320, overlap>50%, threshold 0.1)
  python3 data/stanford2d3d/preprocess.py \\
      --dataset_root "$SCCM_DATA_ROOT/Stanford 2D-3D-Semantics Dataset" \\
      --out_root /mnt/datasets/stanford2d3d_megadepth \\
      --areas area_1 \\
      --out_w 640 --out_h 320 \\
      --min_overlap 0.5 --max_overlap 1.0 --overlap_threshold 0.1 \\
      --visualize --vis_n_pairs 3 --vis_n_kpts 300

  # all areas (the area_* directories must already be extracted under dataset_root)
  python3 data/stanford2d3d/preprocess.py \\
      --dataset_root "$SCCM_DATA_ROOT/Stanford 2D-3D-Semantics Dataset" \\
      --out_root /mnt/datasets/stanford2d3d_megadepth \\
      --split all \\
      --out_w 640 --out_h 320 \\
      --min_overlap 0.5 --max_overlap 1.0 --overlap_threshold 0.1 \\
      --visualize --vis_n_pairs 3
        """,
    )
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--out_root",     required=True)
    parser.add_argument("--areas",        nargs="+", default=None,
                        help="list of areas to process (overrides --split when given)")
    parser.add_argument("--split",        default="all",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--no_resize",    action="store_true")
    parser.add_argument("--out_h",        type=int,   default=320,
                        help="output ERP height (EDM: 320, default 320)")
    parser.add_argument("--out_w",        type=int,   default=640,
                        help="output ERP width (EDM: 640, default 640)")
    parser.add_argument(
        "--min_overlap",
        type=float,
        default=0.5,
        help="lower bound on pair overlap (EDM Stanford test: >50% → default 0.5)",
    )
    parser.add_argument(
        "--max_overlap",
        type=float,
        default=1.0,
        help="upper bound on pair overlap (default 1.0)",
    )
    parser.add_argument("--max_pairs",    type=int,   default=100_000)
    parser.add_argument(
        "--max_baseline",
        type=float,
        default=0.0,
        help="if greater than 0, skip pairs whose camera-center distance (m) exceeds this value (default 0=disabled)",
    )
    parser.add_argument(
        "--overlap_threshold",
        type=float,
        default=0.1,
        help="EDM relative-depth inlier threshold |d_proj-d_B|/d_B (default 0.1)",
    )
    parser.add_argument("--val_ratio",    type=float, default=0.1)
    parser.add_argument("--test_ratio",   type=float, default=0.1)
    parser.add_argument(
        "--by_corpus",
        action="store_true",
        help="if True, assign ALL pairs of an area to its corpus (=AREA_SPLIT) as a single split (disables the intra-area split)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="global random seed (pair shuffling, max_pairs, visualization keypoints; reproducible for identical arguments)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="save verification images under <out_root>/vis/s2d3d_<area>/",
    )
    parser.add_argument("--vis_n_pairs",  type=int,   default=3)
    parser.add_argument("--vis_n_kpts",   type=int,   default=300)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="multiple areas: ProcessPool workers=min(num_workers, number of areas). The ThreadPool inside each area (frames, pairs) uses the same value (sequential if 1).",
    )
    parser.add_argument(
        "--all_to_test",
        action="store_true",
        help="EDM-style: force all 7 areas into the test corpus → a single pairs_test pool (disables the intra-area split).",
    )
    args = parser.parse_args()

    # EDM-style: override AREA_SPLIT so every area's pairs land in pairs_test
    if args.all_to_test:
        AREA_SPLIT["train"] = []
        AREA_SPLIT["val"]   = []
        AREA_SPLIT["test"]  = list(AREAS)
        _AREA_TO_CORPUS.clear()
        _AREA_TO_CORPUS.update({a: "test" for a in AREAS})
        args.by_corpus = True   # consolidate all pairs of each area into its (test) corpus
        print("[--all_to_test] All 7 areas → test corpus (EDM-style consolidated).")

    if args.areas:
        areas = args.areas
    elif args.split == "all":
        areas = AREAS
    else:
        areas = AREA_SPLIT[args.split]

    print(f"Number of areas to process: {len(areas)}")

    kwargs = dict(
        out_h=args.out_h,
        out_w=args.out_w,
        min_overlap=args.min_overlap,
        max_overlap=args.max_overlap,
        max_pairs=args.max_pairs,
        max_baseline=args.max_baseline,
        overlap_threshold=args.overlap_threshold,
        by_corpus=args.by_corpus,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        resize=not args.no_resize,
        visualize=args.visualize,
        vis_n_pairs=args.vis_n_pairs,
        vis_n_kpts=args.vis_n_kpts,
        num_workers=args.num_workers,
    )

    if args.num_workers <= 1:
        for area_name in tqdm(areas, desc="scene/area conversion", unit="area", leave=True):
            print(f"\nProcessing: {area_name}")
            convert_area(
                area_name=area_name,
                dataset_root=args.dataset_root,
                out_root=args.out_root,
                **kwargs,
            )
    else:
        nw = min(args.num_workers, len(areas))
        payloads = [
            (a, args.dataset_root, args.out_root, kwargs) for a in areas
        ]
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futures = [ex.submit(_process_s2d3d_area, p) for p in payloads]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="scene/area conversion",
                unit="area",
                leave=True,
            ):
                fut.result()

    print("\nDone.")


if __name__ == "__main__":
    main()

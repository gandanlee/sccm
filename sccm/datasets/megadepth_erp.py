"""
ERP (Equirectangular) dataset loader for RoMa.

Loads Matterport3D / Stanford2D3D / OB3D scenes preprocessed into
MegaDepth-compatible format with ERP images and radial depth.

Key differences from MegadepthScene:
- No intrinsic matrix K (ERP is defined by image resolution)
- 2:1 aspect ratio (width = 2 * height)
- Yaw augmentation via circular shift (ERP-native)
- Horizontal flip adjusts pose (reflects x-axis)
- scene_info uses pairs_train/pairs_val/pairs_test keys
"""

import os
import math

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as tvf
from PIL import Image

import sccm
from sccm.utils import get_depth_tuple_transform_ops, get_tuple_transform_ops


class ERPScene:
    def __init__(
        self,
        data_root,
        scene_info,
        ht=512,
        wt=1024,
        min_overlap=0.0,
        max_overlap=1.0,
        normalize=True,
        max_num_pairs=100_000,
        scene_name=None,
        pair_split="train",
        use_horizontal_flip_aug=False,
        use_yaw_aug=True,
        colorjiggle_params=None,
        random_eraser=None,
        randomize_size=False,
        **kwargs,  # absorb unused MegadepthScene kwargs
    ) -> None:
        self.data_root = data_root
        self.scene_name = os.path.splitext(scene_name)[0] + f"_{min_overlap}_{max_overlap}"
        self.image_paths = scene_info["image_paths"]
        self.depth_paths = scene_info.get("depth_paths")
        self.intrinsics = scene_info["intrinsics"]
        self.poses = scene_info["poses"]
        # Per-scene depth unit correction (default 1.0). PanoCity stores depth ~10x
        # smaller than the pose scale, so it is fixed with scene_info['depth_scale']=10.
        self.depth_scale = float(scene_info.get("depth_scale", 1.0))

        # Resolve pairs for the requested split
        pairs_key = f"pairs_{pair_split}"
        overlaps_key = f"overlaps_{pair_split}"
        if pairs_key in scene_info and len(scene_info[pairs_key]) > 0:
            self.pairs = np.asarray(scene_info[pairs_key])
            self.overlaps = np.asarray(scene_info[overlaps_key])
        elif "pairs" in scene_info:
            # Legacy format
            self.pairs = np.asarray(scene_info["pairs"])
            self.overlaps = np.asarray(scene_info["overlaps"])
        else:
            self.pairs = np.zeros((0, 2), dtype=np.int64)
            self.overlaps = np.zeros(0, dtype=np.float64)

        # Filter by overlap
        threshold = (self.overlaps > min_overlap) & (self.overlaps < max_overlap)
        self.pairs = self.pairs[threshold]
        self.overlaps = self.overlaps[threshold]

        if len(self.pairs) > max_num_pairs:
            pairinds = np.random.choice(
                np.arange(0, len(self.pairs)), max_num_pairs, replace=False
            )
            self.pairs = self.pairs[pairinds]
            self.overlaps = self.overlaps[pairinds]

        if randomize_size:
            area = ht * wt
            s = int(16 * (math.sqrt(area) // 16))
            sizes = ((ht, wt), (s, s), (wt, ht))
            ht, wt = sizes[sccm.RANK % 3]

        self.im_transform_ops = get_tuple_transform_ops(
            resize=(ht, wt), normalize=normalize,
            colorjiggle_params=colorjiggle_params,
        )
        self.depth_transform_ops = get_depth_tuple_transform_ops(resize=(ht, wt))
        self.wt, self.ht = wt, ht
        self.use_horizontal_flip_aug = use_horizontal_flip_aug
        self.use_yaw_aug = use_yaw_aug
        self.random_eraser = random_eraser

    def load_im(self, im_path):
        return Image.open(im_path)

    def load_depth(self, depth_ref):
        depth = np.array(h5py.File(depth_ref, "r")["depth"])
        return torch.from_numpy(depth).float() * self.depth_scale

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, pair_idx):
        idx1, idx2 = self.pairs[pair_idx]

        # Poses: world-to-camera
        T1 = self.poses[idx1]
        T2 = self.poses[idx2]
        T_1to2 = torch.tensor(
            np.matmul(T2, np.linalg.inv(T1)), dtype=torch.float
        )[:4, :4]

        # Load images
        im_A_ref = os.path.join(self.data_root, self.image_paths[idx1])
        im_B_ref = os.path.join(self.data_root, self.image_paths[idx2])
        im_A = self.load_im(im_A_ref)
        im_B = self.load_im(im_B_ref)

        # Load depth (radial distance)
        if self.depth_paths is not None and len(self.depth_paths) > 0:
            depth_A_ref = os.path.join(self.data_root, self.depth_paths[idx1])
            depth_B_ref = os.path.join(self.data_root, self.depth_paths[idx2])
            depth_A = self.load_depth(depth_A_ref)
            depth_B = self.load_depth(depth_B_ref)
        else:
            depth_A = torch.zeros(im_A.height, im_A.width, dtype=torch.float32)
            depth_B = torch.zeros(im_B.height, im_B.width, dtype=torch.float32)

        # Apply image transforms (resize + normalize)
        im_A, im_B = self.im_transform_ops((im_A, im_B))
        depth_A, depth_B = self.depth_transform_ops(
            (depth_A[None, None], depth_B[None, None])
        )

        im_A, im_B = im_A[None], im_B[None]

        if self.random_eraser is not None:
            im_A, depth_A = self.random_eraser(im_A, depth_A)
            im_B, depth_B = self.random_eraser(im_B, depth_B)

        # ERP yaw augmentation: independent random circular shift per frame
        # (EDM-style fair regularization). Each frame's camera is rotated
        # independently about its y-axis, so the relative pose T_1to2 actually
        # changes — providing real augmentation to yaw-equivariant networks.
        # cf. EDM (CVPR'25, §4.3): θ_A^aug, θ_B^aug ∈ [0, 2π) drawn independently.
        if self.use_yaw_aug:
            delta_A = int(np.random.randint(0, self.wt))
            delta_B = int(np.random.randint(0, self.wt))
            if delta_A > 0 or delta_B > 0:
                theta_A = delta_A * 2.0 * math.pi / self.wt
                theta_B = delta_B * 2.0 * math.pi / self.wt
                cos_A, sin_A = math.cos(theta_A), math.sin(theta_A)
                cos_B, sin_B = math.cos(theta_B), math.sin(theta_B)
                Ry_A = T_1to2.new_tensor([
                    [cos_A, 0, sin_A, 0],
                    [0, 1, 0, 0],
                    [-sin_A, 0, cos_A, 0],
                    [0, 0, 0, 1],
                ])
                Ry_B = T_1to2.new_tensor([
                    [cos_B, 0, sin_B, 0],
                    [0, 1, 0, 0],
                    [-sin_B, 0, cos_B, 0],
                    [0, 0, 0, 1],
                ])
                # Existing same-delta convention: T = Ry @ T @ Ry^T preserves
                # relative pose under common world rotation. For independent
                # rotations on A and B (same convention), the relative pose is:
                #   T_1to2_new = Ry_B @ T_1to2 @ Ry_A^T
                # which reduces to the existing form when Ry_A = Ry_B = Ry.
                T_1to2 = Ry_B @ T_1to2 @ Ry_A.T
                im_A = torch.roll(im_A, delta_A, dims=-1)
                im_B = torch.roll(im_B, delta_B, dims=-1)
                depth_A = torch.roll(depth_A, delta_A, dims=-1)
                depth_B = torch.roll(depth_B, delta_B, dims=-1)

        # Horizontal flip: reflects scene left-right
        if self.use_horizontal_flip_aug and np.random.rand() > 0.5:
            im_A = im_A.flip(-1)
            im_B = im_B.flip(-1)
            depth_A = depth_A.flip(-1)
            depth_B = depth_B.flip(-1)
            # ERP flip = lon -> -lon = x-axis reflection
            S = T_1to2.new_tensor([[-1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
            T_1to2 = S @ T_1to2 @ S

        if sccm.DEBUG_MODE:
            from sccm.utils import tensor_to_pil
            os.makedirs("vis", exist_ok=True)
            tensor_to_pil(im_A[0], unnormalize=True).save("vis/im_A.jpg")
            tensor_to_pil(im_B[0], unnormalize=True).save("vis/im_B.jpg")

        # Dummy K for interface compatibility (not used in ERP loss)
        K_dummy = torch.eye(3, dtype=torch.float32)

        return {
            "im_A": im_A[0],
            "im_A_identifier": self.image_paths[idx1].split("/")[-1].split(".")[0],
            "im_B": im_B[0],
            "im_B_identifier": self.image_paths[idx2].split("/")[-1].split(".")[0],
            "im_A_depth": depth_A[0, 0],
            "im_B_depth": depth_B[0, 0],
            "K1": K_dummy,
            "K2": K_dummy,
            "T_1to2": T_1to2,
            "im_A_path": im_A_ref,
            "im_B_path": im_B_ref,
        }


class ERPBuilder:
    """Builds ERPScene datasets from preprocessed scene_info .npy files.

    Expected data_root structure:
        data_root/
            prep_scene_info/
                mp3d_<scene_id>.npy
                s2d3d_<area>.npy
                ob3d_<scene>.npy
            <scene_folder>/
                images/*.jpg
                depths/*.h5
    """

    def __init__(self, data_root, scene_prefix=None) -> None:
        self.data_root = data_root
        self.scene_info_root = os.path.join(data_root, "prep_scene_info")
        if not os.path.isdir(self.scene_info_root):
            raise FileNotFoundError(
                f"prep_scene_info not found: {self.scene_info_root}"
            )
        all_files = sorted(os.listdir(self.scene_info_root))
        self.all_scenes = [
            f for f in all_files
            if f.endswith(".npy") and "_pairs" not in f
        ]
        if scene_prefix:
            self.all_scenes = [
                f for f in self.all_scenes if f.startswith(scene_prefix)
            ]

    def build_scenes(self, split="train", scene_names=None,
                     scene_corpus_filter=None, **kwargs):
        """Build list of ERPScene datasets.

        Args:
            split: "train" | "val" | "test"
            scene_names: explicit list of .npy filenames (overrides auto-discovery)
            scene_corpus_filter: filter by scene_info["scene_corpus"]
            **kwargs: passed to ERPScene (ht, wt, min_overlap, etc.)
        """
        names = scene_names if scene_names is not None else self.all_scenes

        scenes = []
        for scene_name in names:
            if not scene_name.endswith(".npy"):
                scene_name = scene_name + ".npy"
            scene_path = os.path.join(self.scene_info_root, scene_name)
            if not os.path.exists(scene_path):
                continue
            scene_info = np.load(scene_path, allow_pickle=True).item()

            if scene_corpus_filter is not None:
                corpus = scene_info.get("scene_corpus", "train")
                if corpus != scene_corpus_filter:
                    continue

            scenes.append(
                ERPScene(
                    self.data_root,
                    scene_info,
                    scene_name=scene_name,
                    pair_split=split,
                    **kwargs,
                )
            )
        return scenes

    def weight_scenes(self, concat_dataset, alpha=0.5):
        ns = [len(d) for d in concat_dataset.datasets]
        return torch.cat([torch.ones(n) / n ** alpha for n in ns])

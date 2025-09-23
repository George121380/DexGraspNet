import os
import math
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _rotation_z(theta: float) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float)


class AffordancePairsDataset(Dataset):
    """
    Dataset that expands a single aff_sec_pairs.npy file into per-pair samples.

    Each item returns:
      - points: (N, 3) float tensor
      - center: (3,) float tensor (single conditioning point)
      - target: (N,) float tensor (affordance scores in [0,1])

    Optional augmentations can be enabled via constructor flags.
    """

    def __init__(
        self,
        npy_path: str,
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        augment_rotate_z: bool = False,
        augment_jitter_std: float = 0.0,
        shuffle_points: bool = True,
        only_pair_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not os.path.isfile(npy_path):
            raise FileNotFoundError(f"File not found: {npy_path}")
        data = np.load(npy_path, allow_pickle=True).item()

        self.points_np: np.ndarray = data["points"].astype(np.float32)  # (N,3)
        pairs_db: Dict = data["pairs"]

        # Expand to a list of pairs
        pair_indices: List[int] = sorted([int(k) for k in pairs_db.keys()])

        # Split indices deterministically
        if only_pair_idx is not None:
            # Overfit mode: use a single specified pair index for the dataset
            k = int(only_pair_idx)
            if k not in pair_indices:
                raise KeyError(f"Requested pair index {k} not found in pairs (available: {pair_indices})")
            self.active_indices = [k]
        else:
            rng = random.Random(seed)
            rng.shuffle(pair_indices)
            split_idx = int((1.0 - val_ratio) * len(pair_indices))
            if split == "train":
                self.active_indices = pair_indices[:split_idx]
            elif split == "val":
                self.active_indices = pair_indices[split_idx:]
            else:
                self.active_indices = pair_indices

        self.pairs = pairs_db

        # Augmentation flags
        self.augment_rotate_z = bool(augment_rotate_z)
        self.augment_jitter_std = float(augment_jitter_std)
        self.shuffle_points = bool(shuffle_points)

        # Cache tensor version of base points for speed
        self.points_base = torch.from_numpy(self.points_np.copy())  # (N,3)

    def __len__(self) -> int:
        return len(self.active_indices)

    def _apply_augs(self, pts: torch.Tensor, center: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Rotation around Z-axis
        if self.augment_rotate_z:
            theta = random.uniform(-math.pi, math.pi)
            Rz = _rotation_z(theta).to(dtype=pts.dtype, device=pts.device)
            pts = pts @ Rz.T
            center = center @ Rz.T

        # Jitter
        if self.augment_jitter_std > 0.0:
            noise = torch.randn_like(pts) * self.augment_jitter_std
            pts = pts + noise

        return pts, center

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair_id = self.active_indices[idx]
        pair = self.pairs[pair_id]

        # Single center point (1,3) -> (3,)
        center = torch.from_numpy(pair["left_kps"].astype(np.float32)).view(-1, 3)[0]
        target = torch.from_numpy(pair["aff_scores_right"].astype(np.float32))  # (N,)

        pts = self.points_base.clone()

        # Optional shuffling (keep alignment between pts and target)
        if self.shuffle_points:
            perm = torch.randperm(pts.shape[0])
            pts = pts[perm]
            target = target[perm]

        # Apply simple augmentations
        pts, center = self._apply_augs(pts, center)

        return pts, center, target


class AffordancePairsMultiDataset(Dataset):
    """
    Dataset that aggregates multiple aff_sec_pairs.npy files under a directory.

    Each item returns:
      - points: (N, 3) float tensor
      - center: (3,) float tensor (first conditioning point)
      - target: (N,) float tensor
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        augment_rotate_z: bool = False,
        augment_jitter_std: float = 0.0,
        shuffle_points: bool = True,
        only_pair_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f"Directory not found: {data_dir}")

        # Collect all aff_sec_pairs.npy under immediate subdirs
        npy_files: List[str] = []
        for name in sorted(os.listdir(data_dir)):
            sub = os.path.join(data_dir, name)
            if os.path.isdir(sub):
                f = os.path.join(sub, 'aff_sec_pairs.npy')
                if os.path.isfile(f):
                    npy_files.append(f)
        if len(npy_files) == 0:
            raise FileNotFoundError(f"No aff_sec_pairs.npy found under: {data_dir}")

        # Build flat list of samples
        samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for fpath in npy_files:
            data = np.load(fpath, allow_pickle=True).item()
            points_np: np.ndarray = data["points"].astype(np.float32)  # (N,3)
            pairs_db: Dict = data["pairs"]
            for k in sorted([int(x) for x in pairs_db.keys()]):
                pair = pairs_db[k]
                centers = pair["left_kps"].astype(np.float32).reshape(-1, 3)
                center = centers[0]
                target = pair["aff_scores_right"].astype(np.float32)
                samples.append((points_np, center, target))

        # Determine active indices
        all_indices: List[int] = list(range(len(samples)))
        if only_pair_idx is not None:
            k = int(only_pair_idx)
            if not (0 <= k < len(samples)):
                raise KeyError(f"Requested pair index {k} out of range [0,{len(samples)-1}]")
            self.active_indices = [k]
        else:
            rng = random.Random(seed)
            rng.shuffle(all_indices)
            split_idx = int((1.0 - val_ratio) * len(all_indices))
            if split == "train":
                self.active_indices = all_indices[:split_idx]
            elif split == "val":
                self.active_indices = all_indices[split_idx:]
            else:
                self.active_indices = all_indices

        # Persist data
        self.samples = samples
        self.augment_rotate_z = bool(augment_rotate_z)
        self.augment_jitter_std = float(augment_jitter_std)
        self.shuffle_points = bool(shuffle_points)

    def __len__(self) -> int:
        return len(self.active_indices)

    def _apply_augs(self, pts: torch.Tensor, center: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.augment_rotate_z:
            theta = random.uniform(-math.pi, math.pi)
            Rz = _rotation_z(theta).to(dtype=pts.dtype, device=pts.device)
            pts = pts @ Rz.T
            center = center @ Rz.T
        if self.augment_jitter_std > 0.0:
            noise = torch.randn_like(pts) * self.augment_jitter_std
            pts = pts + noise
        return pts, center

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_idx = self.active_indices[idx]
        points_np, center_np, target_np = self.samples[sample_idx]

        pts = torch.from_numpy(points_np.copy())
        center = torch.from_numpy(center_np.copy())
        target = torch.from_numpy(target_np.copy())

        if self.shuffle_points:
            perm = torch.randperm(pts.shape[0])
            pts = pts[perm]
            target = target[perm]

        pts, center = self._apply_augs(pts, center)
        return pts, center, target


__all__ = ["AffordancePairsDataset", "AffordancePairsMultiDataset"]







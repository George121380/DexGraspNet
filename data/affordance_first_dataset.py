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


def _get_first_center_key(pair: Dict) -> str:
    # Try common key names for the first hand center point
    candidate_keys = [
        "left_kps",  # preferred for first-hand
        "left_center",
        "first_kps",
        "kps_left",
        "kps",
        "center",
    ]
    for k in candidate_keys:
        if k in pair:
            return k
    # Fallback: pick the first array-like item
    for k, v in pair.items():
        try:
            arr = np.asarray(v)
            if arr.ndim >= 1 and arr.shape[-1] == 3:
                return k
        except Exception:
            continue
    raise KeyError("Could not locate a center key for first-hand in pair entry")


def _get_first_target_key(pair: Dict, *, expected_num_points: Optional[int] = None) -> str:
    # Try common key names for targets for the first hand
    candidate_keys = [
        "aff_scores_left",
        "aff_scores_first",
        "scores_left",
        "targets_left",
        "targets",
        "aff_scores",
    ]
    # Prefer candidate keys that exist and (if provided) match expected length (by total size)
    for k in candidate_keys:
        if k in pair:
            if expected_num_points is not None:
                try:
                    arr = np.asarray(pair[k])
                    if arr.size == expected_num_points:
                        return k
                except Exception:
                    pass
            else:
                return k
    # Fallback: choose the first array whose total size matches expected length;
    # if expected length not provided or not matched, raise
    for k, v in pair.items():
        try:
            arr = np.asarray(v)
            if expected_num_points is not None and arr.size == expected_num_points:
                return k
        except Exception:
            continue
    raise KeyError("Could not locate a target key for first-hand in pair entry matching number of points")


class AffordanceFirstPairsDataset(Dataset):
    """
    Dataset that expands a single aff_first_pairs_left.npy file into per-pair samples.

    Each item returns:
      - points: (N, 3) float tensor
      - center: (3,) float tensor (first-hand conditioning point)
      - target: (N,) float tensor
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
        center_key_override: Optional[str] = None,
        target_key_override: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not os.path.isfile(npy_path):
            raise FileNotFoundError(f"File not found: {npy_path}")

        data = np.load(npy_path, allow_pickle=True).item()

        # Required top-level keys
        if "points" not in data:
            raise KeyError("'points' key missing in npy file")
        if "pairs" not in data or not isinstance(data["pairs"], dict):
            raise KeyError("'pairs' key missing or not a dict in npy file")

        self.points_np: np.ndarray = data["points"].astype(np.float32)  # (N,3)
        pairs_db: Dict = data["pairs"]

        # Expand to a list of pair indices
        pair_indices: List[int] = sorted([int(k) for k in pairs_db.keys()])

        # Determine active indices
        if only_pair_idx is not None:
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

        # Defer key detection to __getitem__ to avoid upfront failures and allow per-pair flexibility
        self._center_key = None  # type: ignore[assignment]
        self._target_key = None  # type: ignore[assignment]
        self._override_center_key = center_key_override
        self._override_target_key = target_key_override

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
        pair_id = self.active_indices[idx]
        pair = self.pairs[pair_id]

        # Detect keys lazily (respect overrides if provided)
        center_key = self._override_center_key or self._center_key or _get_first_center_key(pair)
        target_key = self._override_target_key or self._target_key or _get_first_target_key(pair, expected_num_points=self.points_base.shape[0])
        # Center point handling: allow (3,) or (1,3) or (K,3) -> take the first
        center_np = np.asarray(pair[center_key], dtype=np.float32).reshape(-1, 3)[0]
        target_np = np.asarray(pair[target_key], dtype=np.float32).reshape(-1)

        pts = self.points_base.clone()
        center = torch.from_numpy(center_np)
        target = torch.from_numpy(target_np)

        # One-time debug print for the first accessed item to help diagnose schema
        if not hasattr(self, "_debug_printed"):
            self._debug_printed = True  # type: ignore[attr-defined]
            shapes = {}
            for kk, vv in pair.items():
                try:
                    aa = np.asarray(vv)
                    shapes[kk] = tuple(aa.shape)
                except Exception:
                    shapes[kk] = str(type(vv))
            print(
                "[AffFirstDataset Debug] N=", pts.shape[0],
                "center_key=", center_key,
                "center_shape=", tuple(center_np.reshape(-1,3).shape),
                "target_key=", target_key,
                "target_shape=", tuple(target_np.shape),
                "pair_shapes=", shapes,
                flush=True,
            )

        if self.shuffle_points:
            perm = torch.randperm(pts.shape[0])
            pts = pts[perm]
            target = target[perm]

        # Ensure target length matches number of points; if not, try to recover by scanning keys
        if target.numel() != pts.shape[0]:
            N = int(pts.shape[0])
            recovered = None
            for k, v in pair.items():
                try:
                    arr = np.asarray(v)
                    if arr.size == N:
                        recovered = torch.from_numpy(arr.astype(np.float32).reshape(-1))
                        break
                except Exception:
                    continue
            if recovered is not None:
                target = recovered
                if self.shuffle_points:
                    perm = torch.randperm(N)
                    target = target[perm]
            else:
                # Collect shapes for debugging
                shapes = {}
                for kk, vv in pair.items():
                    try:
                        aa = np.asarray(vv)
                        shapes[kk] = tuple(aa.shape)
                    except Exception:
                        shapes[kk] = str(type(vv))
                raise ValueError(
                    f"Target length mismatch: got {target.numel()} but points have {N}. "
                    f"Pair keys and shapes: {shapes}"
                )

        pts, center = self._apply_augs(pts, center)
        return pts, center, target


class AffordanceFirstPairsMultiDataset(Dataset):
    """
    Dataset that aggregates multiple aff_first_pairs_left.npy files under a directory.

    Each item returns:
      - points: (N, 3) float tensor
      - center: (3,) float tensor (first-hand conditioning point)
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
        center_key_override: Optional[str] = None,
        target_key_override: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f"Directory not found: {data_dir}")

        # Collect all aff_first_pairs_left.npy under immediate subdirs
        npy_files: List[str] = []
        for name in sorted(os.listdir(data_dir)):
            sub = os.path.join(data_dir, name)
            if os.path.isdir(sub):
                f = os.path.join(sub, 'aff_first_pairs_left.npy')
                if os.path.isfile(f):
                    npy_files.append(f)
        if len(npy_files) == 0:
            raise FileNotFoundError(f"No aff_first_pairs_left.npy found under: {data_dir}")

        # Build a lazy index instead of materializing all arrays to reduce startup time and memory
        # For each file, we record detected keys once; samples store (fpath, pair_id)
        file_keys: Dict[str, Tuple[str, str]] = {}
        sample_index: List[Tuple[str, int]] = []
        for fpath in npy_files:
            data = np.load(fpath, allow_pickle=True).item()
            if "points" not in data or "pairs" not in data:
                continue
            points_np: np.ndarray = data["points"].astype(np.float32)
            pairs_db: Dict = data["pairs"]
            if len(pairs_db) == 0:
                continue
            k0 = sorted([int(x) for x in pairs_db.keys()])[0]
            p0 = pairs_db[k0]
            center_key = center_key_override or _get_first_center_key(p0)
            target_key = target_key_override or _get_first_target_key(p0, expected_num_points=points_np.shape[0])
            file_keys[fpath] = (center_key, target_key)
            for k in sorted([int(x) for x in pairs_db.keys()]):
                sample_index.append((fpath, int(k)))

        # Determine active indices over the sample_index list
        all_indices: List[int] = list(range(len(sample_index)))
        if only_pair_idx is not None:
            k = int(only_pair_idx)
            if not (0 <= k < len(sample_index)):
                raise KeyError(f"Requested pair index {k} out of range [0,{len(sample_index)-1}]")
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

        self._file_keys = file_keys
        self._sample_index = sample_index
        # Small single-file cache to avoid repeated disk IO when training on a single object
        self._last_file: Optional[str] = None
        self._last_cache: Optional[Dict[str, object]] = None
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
        fpath, pair_id = self._sample_index[sample_idx]

        # Load file lazily with a tiny cache for the last accessed file
        if self._last_file != fpath or self._last_cache is None:
            data = np.load(fpath, allow_pickle=True).item()
            points_np: np.ndarray = data["points"].astype(np.float32)
            pairs_db: Dict = data["pairs"]
            self._last_file = fpath
            self._last_cache = {
                "points": points_np,
                "pairs": pairs_db,
            }
        else:
            points_np = self._last_cache["points"]  # type: ignore[index]
            pairs_db = self._last_cache["pairs"]    # type: ignore[index]

        center_key, target_key = self._file_keys[fpath]
        pair = pairs_db[pair_id]
        centers = np.asarray(pair[center_key], dtype=np.float32).reshape(-1, 3)
        center_np = centers[0]
        target_np = np.asarray(pair[target_key], dtype=np.float32).reshape(-1)

        pts = torch.from_numpy(points_np.copy())
        center = torch.from_numpy(center_np.copy())
        target = torch.from_numpy(target_np.copy())

        if self.shuffle_points:
            perm = torch.randperm(pts.shape[0])
            pts = pts[perm]
            target = target[perm]

        pts, center = self._apply_augs(pts, center)
        return pts, center, target


__all__ = [
    "AffordanceFirstPairsDataset",
    "AffordanceFirstPairsMultiDataset",
]


class AffordanceFirstAggregatedDataset(Dataset):
    """
    Aggregate all pair-level targets in a single aff_first_pairs_<hand>.npy into one multi-modal target.

    Returns one sample:
      - points: (N, 3)
      - center: (3,) mean of all centers (for visualization only)
      - target: (N,) aggregated and normalized
    """

    def __init__(
        self,
        npy_path: str,
        *,
        center_key: Optional[str] = None,
        target_key: Optional[str] = None,
        aggregate_mode: str = "sum",  # "sum" or "max"
        normalize: str = "max",       # "max" | "l1" | "none"
        augment_rotate_z: bool = False,
        augment_jitter_std: float = 0.0,
        shuffle_points: bool = False,
    ) -> None:
        super().__init__()
        if not os.path.isfile(npy_path):
            raise FileNotFoundError(f"File not found: {npy_path}")

        data = np.load(npy_path, allow_pickle=True).item()
        if "points" not in data or "pairs" not in data:
            raise KeyError("Expected keys 'points' and 'pairs' in aggregated file")

        points_np: np.ndarray = data["points"].astype(np.float32)
        pairs_db: Dict = data["pairs"]
        if len(pairs_db) == 0:
            raise ValueError("No pairs to aggregate")

        # Detect keys from first pair if not provided
        sample_pair = pairs_db[sorted([int(k) for k in pairs_db.keys()])[0]]
        c_key = center_key or _get_first_center_key(sample_pair)
        t_key = target_key or _get_first_target_key(sample_pair, expected_num_points=points_np.shape[0])

        # Aggregate targets across all pairs
        agg = None
        centers = []
        for k in sorted([int(x) for x in pairs_db.keys()]):
            pair = pairs_db[k]
            t = np.asarray(pair[t_key], dtype=np.float32).reshape(-1)
            if agg is None:
                agg = t.copy()
            else:
                if aggregate_mode == "sum":
                    agg += t
                elif aggregate_mode == "max":
                    agg = np.maximum(agg, t)
                else:
                    raise ValueError("aggregate_mode must be one of {'sum','max'}")
            c = np.asarray(pair[c_key], dtype=np.float32).reshape(-1, 3)[0]
            centers.append(c)
        assert agg is not None

        # Normalize
        if normalize == "max":
            m = float(np.max(agg))
            if m > 0:
                agg = agg / m
        elif normalize == "l1":
            s = float(np.sum(agg))
            if s > 0:
                agg = agg / s
        elif normalize == "none":
            pass
        else:
            raise ValueError("normalize must be one of {'max','l1','none'}")

        self.points_base = torch.from_numpy(points_np.copy())
        self.target_base = torch.from_numpy(agg.astype(np.float32))
        centers_np = np.stack(centers, axis=0)
        self.center_mean = torch.from_numpy(centers_np.mean(axis=0).astype(np.float32))

        self.augment_rotate_z = bool(augment_rotate_z)
        self.augment_jitter_std = float(augment_jitter_std)
        self.shuffle_points = bool(shuffle_points)

    def __len__(self) -> int:
        return 1

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
        pts = self.points_base.clone()
        center = self.center_mean.clone()
        target = self.target_base.clone()

        if self.shuffle_points:
            perm = torch.randperm(pts.shape[0])
            pts = pts[perm]
            target = target[perm]

        pts, center = self._apply_augs(pts, center)
        return pts, center, target


class AffordanceFirstAggregatedMultiDataset(Dataset):
    """
    Aggregate targets per object under a directory. One sample per subfolder/object.
    """

    def __init__(
        self,
        data_dir: str,
        *,
        center_key: Optional[str] = None,
        target_key: Optional[str] = None,
        aggregate_mode: str = "sum",
        normalize: str = "max",
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        augment_rotate_z: bool = False,
        augment_jitter_std: float = 0.0,
        shuffle_points: bool = False,
    ) -> None:
        super().__init__()
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f"Directory not found: {data_dir}")

        npy_files: List[str] = []
        for name in sorted(os.listdir(data_dir)):
            sub = os.path.join(data_dir, name)
            if os.path.isdir(sub):
                f = os.path.join(sub, 'aff_first_pairs_left.npy')
                if os.path.isfile(f):
                    npy_files.append(f)
        if len(npy_files) == 0:
            raise FileNotFoundError(f"No aff_first_pairs_left.npy found under: {data_dir}")

        # Build index, but do aggregation lazily on access to reduce startup cost
        self._files = npy_files
        all_indices = list(range(len(self._files)))
        rng = random.Random(seed)
        rng.shuffle(all_indices)
        split_idx = int((1.0 - val_ratio) * len(all_indices))
        if split == "train":
            self.active_indices = all_indices[:split_idx]
        elif split == "val":
            self.active_indices = all_indices[split_idx:]
        else:
            self.active_indices = all_indices

        self.center_key = center_key
        self.target_key = target_key
        self.aggregate_mode = aggregate_mode
        self.normalize = normalize
        self.augment_rotate_z = bool(augment_rotate_z)
        self.augment_jitter_std = float(augment_jitter_std)
        self.shuffle_points = bool(shuffle_points)

        self._cache_fpath: Optional[str] = None
        self._cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None  # pts, center, target

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

    def _aggregate_file(self, fpath: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = np.load(fpath, allow_pickle=True).item()
        points_np: np.ndarray = data["points"].astype(np.float32)
        pairs_db: Dict = data["pairs"]
        if len(pairs_db) == 0:
            raise ValueError(f"No pairs in {fpath}")
        sample_pair = pairs_db[sorted([int(k) for k in pairs_db.keys()])[0]]
        c_key = self.center_key or _get_first_center_key(sample_pair)
        t_key = self.target_key or _get_first_target_key(sample_pair, expected_num_points=points_np.shape[0])

        agg = None
        centers = []
        for k in sorted([int(x) for x in pairs_db.keys()]):
            pair = pairs_db[k]
            t = np.asarray(pair[t_key], dtype=np.float32).reshape(-1)
            if agg is None:
                agg = t.copy()
            else:
                if self.aggregate_mode == "sum":
                    agg += t
                elif self.aggregate_mode == "max":
                    agg = np.maximum(agg, t)
                else:
                    raise ValueError("aggregate_mode must be one of {'sum','max'}")
            c = np.asarray(pair[c_key], dtype=np.float32).reshape(-1, 3)[0]
            centers.append(c)

        assert agg is not None
        if self.normalize == "max":
            m = float(np.max(agg))
            if m > 0:
                agg = agg / m
        elif self.normalize == "l1":
            s = float(np.sum(agg))
            if s > 0:
                agg = agg / s
        elif self.normalize == "none":
            pass
        else:
            raise ValueError("normalize must be one of {'max','l1','none'}")

        pts = torch.from_numpy(points_np.copy())
        center = torch.from_numpy(np.stack(centers, axis=0).mean(axis=0).astype(np.float32))
        target = torch.from_numpy(agg.astype(np.float32))
        return pts, center, target

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fpath = self._files[self.active_indices[idx]]
        if self._cache_fpath != fpath or self._cache is None:
            self._cache_fpath = fpath
            self._cache = self._aggregate_file(fpath)
        pts, center, target = (x.clone() for x in self._cache)

        if self.shuffle_points:
            perm = torch.randperm(pts.shape[0])
            pts = pts[perm]
            target = target[perm]

        pts, center = self._apply_augs(pts, center)
        return pts, center, target



import os
import json
import shutil
from typing import Any, Dict

import numpy as np


def load_point_cloud(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Point cloud not found: {path}")
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Point cloud must have shape (N,3), got {arr.shape}")
    return arr.astype(np.float32)


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_npy(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def save_npz(path: str, **arrays) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)


def snapshot_config(src_paths: Dict[str, str], dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for name, path in src_paths.items():
        if path and os.path.exists(path):
            shutil.copy2(path, os.path.join(dst_dir, f"{name}.yaml"))






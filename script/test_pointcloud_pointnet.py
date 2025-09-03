import os
import sys
import time
import argparse
from typing import Tuple

import numpy as np
import torch


def add_repo_root_to_syspath() -> None:
    """Add project root to sys.path to allow local imports without installation."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


add_repo_root_to_syspath()

from models.point_cloud_op import build_pointnet_extractor  # noqa: E402


def load_points(npy_path: str) -> np.ndarray:
    """Load points from .npy and normalize to shape [B, N, 3] or [B, 3, N]."""
    arr = np.load(npy_path)
    if arr.ndim == 2 and arr.shape[1] == 3:  # [N,3] -> [1,N,3]
        arr = arr[None, ...]
    elif arr.ndim == 2 and arr.shape[0] == 3:  # [3,N] -> [1,N,3]
        arr = arr[None, ...].transpose(0, 2, 1)
    elif arr.ndim == 3 and (arr.shape[-1] == 3 or arr.shape[1] == 3):
        pass
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")
    return arr.astype(np.float32)


def to_b3n(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor shape is [B, 3, N]."""
    if t.dim() != 3:
        raise ValueError(f"Expect 3D tensor, got: {tuple(t.shape)}")
    if t.shape[1] == 3:
        return t
    if t.shape[2] == 3:
        return t.transpose(1, 2).contiguous()
    raise ValueError(f"Expect [B, 3, N] or [B, N, 3], got: {tuple(t.shape)}")


def run_inference(npy_path: str, device: str, use_feature_transform: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a forward pass and return (features, trans, trans_feat)."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but device='cuda' was requested")

    points_np = load_points(npy_path)
    points = torch.from_numpy(points_np)
    points = to_b3n(points)

    extractor = build_pointnet_extractor(use_feature_transform=use_feature_transform, device=device)
    extractor.eval()

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        feats, trans, trans_feat = extractor(points)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) * 1000.0

    print(f"Loaded points from: {npy_path}")
    print(f"Input shape: {tuple(points.shape)}  dtype: {points.dtype}  device: {points.device}")
    print(f"Output features: {tuple(feats.shape)}  device: {feats.device}")
    print(f"Inferred in: {dt:.2f} ms")
    return feats, trans, trans_feat


def main() -> None:
    parser = argparse.ArgumentParser(description="Test PointNet feature extraction on a .npy point cloud")
    parser.add_argument("--path", type=str, default="/home/peiqi621/projects/2026-CVPR-BiDexHand/preprocess/results/2_of_Jenga_Classic_Game/obj_points.npy", help="Path to .npy file containing points")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Execution device")
    parser.add_argument("--feature_transform", action="store_true", help="Enable feature transform module")
    args = parser.parse_args()

    feats, trans, trans_feat = run_inference(args.path, args.device, args.feature_transform)
    # Print summary stats to help debugging numeric sanity
    mean = feats.mean().item()
    std = feats.std().item()
    print(f"Feature stats -> mean: {mean:.4f}, std: {std:.4f}")


if __name__ == "__main__":
    main()




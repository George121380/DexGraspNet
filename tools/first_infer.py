import os
import sys
import argparse
import json
from typing import Optional

import numpy as np
import torch


def set_device(device: str) -> torch.device:
    """Select torch device; 'auto' chooses CUDA if available."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def robust_import_model():
    """Import AffordanceFirstPointNet2SSG with a robust fallback.

    This function ensures that the repository root is on sys.path and attempts to
    import the model definition. If the standard import fails, it tries loading the
    module directly from models/affordance_first.py.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from models.affordance_first import AffordanceFirstPointNet2SSG  # type: ignore
        return AffordanceFirstPointNet2SSG
    except Exception:
        import importlib.util as ilu
        model_path = os.path.join(repo_root, 'models', 'affordance_first.py')
        spec = ilu.spec_from_file_location('affordance_first', model_path)
        assert spec and spec.loader
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return getattr(mod, 'AffordanceFirstPointNet2SSG')


@torch.no_grad()
def run_inference(model_path: str, points_path: str, output_dir: str, device: str = 'auto') -> str:
    """Run first-stage inference on a single preprocessed point cloud.

    Args:
        model_path: Path to a checkpoint containing 'model' state_dict.
        points_path: Path to a .npy file with shape (N, 3) float32/float64.
        output_dir: Directory to save 'scores_first.npy'.
        device: 'auto' | explicit torch device string.

    Returns:
        Path to the saved scores file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Safety settings for older PyTorch builds on newer GPUs
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    target_device = set_device(device)

    # Load points
    points_np = np.load(points_path)
    if points_np.ndim != 2 or points_np.shape[1] != 3:
        raise ValueError(f"Expected points of shape (N,3), got {points_np.shape}")
    points = torch.from_numpy(points_np[None, ...]).float().to(target_device)  # (1, N, 3)

    # Build model
    AffordanceFirstPointNet2SSG = robust_import_model()
    model = AffordanceFirstPointNet2SSG(use_xyz=True)
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(target_device)
    model.eval()

    # Forward
    logits = model(points)  # (1, 1, N)
    scores = torch.sigmoid(logits.squeeze(1))  # (1, N)
    scores_np = scores.squeeze(0).detach().cpu().numpy()  # (N,)

    out_path = os.path.join(output_dir, 'scores_first.npy')
    np.save(out_path, scores_np)

    # Optional manifest for debugging
    manifest = {
        'points_path': os.path.abspath(points_path),
        'model_path': os.path.abspath(model_path),
        'output_scores': os.path.abspath(out_path),
        'num_points': int(points_np.shape[0]),
        'device': str(target_device),
    }
    with open(os.path.join(output_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    return out_path


def main():
    parser = argparse.ArgumentParser(description='First-stage affordance inference on a single point cloud')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--points_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    run_inference(
        model_path=os.path.abspath(args.model_path),
        points_path=os.path.abspath(args.points_path),
        output_dir=os.path.abspath(args.output_dir),
        device=args.device,
    )


if __name__ == '__main__':
    main()





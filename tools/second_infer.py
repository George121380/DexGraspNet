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
    """Import AffordancePointNet2SSG (second-stage) with a robust fallback.

    This function ensures that the repository root is on sys.path and attempts to
    import the second-stage model definition. If the standard import fails, it tries
    loading the module directly from models/affordance_second.py.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from models.affordance_second import AffordancePointNet2SSG  # type: ignore
        return AffordancePointNet2SSG
    except Exception:
        import importlib.util as ilu
        model_path = os.path.join(repo_root, 'models', 'affordance_second.py')
        spec = ilu.spec_from_file_location('affordance_second', model_path)
        assert spec and spec.loader
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return getattr(mod, 'AffordancePointNet2SSG')


@torch.no_grad()
def run_inference(model_path: str, points_path: str, center_path: str, output_dir: str,
                  device: str = 'auto', condition_mode: str = 'rd', use_condition: bool = True) -> str:
    """Run second-stage inference on a single preprocessed point cloud with a center.

    This follows the training/evaluation signature: model(pts, centers), where
    pts has shape (B,N,3) and centers has shape (B,3).
    """
    os.makedirs(output_dir, exist_ok=True)


    # Safety settings for older PyTorch builds on newer GPUs
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    target_device = set_device(device)

    points_np = np.load(points_path)
    if points_np.ndim != 2 or points_np.shape[1] != 3:
        raise ValueError(f"Expected points of shape (N,3), got {points_np.shape}")
    center_np = np.load(center_path)
    if center_np.shape != (3,):
        raise ValueError(f"Expected center of shape (3,), got {center_np.shape}")

    # torch expects (B,N,3) and (B,3)
    pts_t = torch.from_numpy(points_np[None, ...]).float().to(target_device)
    cen_t = torch.from_numpy(center_np[None, ...]).float().to(target_device)

    AffordancePointNet2SSG = robust_import_model()
    model = AffordancePointNet2SSG(use_condition=use_condition, condition_mode=condition_mode, use_xyz=True)
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(target_device)
    model.eval()

    # Preferred signature for second model: forward(pts, centers)
    logits = model(pts_t, cen_t)

    scores = torch.sigmoid(logits.squeeze(1))  # (1, N)
    scores_np = scores.squeeze(0).detach().cpu().numpy()

    out_path = os.path.join(output_dir, 'scores_second.npy')
    np.save(out_path, scores_np)

    manifest = {
        'points_path': os.path.abspath(points_path),
        'center_path': os.path.abspath(center_path),
        'model_path': os.path.abspath(model_path),
        'output_scores': os.path.abspath(out_path),
        'num_points': int(points_np.shape[0]),
        'device': str(target_device),
        'condition_mode': condition_mode,
        'use_condition': bool(use_condition),
    }
    with open(os.path.join(output_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    return out_path


def main():
    parser = argparse.ArgumentParser(description='Second-stage affordance inference on a single point cloud with a center')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--points_path', type=str, required=True)
    parser.add_argument('--center_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--condition_mode', type=str, default='rd', choices=['none','r','rd','kp'])
    parser.add_argument('--use_condition', action='store_true', default=True)
    args = parser.parse_args()

    run_inference(
        model_path=os.path.abspath(args.model_path),
        points_path=os.path.abspath(args.points_path),
        center_path=os.path.abspath(args.center_path),
        output_dir=os.path.abspath(args.output_dir),
        device=args.device,
        condition_mode=args.condition_mode,
        use_condition=bool(args.use_condition),
    )


if __name__ == '__main__':
    main()



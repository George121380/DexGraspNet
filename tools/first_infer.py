import os
import sys
import argparse
import json
from typing import Optional

import numpy as np
import torch
import time


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

    # Backend settings: allow cuDNN autotune when CUDA is available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    target_device = set_device(device)

    t0 = time.perf_counter()
    # Load points
    points_np = np.load(points_path)
    if points_np.ndim != 2 or points_np.shape[1] != 3:
        raise ValueError(f"Expected points of shape (N,3), got {points_np.shape}")
    # Optional downsampling (env/CLI controlled); default from CLI
    max_points_env = os.getenv('PN_MAX_POINTS', '')
    try:
        max_points_from_env = int(max_points_env) if len(max_points_env) > 0 else None
    except Exception:
        max_points_from_env = None
    max_points_cli = getattr(run_inference, '_max_points', None)
    max_points = max_points_cli if max_points_cli is not None else max_points_from_env
    if isinstance(max_points, int) and points_np.shape[0] > max_points:
        sel = np.random.choice(points_np.shape[0], size=max_points, replace=False)
        points_np = points_np[sel]
    points = torch.from_numpy(points_np[None, ...]).float().to(target_device)  # (1, N, 3)
    t1 = time.perf_counter()

    # Build model
    t_imp0 = time.perf_counter()
    AffordanceFirstPointNet2SSG = robust_import_model()
    t_imp1 = time.perf_counter()
    model = AffordanceFirstPointNet2SSG(use_xyz=True)
    t_load0 = time.perf_counter()
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(target_device)
    model.eval()
    t_load1 = time.perf_counter()

    # Forward
    t_fwd0 = time.perf_counter()
    logits = model(points)  # (1, 1, N)
    t_fwd1 = time.perf_counter()
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
        'timing_sec': {
            'load_points': t1 - t0,
            'import_model': t_imp1 - t_imp0,
            'load_ckpt_and_to_device': t_load1 - t_load0,
            'forward': t_fwd1 - t_fwd0,
            'total': t_fwd1 - t0,
        },
        'max_points': int(max_points) if isinstance(max_points, int) else None,
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
    parser.add_argument('--max_points', type=int, default=8192, help='If set, randomly downsample points to this count before inference')
    args = parser.parse_args()

    # Stash CLI max_points for use inside run_inference without changing signature
    setattr(run_inference, '_max_points', int(args.max_points) if args.max_points is not None else None)

    run_inference(
        model_path=os.path.abspath(args.model_path),
        points_path=os.path.abspath(args.points_path),
        output_dir=os.path.abspath(args.output_dir),
        device=args.device,
    )


if __name__ == '__main__':
    main()





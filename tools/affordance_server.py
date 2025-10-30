import os
import sys
import argparse
import json
import time
from typing import Optional

import numpy as np
import torch


def set_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def import_affordance_first():
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


def import_affordance_second():
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


class AffordanceServer:
    def __init__(self,
                 first_ckpt: str,
                 second_ckpt: Optional[str],
                 device: str = 'auto',
                 default_condition_mode: str = 'rd',
                 default_use_condition: bool = True) -> None:
        self.device = set_device(device)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # Load first-stage model
        AffordanceFirstPointNet2SSG = import_affordance_first()
        self.first_model = AffordanceFirstPointNet2SSG(use_xyz=True)
        ckpt_first = torch.load(os.path.abspath(first_ckpt), map_location='cpu')
        self.first_model.load_state_dict(ckpt_first['model'])
        self.first_model.to(self.device)
        self.first_model.eval()

        # Load second-stage model if provided
        self.second_model = None
        self.second_defaults = {
            'condition_mode': default_condition_mode,
            'use_condition': bool(default_use_condition),
        }
        if second_ckpt is not None and len(str(second_ckpt)) > 0 and os.path.isfile(os.path.abspath(second_ckpt)):
            AffordancePointNet2SSG = import_affordance_second()
            self.second_model = AffordancePointNet2SSG(
                use_condition=bool(default_use_condition),
                condition_mode=str(default_condition_mode),
                use_xyz=True,
            )
            ckpt_second = torch.load(os.path.abspath(second_ckpt), map_location='cpu')
            self.second_model.load_state_dict(ckpt_second['model'])
            self.second_model.to(self.device)
            self.second_model.eval()

    @torch.no_grad()
    def run_first(self, points_path: str, output_dir: str) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        t0 = time.perf_counter()
        pts = np.load(points_path)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Expected points of shape (N,3), got {pts.shape}")
        pts_t = torch.from_numpy(pts[None, ...]).float().to(self.device)
        t1 = time.perf_counter()
        logits = self.first_model(pts_t)
        t2 = time.perf_counter()
        scores = torch.sigmoid(logits.squeeze(1)).squeeze(0).detach().cpu().numpy()
        out_path = os.path.join(output_dir, 'scores_first.npy')
        np.save(out_path, scores)
        manifest = {
            'points_path': os.path.abspath(points_path),
            'output_scores': os.path.abspath(out_path),
            'num_points': int(pts.shape[0]),
            'device': str(self.device),
            'timing_sec': {
                'load_points': t1 - t0,
                'forward': t2 - t1,
                'total': t2 - t0,
            },
        }
        with open(os.path.join(output_dir, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)
        return {'scores_path': os.path.abspath(out_path), 'num_points': int(pts.shape[0])}

    @torch.no_grad()
    def run_second(self, points_path: str, center_path: str, output_dir: str,
                   condition_mode: Optional[str] = None, use_condition: Optional[bool] = None) -> dict:
        if self.second_model is None:
            raise RuntimeError('Second-stage model not loaded on server')
        os.makedirs(output_dir, exist_ok=True)
        # Enforce constructor-time defaults
        if condition_mode is not None and str(condition_mode) != str(self.second_defaults['condition_mode']):
            raise ValueError('condition_mode differs from server default; restart server with desired mode')
        if use_condition is not None and bool(use_condition) != bool(self.second_defaults['use_condition']):
            raise ValueError('use_condition differs from server default; restart server with desired setting')

        t0 = time.perf_counter()
        pts = np.load(points_path)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Expected points of shape (N,3), got {pts.shape}")
        cen = np.load(center_path)
        if cen.shape != (3,):
            raise ValueError(f"Expected center of shape (3,), got {cen.shape}")
        pts_t = torch.from_numpy(pts[None, ...]).float().to(self.device)
        cen_t = torch.from_numpy(cen[None, ...]).float().to(self.device)
        t1 = time.perf_counter()
        logits = self.second_model(pts_t, cen_t)
        t2 = time.perf_counter()
        scores = torch.sigmoid(logits.squeeze(1)).squeeze(0).detach().cpu().numpy()
        out_path = os.path.join(output_dir, 'scores_second.npy')
        np.save(out_path, scores)
        manifest = {
            'points_path': os.path.abspath(points_path),
            'center_path': os.path.abspath(center_path),
            'output_scores': os.path.abspath(out_path),
            'num_points': int(pts.shape[0]),
            'device': str(self.device),
            'condition_mode': str(self.second_defaults['condition_mode']),
            'use_condition': bool(self.second_defaults['use_condition']),
            'timing_sec': {
                'load_io': t1 - t0,
                'forward': t2 - t1,
                'total': t2 - t0,
            },
        }
        with open(os.path.join(output_dir, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)
        return {'scores_path': os.path.abspath(out_path), 'num_points': int(pts.shape[0])}


def main():
    parser = argparse.ArgumentParser(description='Persistent affordance inference server (first + optional second)')
    parser.add_argument('--first_ckpt', type=str, required=True)
    parser.add_argument('--second_ckpt', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--condition_mode', type=str, default='rd', choices=['none','r','rd','kp'])
    parser.add_argument('--use_condition', action='store_true', default=True)
    args = parser.parse_args()

    server = AffordanceServer(
        first_ckpt=os.path.abspath(args.first_ckpt),
        second_ckpt=(os.path.abspath(args.second_ckpt) if args.second_ckpt else None),
        device=args.device,
        default_condition_mode=args.condition_mode,
        default_use_condition=bool(args.use_condition),
    )

    # Signal readiness
    print('READY', flush=True)

    # Simple line-delimited JSON protocol
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception as e:
            print(json.dumps({'ok': False, 'error': f'Invalid JSON: {e}'}), flush=True)
            continue

        try:
            cmd = msg.get('type', '')
            if cmd == 'shutdown':
                print(json.dumps({'ok': True, 'status': 'bye'}), flush=True)
                break
            elif cmd == 'first':
                points_path = msg['points_path']
                output_dir = msg['output_dir']
                result = server.run_first(points_path, output_dir)
                print(json.dumps({'ok': True, **result}), flush=True)
            elif cmd == 'second':
                points_path = msg['points_path']
                center_path = msg['center_path']
                output_dir = msg['output_dir']
                cond_mode = msg.get('condition_mode')
                use_cond = msg.get('use_condition')
                result = server.run_second(points_path, center_path, output_dir, cond_mode, use_cond)
                print(json.dumps({'ok': True, **result}), flush=True)
            else:
                print(json.dumps({'ok': False, 'error': f'Unknown command: {cmd}'}), flush=True)
        except Exception as e:
            print(json.dumps({'ok': False, 'error': str(e)}), flush=True)


if __name__ == '__main__':
    main()



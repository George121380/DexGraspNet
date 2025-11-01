import argparse
import os
import sys
import types
import importlib.util

import numpy as np
import plotly.graph_objects as go
import torch
import transforms3d


TRANSLATION_NAMES = ['WRJTx', 'WRJTy', 'WRJTz']
ROTATION_NAMES = ['WRJRx', 'WRJRy', 'WRJRz']
JOINT_NAMES = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
]


def _load_entry(path: str) -> dict:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and len(arr) > 0:
        first = arr[0]
        if isinstance(first, dict):
            return first
        if hasattr(first, 'item'):
            maybe = first.item()
            if isinstance(maybe, dict):
                return maybe
        return first
    raise RuntimeError(f"Invalid entry file: {path}")


def _ensure_qpos_dict(data) -> dict:
    if isinstance(data, dict):
        return {k: float(v) for k, v in data.items()}
    raise RuntimeError("qpos data must be dict")


def _qpos_dict_to_pose(qpos: dict, device: str = 'cpu') -> torch.Tensor:
    trans = [float(qpos[name]) for name in TRANSLATION_NAMES]
    rot = transforms3d.euler.euler2mat(float(qpos['WRJRx']), float(qpos['WRJRy']), float(qpos['WRJRz']))
    rot6d = rot[:, :2].T.reshape(-1).tolist()
    joints = [float(qpos[name]) for name in JOINT_NAMES]
    data = trans + rot6d + joints
    return torch.tensor(data, dtype=torch.float, device=device).unsqueeze(0)


def _load_modules(vis_root: str) -> tuple:
    vis_root = os.path.abspath(vis_root)
    if vis_root not in sys.path:
        sys.path.insert(0, vis_root)

    def load_module(name: str, rel: str):
        full = os.path.join(vis_root, rel)
        spec = importlib.util.spec_from_file_location(name, full)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    utils_dir = os.path.join(vis_root, 'utils')
    pkg = types.ModuleType('utils')
    pkg.__path__ = [utils_dir]
    sys.modules['utils'] = pkg

    common_mod = load_module('utils.common', os.path.join('utils', 'common.py'))
    sys.modules['utils.common'] = common_mod
    hand_mod = load_module('dex_hand', os.path.join('utils', 'hand_model.py'))
    obj_mod = load_module('dex_obj', os.path.join('utils', 'object_model.py'))
    return hand_mod.HandModel, obj_mod.ObjectModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry', required=True, type=str)
    parser.add_argument('--object_code', required=True, type=str)
    parser.add_argument('--html', required=True, type=str)
    parser.add_argument('--vis_root', required=True, type=str)
    parser.add_argument('--mesh_root', required=True, type=str)
    parser.add_argument('--background', default='#E2F0D9', type=str)
    parser.add_argument('--baseline', default=None, type=str)
    parser.add_argument('--pk_root', default=None, type=str)
    parser.add_argument('--ref_scale_dir', default=None, type=str)
    args = parser.parse_args()

    if args.pk_root and args.pk_root not in sys.path:
        sys.path.insert(0, args.pk_root)

    try:
        import pytorch_kinematics  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("pytorch_kinematics not available") from exc

    entry = _load_entry(args.entry)
    baseline_entry = _load_entry(args.baseline) if args.baseline else None

    HandModel, ObjectModel = _load_modules(args.vis_root)
    cwd = os.getcwd()
    os.chdir(args.vis_root)
    try:
        device = 'cpu'
        left_hand = HandModel(
            mjcf_path=os.path.join('mjcf', 'left_shadow_hand.xml'),
            mesh_path=os.path.join('mjcf', 'meshes'),
            contact_points_path=os.path.join('mjcf', 'left_hand_contact_points.json'),
            penetration_points_path=os.path.join('mjcf', 'penetration_points.json'),
            device=device,
            handedness='left_hand'
        )
        right_hand = HandModel(
            mjcf_path=os.path.join('mjcf', 'right_shadow_hand.xml'),
            mesh_path=os.path.join('mjcf', 'meshes'),
            contact_points_path=os.path.join('mjcf', 'right_hand_contact_points.json'),
            penetration_points_path=os.path.join('mjcf', 'penetration_points.json'),
            device=device,
            handedness='right_hand'
        )

        object_model = ObjectModel(
            data_root_path=args.mesh_root,
            batch_size_each=1,
            num_samples=0,
            device=device,
            size='large'
        )
        object_model.initialize(args.object_code)
        scale = float(entry.get('scale', 1.0))
        object_model.object_scale_tensor = torch.tensor([[scale]], dtype=torch.float, device=device)

        left_pose = _qpos_dict_to_pose(entry['qpos_left'], device=device)
        right_pose = _qpos_dict_to_pose(entry['qpos_right'], device=device)
        left_hand.set_parameters(left_pose)
        right_hand.set_parameters(right_pose)

        traces = []
        traces.extend(right_hand.get_plotly_data(i=0, opacity=1.0, color='lightslategray', with_contact_points=False))
        traces.extend(object_model.get_plotly_data(i=0, color='seashell', opacity=1.0))
        traces.extend(left_hand.get_plotly_data(i=0, opacity=1.0, color='lightslategray', with_contact_points=False))

        if baseline_entry is not None:
            left_pose_st = _qpos_dict_to_pose(baseline_entry['qpos_left'], device=device)
            right_pose_st = _qpos_dict_to_pose(baseline_entry['qpos_right'], device=device)
            left_hand.set_parameters(left_pose_st)
            right_hand.set_parameters(right_pose_st)
            traces.extend(right_hand.get_plotly_data(i=0, opacity=0.35, color='orange', with_contact_points=False))
            traces.extend(left_hand.get_plotly_data(i=0, opacity=0.35, color='orange', with_contact_points=False))
            left_hand.set_parameters(left_pose)
            right_hand.set_parameters(right_pose)

        fig = go.Figure(traces)
        fig.update_layout(paper_bgcolor=args.background, plot_bgcolor=args.background)
        fig.update_layout(scene_aspectmode='data')
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            )
        )
        os.makedirs(os.path.dirname(args.html), exist_ok=True)
        fig.write_html(args.html)
    finally:
        os.chdir(cwd)


if __name__ == '__main__':
    main()


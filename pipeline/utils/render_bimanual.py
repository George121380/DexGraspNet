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
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--palm_offset', type=float, default=None)
    parser.add_argument('--palm_normal_len', type=float, default=None)
    parser.add_argument('--kpleft', type=str, default=None)
    parser.add_argument('--kpright', type=str, default=None)
    parser.add_argument('--points', type=str, default=None)
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

        hand_opacity = 0.35 if args.debug else 1.0
        obj_opacity = 0.25 if args.debug else 1.0
        traces = []
        traces.extend(right_hand.get_plotly_data(i=0, opacity=hand_opacity, color='lightslategray', with_contact_points=False))
        traces.extend(object_model.get_plotly_data(i=0, color='seashell', opacity=obj_opacity))
        traces.extend(left_hand.get_plotly_data(i=0, opacity=hand_opacity, color='lightslategray', with_contact_points=False))

        # Add palm center markers and normals for debugging (use kinematics of palm link)
        def _mat_np(mat):
            if hasattr(mat, 'detach'):
                mat = mat.detach().cpu().numpy()
            return mat[0] if getattr(mat, 'ndim', 2) == 3 else mat

        def _add_palm_marker_from_model(hand_model, color_pts: str, color_vec: str, name: str, normal_len: float = 0.06, center_offset: float = 0.02):
            # Global pose of the hand (world)
            g_t = hand_model.global_translation[0].detach().cpu().numpy().astype(np.float32)
            g_R = hand_model.global_rotation[0].detach().cpu().numpy().astype(np.float32)

            # Palm link vertices in local hand base -> world
            v_local = hand_model.mesh['robot0:palm']['vertices'].detach().cpu().numpy().astype(np.float32)
            # Transform to palm link frame, then to world
            v_palm = hand_model.current_status['robot0:palm'].transform_points(hand_model.mesh['robot0:palm']['vertices'])
            if len(v_palm.shape) == 3:
                v_palm = v_palm[0]
            v_world = (v_palm @ g_R.T + g_t).detach().cpu().numpy().astype(np.float32)

            # Geometric palm center (mesh centroid)
            p_centroid = v_world.mean(axis=0)

            # Wrist world pos for midline direction
            try:
                M_wrist_local = _mat_np(hand_model.current_status['robot0:wrist_child'].get_matrix())
                p_wrist_world = g_t + g_R @ M_wrist_local[:3, 3].astype(np.float32)
            except Exception:
                p_wrist_world = p_centroid
            dir_mid_world = p_centroid - p_wrist_world
            dir_mid_world = dir_mid_world / (np.linalg.norm(dir_mid_world) + 1e-8)

            # Shift slightly further towards palm center from wrist side
            p_center = p_centroid + center_offset * dir_mid_world

            # Palm normal by PCA (smallest eigenvector of covariance)
            vv = v_world - p_centroid
            C = vv.T @ vv
            eigvals, eigvecs = np.linalg.eigh(C)
            n_world = eigvecs[:, 0]  # smallest eigenvalue
            n_world = n_world / (np.linalg.norm(n_world) + 1e-8)
            # For right hand, flip to keep outward convention consistent with left
            if 'right' in name:
                n_world = -n_world
            # Ensure normal is perpendicular; choose sign that points roughly towards object side (use -dir_mid cross?)
            # Use heuristic: normal should be roughly orthogonal to dir_mid; keep current sign
            p2 = p_center + normal_len * n_world

            traces.append(go.Scatter3d(x=[p_center[0]], y=[p_center[1]], z=[p_center[2]], mode='markers',
                                       marker=dict(size=5, color=color_pts), name=f'{name}_palm'))
            traces.append(go.Scatter3d(x=[p_center[0], p2[0]], y=[p_center[1], p2[1]], z=[p_center[2], p2[2]], mode='lines',
                                       line=dict(color=color_vec, width=6), name=f'{name}_normal'))

        if args.debug:
            center_offset_cfg = float(args.palm_offset) if args.palm_offset is not None else 0.05
            normal_len_cfg = float(args.palm_normal_len) if args.palm_normal_len is not None else 0.06
            _add_palm_marker_from_model(left_hand, color_pts='green', color_vec='green', name='left', normal_len=normal_len_cfg, center_offset=center_offset_cfg)
            _add_palm_marker_from_model(right_hand, color_pts='blue', color_vec='blue', name='right', normal_len=normal_len_cfg, center_offset=center_offset_cfg)
            # Keypoint markers if provided
            try:
                if args.kpleft and os.path.exists(args.kpleft):
                    kpl = np.load(args.kpleft).astype(np.float32).reshape(3)
                    traces.append(go.Scatter3d(x=[kpl[0]], y=[kpl[1]], z=[kpl[2]], mode='markers',
                                               marker=dict(size=6, color='lime', symbol='diamond'), name='kp_left'))
                if args.kpright and os.path.exists(args.kpright):
                    kpr = np.load(args.kpright).astype(np.float32).reshape(3)
                    traces.append(go.Scatter3d(x=[kpr[0]], y=[kpr[1]], z=[kpr[2]], mode='markers',
                                               marker=dict(size=6, color='deepskyblue', symbol='diamond'), name='kp_right'))
                if args.points and os.path.exists(args.points):
                    pc = np.load(args.points).astype(np.float32)
                    ctd = pc.mean(axis=0)
                    traces.append(go.Scatter3d(x=[ctd[0]], y=[ctd[1]], z=[ctd[2]], mode='markers',
                                               marker=dict(size=7, color='magenta', symbol='x'), name='pc_centroid'))
                    # Draw target normal lines (kp -> centroid)
                    if 'kpl' in locals():
                        traces.append(go.Scatter3d(x=[kpl[0], ctd[0]], y=[kpl[1], ctd[1]], z=[kpl[2], ctd[2]],
                                                   mode='lines', line=dict(color='lime', width=4, dash='dash'), name='n_target_left'))
                    if 'kpr' in locals():
                        traces.append(go.Scatter3d(x=[kpr[0], ctd[0]], y=[kpr[1], ctd[1]], z=[kpr[2], ctd[2]],
                                                   mode='lines', line=dict(color='deepskyblue', width=4, dash='dash'), name='n_target_right'))
            except Exception:
                pass

        if baseline_entry is not None:
            left_pose_st = _qpos_dict_to_pose(baseline_entry['qpos_left'], device=device)
            right_pose_st = _qpos_dict_to_pose(baseline_entry['qpos_right'], device=device)
            left_hand.set_parameters(left_pose_st)
            right_hand.set_parameters(right_pose_st)
            traces.extend(right_hand.get_plotly_data(i=0, opacity=0.25 if args.debug else 0.35, color='orange', with_contact_points=False))
            traces.extend(left_hand.get_plotly_data(i=0, opacity=0.25 if args.debug else 0.35, color='orange', with_contact_points=False))
            if args.debug:
                center_offset_cfg = float(args.palm_offset) if args.palm_offset is not None else 0.05
                normal_len_cfg = float(args.palm_normal_len) if args.palm_normal_len is not None else 0.06
                _add_palm_marker_from_model(left_hand, color_pts='orange', color_vec='orange', name='left_st', normal_len=normal_len_cfg, center_offset=center_offset_cfg)
                _add_palm_marker_from_model(right_hand, color_pts='orange', color_vec='orange', name='right_st', normal_len=normal_len_cfg, center_offset=center_offset_cfg)
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


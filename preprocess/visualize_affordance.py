import os
import sys

# Allow importing hand_model and object_model from third_party/BimanGrasp-Dataset
THIRD_PARTY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'third_party', 'BimanGrasp-Dataset')
sys.path.append(THIRD_PARTY_DIR)

import argparse
import numpy as np
import torch
import transforms3d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import trimesh as tm

from hand_model import HandModel
from object_model import ObjectModel


translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
joint_names = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
]


def build_hand_pose_tensor(qpos_dict: dict, device: str) -> torch.Tensor:
    rot = np.array(transforms3d.euler.euler2mat(*[qpos_dict[name] for name in rot_names]))
    rot6 = rot[:, :2].T.ravel().tolist()
    hand_pose = torch.tensor(
        [qpos_dict[name] for name in translation_names] + rot6 + [qpos_dict[name] for name in joint_names],
        dtype=torch.float,
        device=device
    )
    return hand_pose


def ensure_k_points(points: torch.Tensor, k: int) -> torch.Tensor:
    n = points.shape[0]
    if n == 0:
        raise ValueError('Empty point cloud encountered when ensuring k points.')
    if n >= k:
        idx = torch.randperm(n, device=points.device)[:k]
        return points[idx]
    repeat = (k + n - 1) // n
    out = points.repeat(repeat, 1)[:k]
    return out


def furthest_point_sampling(points: torch.Tensor, m: int, start_index: int = 0) -> torch.Tensor:
    device = points.device
    num_points = points.shape[0]
    m = min(m, num_points)
    sampled_indices = torch.zeros(m, dtype=torch.long, device=device)
    distances = torch.full((num_points,), float('inf'), device=device, dtype=points.dtype)
    farthest = torch.tensor(start_index % num_points, device=device, dtype=torch.long)

    for i in range(m):
        sampled_indices[i] = farthest
        centroid = points[farthest].unsqueeze(0)
        dist = torch.sum((points - centroid) ** 2, dim=1)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances)
    return points[sampled_indices]


def sample_object_points_with_trimesh(mesh: tm.Trimesh, base_n: int, scale: float, x_shift: float, device: str) -> torch.Tensor:
    pts = mesh.sample(base_n)
    if scale is not None:
        pts = pts * float(scale)
    if x_shift != 0.0:
        pts[:, 0] += float(x_shift)
    return torch.tensor(pts, dtype=torch.float, device=device)


def compute_affordance(object_points: torch.Tensor, hand_points: torch.Tensor, dmax: float) -> torch.Tensor:
    dists = torch.cdist(object_points.unsqueeze(0), hand_points.unsqueeze(0), p=2)
    dmin = dists.min(dim=-1).values.squeeze(0)
    afford = 1.0 - torch.clamp(dmin / max(dmax, 1e-8), min=0.0, max=1.0)
    return afford


def make_side_by_side_figure(right_hand_plotly, left_hand_plotly, object_plotly, obj_points_np, afford_np):
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], column_widths=[0.5, 0.5])

    for trace in right_hand_plotly + object_plotly + left_hand_plotly:
        fig.add_trace(trace, row=1, col=1)

    scatter = go.Scatter3d(
        x=obj_points_np[:, 0], y=obj_points_np[:, 1], z=obj_points_np[:, 2],
        mode='markers',
        marker=dict(size=3, color=afford_np, colorscale='Viridis', cmin=0.0, cmax=1.0, colorbar=dict(title='afford'))
    )
    fig.add_trace(scatter, row=1, col=2)

    for c in [1, 2]:
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data', row=1, col=c)

    default_eye = dict(x=2.2, y=2.2, z=2.2)
    fig.update_layout(scene_camera=dict(eye=default_eye), scene2_camera=dict(eye=default_eye))
    fig.update_layout(paper_bgcolor='#E2F0D9', plot_bgcolor='#E2F0D9', margin=dict(l=0, r=0, t=30, b=0), title_text='Bimanual Grasp Affordance', title_x=0.5)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_name', type=str, required=True)
    parser.add_argument('--result_path', type=str, default=os.path.join(THIRD_PARTY_DIR, 'BimanGrasp-Dataset-Release-v1'))
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--k', type=int, default=1024)
    parser.add_argument('--dmax', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_html', type=str, default=None)
    parser.add_argument('--base_n', type=int, default=30000)
    parser.add_argument('--pre_rand_n', type=int, default=10000)
    parser.add_argument('--use_fps', action='store_true', default=False)
    parser.add_argument('--x_shift', type=float, default=0.0)
    args = parser.parse_args()

    device = args.device
    models_dir = os.path.join(THIRD_PARTY_DIR, 'models')
    meshes_dir = os.path.join(models_dir, 'meshes')
    left_mjcf = os.path.join(models_dir, 'left_shadow_hand_wrist_free.xml')
    right_mjcf = os.path.join(models_dir, 'right_shadow_hand_wrist_free.xml')
    left_contact_json = os.path.join(models_dir, 'left_hand_contact_points.json')
    right_contact_json = os.path.join(models_dir, 'right_hand_contact_points.json')
    penetration_json = os.path.join(models_dir, 'penetration_points.json')
    object_root_dir = os.path.join(THIRD_PARTY_DIR, 'Object-Release-v1')

    npy_path = os.path.join(args.result_path, args.object_name + '.npy')
    data = np.load(npy_path, allow_pickle=True)
    data_dict = data[args.num]

    right_qpos = data_dict['qpos_right']
    right_hand_pose = build_hand_pose_tensor(right_qpos, device)
    left_qpos = data_dict['qpos_left']
    left_hand_pose = build_hand_pose_tensor(left_qpos, device)

    left_hand_model = HandModel(
        mjcf_path=left_mjcf, mesh_path=meshes_dir,
        contact_points_path=left_contact_json, penetration_points_path=penetration_json,
        n_surface_points=4096, device=device, handedness='left_hand')

    right_hand_model = HandModel(
        mjcf_path=right_mjcf, mesh_path=meshes_dir,
        contact_points_path=right_contact_json, penetration_points_path=penetration_json,
        n_surface_points=4096, device=device, handedness='right_hand')

    object_model = ObjectModel(
        data_root_path=object_root_dir, batch_size_each=1, num_samples=args.k, device=device)

    right_hand_model.set_parameters(right_hand_pose.unsqueeze(0))
    left_hand_model.set_parameters(left_hand_pose.unsqueeze(0))

    object_model.initialize(args.object_name)
    object_model.object_scale_tensor = torch.tensor(data_dict['scale'], dtype=torch.float, device=device).reshape(1, 1)

    right_hand_plotly = right_hand_model.get_plotly_data(i=0, opacity=1, color='lightslategray', with_contact_points=False)
    left_hand_plotly = left_hand_model.get_plotly_data(i=0, opacity=1, color='powderblue', with_contact_points=False)
    object_plotly = object_model.get_plotly_data(i=0, color='seashell', opacity=1)

    right_pts = right_hand_model.get_surface_points()[0]
    left_pts = left_hand_model.get_surface_points()[0]
    hand_points = torch.cat([right_pts, left_pts], dim=0).to(dtype=torch.float, device=device)

    mesh = object_model.object_mesh_list[0]
    scale_tensor = object_model.object_scale_tensor[0, 0]
    scale_val = float(scale_tensor.item() if isinstance(scale_tensor, torch.Tensor) else scale_tensor)

    base_points = sample_object_points_with_trimesh(mesh, base_n=args.base_n, scale=scale_val, x_shift=args.x_shift, device=device)
    pre_points = base_points
    if args.use_fps and pre_points.shape[0] > args.pre_rand_n > 0:
        idx = torch.randperm(pre_points.shape[0], device=device)[:args.pre_rand_n]
        pre_points = pre_points[idx]

    if args.use_fps:
        obj_points = furthest_point_sampling(pre_points, args.k)
    else:
        obj_points = ensure_k_points(pre_points, args.k)

    afford = compute_affordance(obj_points, hand_points, dmax=args.dmax)
    obj_points_np = obj_points.detach().cpu().numpy()
    afford_np = afford.detach().cpu().numpy()

    fig = make_side_by_side_figure(right_hand_plotly, left_hand_plotly, object_plotly, obj_points_np, afford_np)
    if args.save_html is not None:
        os.makedirs(os.path.dirname(args.save_html), exist_ok=True)
        fig.write_html(args.save_html)
    fig.show()


if __name__ == '__main__':
    main()

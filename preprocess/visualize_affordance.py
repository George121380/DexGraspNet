import os
import sys

# Allow importing project-local packages and third_party modules
PROJ_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
THIRD_PARTY_DIR = os.path.join(PROJ_ROOT, 'third_party', 'BimanGrasp-Dataset')
if THIRD_PARTY_DIR not in sys.path:
    sys.path.insert(0, THIRD_PARTY_DIR)

import argparse
import numpy as np
import torch
import transforms3d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import trimesh as tm

from preprocess.utils import (
    get_grasp_center_point,
    build_hand_pose_tensor,
    sample_object_points_with_trimesh,
    load_models_and_data,
)


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


def make_three_panel_figure(right_hand_plotly, left_hand_plotly, object_plotly,
                            obj_points_np,
                            afford_right_np,
                            afford_left_np):
    fig = make_subplots(rows=1, cols=3,
                        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
                        column_widths=[0.34, 0.33, 0.33])

    # Left: original mesh + both hands
    for trace in right_hand_plotly + object_plotly + left_hand_plotly:
        fig.add_trace(trace, row=1, col=1)

    # Middle: right-hand affordance
    scatter_right = go.Scatter3d(
        x=obj_points_np[:, 0], y=obj_points_np[:, 1], z=obj_points_np[:, 2],
        mode='markers',
        marker=dict(size=2, color=afford_right_np, colorscale='Viridis', cmin=0.0, cmax=1.0,
                    colorbar=dict(title='afford_right'))
    )
    fig.add_trace(scatter_right, row=1, col=2)

    # Right: left-hand affordance
    scatter_left = go.Scatter3d(
        x=obj_points_np[:, 0], y=obj_points_np[:, 1], z=obj_points_np[:, 2],
        mode='markers',
        marker=dict(size=2, color=afford_left_np, colorscale='Viridis', cmin=0.0, cmax=1.0,
                    colorbar=dict(title='afford_left'))
    )
    fig.add_trace(scatter_left, row=1, col=3)

    for c in [1, 2, 3]:
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data', row=1, col=c)

    default_eye = dict(x=3.0, y=3.0, z=3.0)
    fig.update_layout(scene_camera=dict(eye=default_eye),
                      scene2_camera=dict(eye=default_eye),
                      scene3_camera=dict(eye=default_eye))
    fig.update_layout(paper_bgcolor='#E2F0D9', plot_bgcolor='#E2F0D9',
                      margin=dict(l=0, r=0, t=30, b=0),
                      title_text='Bimanual Grasp Affordance (Split by Hand)', title_x=0.5)
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
    obj_points, right_pts, left_pts, data_dict, right_hand_model, left_hand_model, object_model = load_models_and_data(
        object_name=args.object_name,
        result_path=args.result_path,
        device=device,
        base_n=args.base_n,
        pre_rand_n=args.pre_rand_n,
        use_fps=args.use_fps,
        k=args.k,
        x_shift=args.x_shift,
    )

    right_hand_plotly = right_hand_model.get_plotly_data(i=0, opacity=1, color='lightslategray', with_contact_points=False)
    left_hand_plotly = left_hand_model.get_plotly_data(i=0, opacity=1, color='powderblue', with_contact_points=False)
    object_plotly = object_model.get_plotly_data(i=0, color='seashell', opacity=1)

    right_pts = right_hand_model.get_surface_points()[0]
    left_pts = left_hand_model.get_surface_points()[0]

    # Separate affordances
    right_afford = compute_affordance(obj_points, right_pts.to(dtype=torch.float, device=device), dmax=args.dmax)
    left_afford = compute_affordance(obj_points, left_pts.to(dtype=torch.float, device=device), dmax=args.dmax)

    obj_points_np = obj_points.detach().cpu().numpy()
    afford_right_np = right_afford.detach().cpu().numpy()
    afford_left_np = left_afford.detach().cpu().numpy()

    # Compute representative grasp center points for right and left hands
    right_center = get_grasp_center_point(obj_points, right_pts.to(dtype=torch.float, device=device))
    left_center = get_grasp_center_point(obj_points, left_pts.to(dtype=torch.float, device=device))
    right_center_np = right_center.detach().cpu().numpy()
    left_center_np = left_center.detach().cpu().numpy()

    fig = make_three_panel_figure(right_hand_plotly, left_hand_plotly, object_plotly,
                                  obj_points_np, afford_right_np, afford_left_np)

    # Add center point markers to middle and right panels for inspection
    fig.add_trace(go.Scatter3d(x=[right_center_np[0]], y=[right_center_np[1]], z=[right_center_np[2]],
                               mode='markers', marker=dict(size=5, color='red'), name='right_center'),
                  row=1, col=2)
    fig.add_trace(go.Scatter3d(x=[left_center_np[0]], y=[left_center_np[1]], z=[left_center_np[2]],
                               mode='markers', marker=dict(size=5, color='red'), name='left_center'),
                  row=1, col=3)

    # Also visualize hands and object in the affordance panels for context
    # Re-generate traces to avoid reusing the same trace objects across subplots
    right_hand_plotly_c2 = right_hand_model.get_plotly_data(i=0, opacity=0.6, color='lightslategray', with_contact_points=False)
    object_plotly_c2 = object_model.get_plotly_data(i=0, color='seashell', opacity=0.3)
    for t in object_plotly_c2 + right_hand_plotly_c2:
        fig.add_trace(t, row=1, col=2)

    left_hand_plotly_c3 = left_hand_model.get_plotly_data(i=0, opacity=0.6, color='powderblue', with_contact_points=False)
    object_plotly_c3 = object_model.get_plotly_data(i=0, color='seashell', opacity=0.3)
    for t in object_plotly_c3 + left_hand_plotly_c3:
        fig.add_trace(t, row=1, col=3)
    if args.save_html is not None:
        os.makedirs(os.path.dirname(args.save_html), exist_ok=True)
        fig.write_html(args.save_html)
    fig.show()


if __name__ == '__main__':
    main()

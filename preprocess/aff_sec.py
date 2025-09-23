import os
import sys
import argparse
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Make local and third_party modules importable
CUR_DIR = os.path.dirname(__file__)
PROJ_ROOT = os.path.dirname(CUR_DIR)
THIRD_PARTY_DIR = os.path.join(PROJ_ROOT, 'third_party', 'BimanGrasp-Dataset')
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
if THIRD_PARTY_DIR not in sys.path:
    sys.path.insert(0, THIRD_PARTY_DIR)

from preprocess.utils import build_hand_pose_tensor, get_grasp_center_point
from hand_model import HandModel


def gaussian_field(points_xyz: torch.Tensor, centers_xyz: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Compute Gaussian field values on points given multiple centers.

    Args:
        points_xyz: (N, 3) float tensor, target (object) points
        centers_xyz: (M, 3) float tensor, left-hand keypoints as Gaussian centers
        sigma: standard deviation of Gaussian kernel

    Returns:
        values: (N,) tensor, aggregated Gaussian (take max over centers)
    """
    # (N, 1, 3) - (1, M, 3) -> (N, M, 3)
    diff = points_xyz.unsqueeze(1) - centers_xyz.unsqueeze(0)
    dist2 = (diff * diff).sum(dim=-1)  # (N, M)
    vals = torch.exp(-dist2 / (2.0 * (sigma ** 2)))  # (N, M)
    # Aggregate per point. Max tends to create a peaked field around nearest keypoint.
    values, _ = vals.max(dim=1)  # (N,)
    return values


def compute_hand_keypoints(hand_model: HandModel, pose_29: torch.Tensor, max_k: int = 64) -> torch.Tensor:
    """
    Get a set of hand keypoints (e.g., fingertips mesh vertices) in world coordinates.
    Strategy: Use all mesh vertices of selected distal links (fingertips) and downsample.

    Args:
        hand_model: instantiated HandModel (left or right)
        pose_29: (29,) tensor of hand pose

    Returns:
        keypoints: (K, 3) tensor
    """
    hand_model.set_parameters(pose_29.unsqueeze(0))
    # Collect fingertip meshes by name heuristic
    fingertip_names = [
        'robot0:FFDist', 'robot0:MFDist', 'robot0:RFDist', 'robot0:LFDist', 'robot0:THDist'
    ]
    verts_world = []
    for link_name, link_data in hand_model.mesh.items():
        if any(ft in link_name for ft in fingertip_names):
            v_local = link_data['vertices']  # (V,3) local
            v = hand_model.current_status[link_name].transform_points(v_local)
            if len(v.shape) == 3:
                v = v[0]
            v = v @ hand_model.global_rotation[0].T + hand_model.global_translation[0]
            verts_world.append(v)
    if len(verts_world) == 0:
        # Fallback: use all links' vertices
        for link_name, link_data in hand_model.mesh.items():
            v_local = link_data['vertices']
            v = hand_model.current_status[link_name].transform_points(v_local)
            if len(v.shape) == 3:
                v = v[0]
            v = v @ hand_model.global_rotation[0].T + hand_model.global_translation[0]
            verts_world.append(v)
    kp = torch.cat(verts_world, dim=0)  # (T, 3)
    # Downsample to a reasonable number via random sampling to avoid heavy computation
    K = min(max_k, kp.shape[0])
    idx = torch.randperm(kp.shape[0])[:K]
    return kp[idx]


def visualize(points_xyz: np.ndarray, aff_vals: np.ndarray, left_center_pts: np.ndarray, right_center_pts: np.ndarray, out_html: str):
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]])

    # Left: object cloud + left keypoints
    fig.add_trace(go.Scatter3d(x=points_xyz[:,0], y=points_xyz[:,1], z=points_xyz[:,2],
                               mode='markers', marker=dict(size=2, color='lightgray'), name='object'),
                  row=1, col=1)
    fig.add_trace(go.Scatter3d(x=left_center_pts[:,0], y=left_center_pts[:,1], z=left_center_pts[:,2],
                               mode='markers', marker=dict(size=4, color='blue'), name='left_kps'),
                  row=1, col=1)
    fig.add_trace(go.Scatter3d(x=right_center_pts[:,0], y=right_center_pts[:,1], z=right_center_pts[:,2],
                               mode='markers', marker=dict(size=4, color='red'), name='right_kps'),
                  row=1, col=1)

    # Right: affordance (Gaussian field around left keypoints)
    fig.add_trace(go.Scatter3d(x=points_xyz[:,0], y=points_xyz[:,1], z=points_xyz[:,2],
                               mode='markers',
                               marker=dict(size=2, color=aff_vals, colorscale='Viridis', showscale=True),
                               name='right_aff_gauss'),
                  row=1, col=2)
    fig.add_trace(go.Scatter3d(x=right_center_pts[:,0], y=right_center_pts[:,1], z=right_center_pts[:,2],
                               mode='markers', marker=dict(size=4, color='red'), name='right_kps'),
                  row=1, col=2)

    for c in [1, 2]:
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data', row=1, col=c)
    fig.update_layout(title='Affordance (right) conditioned on left keypoints (Gaussian field)', title_x=0.5)

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(out_html)
    fig.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='preprocess/results/<obj_name> directory')
    parser.add_argument('--num', type=int, default=0, help='pose index to build from grasp_pairs.npy')
    parser.add_argument('--sigma', type=float, default=0.02, help='Gaussian sigma in meters')
    parser.add_argument('--save_html', type=str, default=None, help='output html path')
    parser.add_argument('--right_kps_use_center_only', action='store_true', help='use only right grasp center as keypoint center(s)')
    parser.add_argument('--left_kps_max', type=int, default=64, help='max number of left hand keypoints')
    parser.add_argument('--right_kps_max', type=int, default=32, help='max number of right hand keypoints from mesh (if not center-only)')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    obj_pts_path = os.path.join(args.dir, 'obj_points.npy')
    pairs_db_path = os.path.join(args.dir, 'grasp_pairs.npy')
    if not os.path.exists(obj_pts_path):
        raise FileNotFoundError('Expected obj_points.npy in the directory.')
    if not os.path.exists(pairs_db_path):
        # legacy fallback
        legacy_path = os.path.join(args.dir, f'grasp_pairs_{args.num}.npy')
        if os.path.exists(legacy_path):
            pairs_db = np.load(legacy_path, allow_pickle=True).item()
            pairs_db = {
                'object_name': os.path.basename(args.dir.rstrip('/')),
                'pairs': {int(args.num): pairs_db}
            }
        else:
            # try legacy single-file name
            legacy2 = os.path.join(args.dir, 'grasp_pairs.npy')
            if os.path.exists(legacy2):
                pairs_db = np.load(legacy2, allow_pickle=True).item()
                if 'pairs' not in pairs_db:
                    pairs_db = {
                        'object_name': os.path.basename(args.dir.rstrip('/')),
                        'pairs': {int(args.num): pairs_db}
                    }
            else:
                raise FileNotFoundError('Expected grasp_pairs.npy or grasp_pairs_<num>.npy in the directory.')
    else:
        pairs_db = np.load(pairs_db_path, allow_pickle=True).item()

    obj_points = np.load(obj_pts_path)
    if str(args.num) not in pairs_db['pairs'] and int(args.num) not in pairs_db['pairs']:
        raise KeyError(f"Pose {args.num} not found in grasp_pairs.npy")
    # Normalize keys to int
    pair = pairs_db['pairs'][int(args.num)]

    # Build hand models and recover left-hand keypoints
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    models_dir = os.path.join(THIRD_PARTY_DIR, 'models')
    meshes_dir = os.path.join(models_dir, 'meshes')
    left_mjcf = os.path.join(models_dir, 'left_shadow_hand_wrist_free.xml')
    right_mjcf = os.path.join(models_dir, 'right_shadow_hand_wrist_free.xml')
    left_contact_json = os.path.join(models_dir, 'left_hand_contact_points.json')
    right_contact_json = os.path.join(models_dir, 'right_hand_contact_points.json')
    penetration_json = os.path.join(models_dir, 'penetration_points.json')

    left_hand_model = HandModel(
        mjcf_path=left_mjcf,
        mesh_path=meshes_dir,
        contact_points_path=left_contact_json,
        penetration_points_path=penetration_json,
        device=device,
        n_surface_points=4096,
        handedness='left_hand'
    )
    right_hand_model = HandModel(
        mjcf_path=right_mjcf,
        mesh_path=meshes_dir,
        contact_points_path=right_contact_json,
        penetration_points_path=penetration_json,
        device=device,
        n_surface_points=4096,
        handedness='right_hand'
    )

    left_pose = build_hand_pose_tensor(pair['left']['qpos'], device)
    right_pose = build_hand_pose_tensor(pair['right']['qpos'], device)
    # Ensure kinematics set for surface points and transforms
    left_hand_model.set_parameters(left_pose.unsqueeze(0))
    right_hand_model.set_parameters(right_pose.unsqueeze(0))

    # Compute left grasp center via get_grasp_center_point (reference method)
    pts_t = torch.from_numpy(obj_points).float()
    left_surface = left_hand_model.get_surface_points()[0]
    left_center_t = get_grasp_center_point(pts_t, left_surface)
    left_kps = left_center_t.view(1, 3)  # (1,3)

    # Right-hand keypoints (Gaussian centers): use right grasp center and/or distal vertices
    right_centers_list = []
    # Compute right grasp center via the same reference method
    right_surface = right_hand_model.get_surface_points()[0]
    right_center_t = get_grasp_center_point(pts_t, right_surface)
    right_centers_list.append(right_center_t.view(1, 3))
    if not args.right_kps_use_center_only:
        right_kps_mesh = compute_hand_keypoints(right_hand_model, right_pose, max_k=args.right_kps_max)
        right_centers_list.append(right_kps_mesh)
    right_centers = torch.cat(right_centers_list, dim=0) if len(right_centers_list) > 0 else compute_hand_keypoints(right_hand_model, right_pose, max_k=args.right_kps_max)

    # Compute Gaussian affordance for right hand around right-hand keypoints; left_kps are the condition carried in batch
    pts = torch.from_numpy(obj_points).to(device).float()
    aff = gaussian_field(pts, right_centers, sigma=args.sigma)  # (N,)
    aff_np = aff.detach().cpu().numpy()

    # Optional: normalize to [0,1]
    if aff_np.max() > 0:
        aff_np = aff_np / aff_np.max()

    # Visualize
    out_html = args.save_html if args.save_html is not None else os.path.join(args.dir, f'aff_sec_right_cond_left_{args.num}.html')
    visualize(obj_points, aff_np, left_kps.detach().cpu().numpy(), right_centers.detach().cpu().numpy(), out_html)


if __name__ == '__main__':
    main()



import torch
import os
import sys
import numpy as np
import transforms3d
import trimesh as tm

# third_party imports for models
PROJ_ROOT = os.path.dirname(os.path.dirname(__file__))
THIRD_PARTY_DIR = os.path.join(PROJ_ROOT, 'third_party', 'BimanGrasp-Dataset')
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)
if THIRD_PARTY_DIR not in sys.path:
    sys.path.append(THIRD_PARTY_DIR)

from hand_model import HandModel
from object_model import ObjectModel


def get_grasp_center_point(
    object_points: torch.Tensor,
    hand_surface_points: torch.Tensor,
    distance_threshold: float = None,
    use_weighted_centroid: bool = True,
) -> torch.Tensor:
    """
    Compute a representative grasp center point on the object point cloud for a given hand pose.

    Args:
        object_points: (N, 3) torch.float tensor on any device
        hand_surface_points: (M, 3) torch.float tensor on the same device
        distance_threshold: optional float; if None, it will be set to 0.02 * bbox_diag(object_points)
        use_weighted_centroid: if True, compute a distance-weighted centroid when neighbors exist

    Returns:
        center_point: (3,) torch.float tensor on the same device as inputs
    """
    if object_points.ndim != 2 or object_points.shape[1] != 3:
        raise ValueError("object_points must be (N,3)")
    if hand_surface_points.ndim != 2 or hand_surface_points.shape[1] != 3:
        raise ValueError("hand_surface_points must be (M,3)")

    # Set an adaptive distance threshold based on object size if not provided
    if distance_threshold is None:
        bbox_min = torch.min(object_points, dim=0).values
        bbox_max = torch.max(object_points, dim=0).values
        bbox_diag = torch.norm(bbox_max - bbox_min)
        distance_threshold = 0.02 * float(bbox_diag.item() if hasattr(bbox_diag, "item") else bbox_diag)
        # Fallback in degenerate cases
        if distance_threshold <= 0:
            distance_threshold = 1e-2

    # Compute nearest-hand distance for each object point
    # cdist: (1, N, M) → min over M → (N,)
    dists = torch.cdist(object_points.unsqueeze(0), hand_surface_points.unsqueeze(0), p=2)
    dmin = dists.min(dim=-1).values.squeeze(0)

    # Select neighbors within threshold
    neighbor_mask = dmin <= distance_threshold
    if torch.any(neighbor_mask):
        neighbors = object_points[neighbor_mask]
        if not use_weighted_centroid:
            center = neighbors.mean(dim=0)
        else:
            # Inverse-distance weighting with small epsilon to avoid inf
            eps = 1e-6
            w = 1.0 / (dmin[neighbor_mask] + eps)
            w = w / (w.sum() + eps)
            center = (neighbors * w.unsqueeze(-1)).sum(dim=0)
        return center

    # If no neighbors under threshold, pick the closest object point to the hand
    idx = torch.argmin(dmin)
    return object_points[idx]


def build_hand_pose_tensor(qpos_dict: dict, device: str) -> torch.Tensor:
    """
    Convert qpos dict to HandModel-compatible hand_pose tensor: [tx,ty,tz, rot6d(6), joint(20)].
    Rotation is built from transforms3d.euler -> R, then first two columns form 6D.
    """
    rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
    translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
    joint_names = [
        'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
        'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
        'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
        'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
        'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
    ]
    rot = np.array(transforms3d.euler.euler2mat(*[qpos_dict[name] for name in rot_names]))
    rot6 = rot[:, :2].T.ravel().tolist()
    hand_pose = torch.tensor(
        [qpos_dict[name] for name in translation_names] + rot6 + [qpos_dict[name] for name in joint_names],
        dtype=torch.float,
        device=device
    )
    return hand_pose


def sample_object_points_with_trimesh(mesh: tm.Trimesh, base_n: int, scale: float, x_shift: float, device: str) -> torch.Tensor:
    pts = mesh.sample(base_n)
    if scale is not None:
        pts = pts * float(scale)
    if x_shift != 0.0:
        pts[:, 0] += float(x_shift)
    return torch.tensor(pts, dtype=torch.float, device=device)


def load_models_and_data(object_name: str, result_path: str, device: str = 'cpu',
                         base_n: int = 30000, pre_rand_n: int = 10000, use_fps: bool = True,
                         k: int = 1024, x_shift: float = 0.0, num_idx: int = 0):
    """
    Load BimanGrasp object/hand models and one pose sample; return object points, left/right hand surface points.
    """
    models_dir = os.path.join(THIRD_PARTY_DIR, 'models')
    meshes_dir = os.path.join(models_dir, 'meshes')
    left_mjcf = os.path.join(models_dir, 'left_shadow_hand_wrist_free.xml')
    right_mjcf = os.path.join(models_dir, 'right_shadow_hand_wrist_free.xml')
    left_contact_json = os.path.join(models_dir, 'left_hand_contact_points.json')
    right_contact_json = os.path.join(models_dir, 'right_hand_contact_points.json')
    penetration_json = os.path.join(models_dir, 'penetration_points.json')
    object_root_dir = os.path.join(THIRD_PARTY_DIR, 'Object-Release-v1')

    npy_path = os.path.join(result_path, object_name + '.npy')
    data = np.load(npy_path, allow_pickle=True)
    data_dict = data[num_idx]

    right_qpos = data_dict['qpos_right']
    left_qpos = data_dict['qpos_left']
    right_hand_pose = build_hand_pose_tensor(right_qpos, device)
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
        data_root_path=object_root_dir, batch_size_each=1, num_samples=k, device=device)

    right_hand_model.set_parameters(right_hand_pose.unsqueeze(0))
    left_hand_model.set_parameters(left_hand_pose.unsqueeze(0))

    object_model.initialize(object_name)
    object_model.object_scale_tensor = torch.tensor(data_dict['scale'], dtype=torch.float, device=device).reshape(1, 1)

    mesh = object_model.object_mesh_list[0]
    scale_tensor = object_model.object_scale_tensor[0, 0]
    scale_val = float(scale_tensor.item() if isinstance(scale_tensor, torch.Tensor) else scale_tensor)
    base_points = sample_object_points_with_trimesh(mesh, base_n=base_n, scale=scale_val, x_shift=x_shift, device=device)
    pre_points = base_points
    if use_fps and pre_points.shape[0] > pre_rand_n > 0:
        idx = torch.randperm(pre_points.shape[0], device=device)[:pre_rand_n]
        pre_points = pre_points[idx]

    if use_fps:
        # simple pure torch FPS
        obj_points = _fps(pre_points, k)
    else:
        obj_points = _ensure_k(pre_points, k)

    right_pts = right_hand_model.get_surface_points()[0]
    left_pts = left_hand_model.get_surface_points()[0]
    return obj_points, right_pts, left_pts, data_dict, right_hand_model, left_hand_model, object_model


def _ensure_k(points: torch.Tensor, k: int) -> torch.Tensor:
    n = points.shape[0]
    if n >= k:
        idx = torch.randperm(n, device=points.device)[:k]
        return points[idx]
    repeat = (k + n - 1) // n
    return points.repeat(repeat, 1)[:k]


def _fps(points: torch.Tensor, m: int) -> torch.Tensor:
    device = points.device
    num_points = points.shape[0]
    m = min(m, num_points)
    sampled_indices = torch.zeros(m, dtype=torch.long, device=device)
    distances = torch.full((num_points,), float('inf'), device=device, dtype=points.dtype)
    farthest = torch.tensor(0, device=device, dtype=torch.long)
    for i in range(m):
        sampled_indices[i] = farthest
        centroid = points[farthest].unsqueeze(0)
        dist = torch.sum((points - centroid) ** 2, dim=1)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances)
    return points[sampled_indices]




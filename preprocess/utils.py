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


def sample_object_points_visible_with_trimesh(
    mesh: tm.Trimesh,
    target_n: int,
    scale: float,
    x_shift: float,
    device: str,
    camera_origin: np.ndarray = None,
    oversample_ratio: int = 10,
    normal_facing: bool = True,
    do_occlusion_test: bool = True,
) -> torch.Tensor:
    """
    Sample points only on camera-visible outer surface of the mesh.

    Strategy:
      1) Oversample surface points with face indices
      2) Filter by front-facing normals w.r.t camera
      3) Ray cast to keep only first-hit intersections (visible points)
      4) FPS/Repeat to ensure exactly target_n points
    """
    # Prepare a scaled mesh for consistency
    mesh_scaled = mesh.copy()
    if scale is not None:
        mesh_scaled.apply_scale(float(scale))
    if x_shift != 0.0:
        mesh_scaled.apply_translation([float(x_shift), 0.0, 0.0])

    # Derive a reasonable camera origin if not given
    if camera_origin is None:
        bounds = mesh_scaled.bounds  # (2,3)
        center = bounds.mean(axis=0)
        diag = np.linalg.norm(bounds[1] - bounds[0])
        if not np.isfinite(diag) or diag <= 0:
            diag = 1.0
        eye_dir = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        eye_dir = eye_dir / np.linalg.norm(eye_dir)
        camera_origin = center + eye_dir * (2.5 * diag)
    camera_origin = np.asarray(camera_origin, dtype=np.float64).reshape(1, 3)

    # Oversample
    count = max(target_n * max(1, oversample_ratio), target_n)
    # Use trimesh sampling with face indices
    cand_pts, cand_face_idx = tm.sample.sample_surface(mesh_scaled, count)
    cand_pts = cand_pts.astype(np.float64)
    cand_face_idx = np.asarray(cand_face_idx, dtype=np.int64)

    # Front-facing filter
    mask_visible = np.ones(len(cand_pts), dtype=bool)
    if normal_facing and mesh_scaled.face_normals is not None and len(mesh_scaled.face_normals) > 0:
        face_normals = mesh_scaled.face_normals[cand_face_idx]  # (M,3)
        view_vec = cand_pts - camera_origin  # broadcast -> (M,3)
        # Normalize view vectors
        view_dist = np.linalg.norm(view_vec, axis=1) + 1e-12
        view_dir = (view_vec.T / view_dist).T
        # Keep points where normal faces the camera: n · view_dir < 0
        mask_visible &= (np.einsum('ij,ij->i', face_normals, view_dir) < 0.0)

    # Occlusion test with ray casting (requires rtree or pyembree). If unavailable, skip.
    has_accel = False
    if do_occlusion_test:
        try:
            import rtree  # noqa: F401
            has_accel = True
        except Exception:
            try:
                from trimesh.ray import ray_pyembree  # noqa: F401
                has_accel = True
            except Exception:
                has_accel = False

    if do_occlusion_test and has_accel and mask_visible.any():
        sub_pts = cand_pts[mask_visible]
        view_vec = sub_pts - camera_origin  # (m,3)
        view_dist = np.linalg.norm(view_vec, axis=1)
        valid = view_dist > 1e-9
        sub_pts = sub_pts[valid]
        view_dist = view_dist[valid]
        directions = (view_vec[valid].T / view_dist).T  # normalized
        origins = np.repeat(camera_origin, repeats=sub_pts.shape[0], axis=0)

        # Ray query
        try:
            ray_dists = mesh_scaled.ray.intersects_first(origins, directions)
        except Exception:
            ray_dists = None

        if ray_dists is None:
            # If ray module failed, skip occlusion filtering
            final_mask = mask_visible
        else:
            # distances to sampled points along the same rays equal to view_dist
            # consider visible when first hit distance ~= view_dist
            atol = max(1e-4, 1e-3 * float(view_dist.max()))
            hit_ok = np.isfinite(ray_dists) & (np.abs(ray_dists - view_dist) <= atol)

            # Re-embed to full candidate mask
            mask_visible_idx = np.where(mask_visible)[0][valid]
            final_mask = np.zeros_like(mask_visible, dtype=bool)
            final_mask[mask_visible_idx[hit_ok]] = True
    else:
        final_mask = mask_visible

    visible_pts = cand_pts[final_mask]
    if visible_pts.shape[0] == 0:
        # Fallback: return standard samples to avoid empty clouds
        return sample_object_points_with_trimesh(mesh, base_n=target_n, scale=scale, x_shift=x_shift, device=device)

    # Downsample or pad to target_n
    pts_t = torch.tensor(visible_pts, dtype=torch.float, device=device)
    if pts_t.shape[0] >= target_n:
        # Use simple FPS for better coverage
        pts_t = _fps(pts_t, target_n)
    else:
        pts_t = _ensure_k(pts_t, target_n)
    return pts_t


def sample_object_points_outer_surface(
    mesh: tm.Trimesh,
    target_n: int,
    scale: float,
    x_shift: float,
    device: str,
    object_center: np.ndarray = None,
    oversample_ratio: int = 10,
) -> torch.Tensor:
    """
    Sample points from the outer surface only, removing interior-facing surfaces.

    Heuristic: keep samples whose face normal points outward relative to the
    vector from object center to the sample point, i.e., dot(n, p - center) > 0.
    This approximates the outer shell while allowing concavities.
    """
    mesh_scaled = mesh.copy()
    if scale is not None:
        mesh_scaled.apply_scale(float(scale))
    if x_shift != 0.0:
        mesh_scaled.apply_translation([float(x_shift), 0.0, 0.0])

    if object_center is None:
        bounds = mesh_scaled.bounds
        object_center = bounds.mean(axis=0)
    object_center = np.asarray(object_center, dtype=np.float64).reshape(1, 3)

    # Oversample surface points with face indices
    count = max(target_n * max(1, oversample_ratio), target_n)
    pts, face_idx = tm.sample.sample_surface(mesh_scaled, count)
    pts = pts.astype(np.float64)
    face_idx = np.asarray(face_idx, dtype=np.int64)

    # Outward filter
    face_normals = mesh_scaled.face_normals[face_idx]  # (M,3)
    outward_vec = pts - object_center  # (M,3)
    # Normalize outward_vec to avoid scaling effects
    ov_len = np.linalg.norm(outward_vec, axis=1) + 1e-12
    ov_dir = (outward_vec.T / ov_len).T
    keep = (np.einsum('ij,ij->i', face_normals, ov_dir) > 0.0)
    pts_filtered = pts[keep]

    if pts_filtered.shape[0] == 0:
        # Fallback to plain surface sampling
        return sample_object_points_with_trimesh(mesh, base_n=target_n, scale=scale, x_shift=x_shift, device=device)

    pts_t = torch.tensor(pts_filtered, dtype=torch.float, device=device)
    if pts_t.shape[0] >= target_n:
        pts_t = _fps(pts_t, target_n)
    else:
        pts_t = _ensure_k(pts_t, target_n)
    return pts_t


def sample_object_points_outer_surface_voxel(
    mesh: tm.Trimesh,
    target_n: int,
    scale: float,
    x_shift: float,
    device: str,
    resolution: int = 96,
    oversample_ratio: int = 6,
) -> torch.Tensor:
    """
    Approximate outer shell via voxelization (hollow boundary) and sample on it.
    This removes internal contacting faces between parts since only boundary voxels remain.
    """
    mesh_scaled = mesh.copy()
    if scale is not None:
        mesh_scaled.apply_scale(float(scale))
    if x_shift != 0.0:
        mesh_scaled.apply_translation([float(x_shift), 0.0, 0.0])

    bounds = mesh_scaled.bounds
    diag = np.linalg.norm(bounds[1] - bounds[0])
    if not np.isfinite(diag) or diag <= 0:
        diag = 1.0
    pitch = max(diag / float(max(8, resolution)), 1e-4)

    try:
        vox = mesh_scaled.voxelized(pitch)
        # Fill internal gaps/holes before extracting outer shell
        try:
            vox_filled = vox.fill(method='holes')
        except Exception:
            vox_filled = vox
        shell = vox_filled.hollow()
        shell_mesh = shell.as_boxes()
        if shell_mesh.is_empty:
            # Fallback to convex hull if hollow failed
            shell_mesh = mesh_scaled.convex_hull
        count = max(target_n * max(1, oversample_ratio), target_n)
        pts, _ = tm.sample.sample_surface(shell_mesh, count)
        pts_t = torch.tensor(pts, dtype=torch.float, device=device)
        if pts_t.shape[0] >= target_n:
            pts_t = _fps(pts_t, target_n)
        else:
            pts_t = _ensure_k(pts_t, target_n)
        return pts_t
    except Exception:
        # Any voxelization error -> return empty to trigger fallback in caller
        return torch.zeros((0, 3), dtype=torch.float, device=device)


def load_models_and_data(object_name: str, result_path: str, device: str = 'cpu',
                         base_n: int = 30000, pre_rand_n: int = 10000, use_fps: bool = True,
                         k: int = 1024, x_shift: float = 0.0, num_idx: int = 0,
                         visible_only: bool = False, camera_origin: np.ndarray = None,
                         outer_only: bool = True):
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
    if outer_only:
        # Try voxel hollow shell first
        obj_points = sample_object_points_outer_surface_voxel(
            mesh=mesh, target_n=k, scale=scale_val, x_shift=x_shift, device=device,
            resolution=96, oversample_ratio=max(2, base_n // max(1, k))
        )
        if obj_points.shape[0] == 0:
            # Fallback to normal-based outer surface heuristic
            obj_points = sample_object_points_outer_surface(
                mesh=mesh, target_n=k, scale=scale_val, x_shift=x_shift, device=device,
                object_center=None, oversample_ratio=max(2, base_n // max(1, k))
            )
    elif visible_only:
        # Directly sample visible-only points and ensure exactly k
        obj_points = sample_object_points_visible_with_trimesh(
            mesh=mesh, target_n=k, scale=scale_val, x_shift=x_shift, device=device,
            camera_origin=camera_origin, oversample_ratio=max(2, base_n // max(1, k))
        )
    else:
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




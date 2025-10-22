import os
import sys
import numpy as np
import torch
import transforms3d
import trimesh as tm

# Allow importing project-local packages
PROJ_ROOT = os.path.dirname(os.path.dirname(__file__))
THIRD_PARTY_DIR = os.path.join(PROJ_ROOT, 'third_party', 'BimanGrasp-Dataset')
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
if THIRD_PARTY_DIR not in sys.path:
    sys.path.insert(0, THIRD_PARTY_DIR)

from hand_model import HandModel


def ensure_object_symlink(object_name: str, sapien_mesh_root: str) -> str:
    """
    Ensure Object-Release-v1/<OBJ_NAME>/coacd symlink points to the SAPIEN preprocessed meshes.
    Returns the destination object directory under Object-Release-v1.
    """
    object_release_dir = os.path.join(THIRD_PARTY_DIR, 'Object-Release-v1')
    os.makedirs(object_release_dir, exist_ok=True)
    src = os.path.join(sapien_mesh_root, object_name, 'coacd')
    dst_dir = os.path.join(object_release_dir, object_name)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, 'coacd')
    if os.path.islink(dst) or os.path.exists(dst):
        try:
            if os.path.islink(dst):
                os.unlink(dst)
        except Exception:
            pass
    os.symlink(src, dst)
    return dst_dir


def gaussian_field(points_xyz: torch.Tensor, centers_xyz: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Compute Gaussian field values on points given multiple centers (max aggregation).
    """
    diff = points_xyz.unsqueeze(1) - centers_xyz.unsqueeze(0)  # (N, M, 3)
    dist2 = (diff * diff).sum(dim=-1)  # (N, M)
    vals = torch.exp(-dist2 / (2.0 * (sigma ** 2)))
    values, _ = vals.max(dim=1)
    return values


def build_hand_pose_tensor(qpos_dict: dict, device: str) -> torch.Tensor:
    """
    Convert qpos dict to hand_pose tensor: [tx,ty,tz, rot6d(6), joint(20)].
    Euler rotation is converted to matrix and then 6D (first two columns).
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
    R = np.array(transforms3d.euler.euler2mat(*[qpos_dict[name] for name in rot_names]))
    rot6 = R[:, :2].T.ravel().tolist()
    hand_pose = torch.tensor(
        [qpos_dict[name] for name in translation_names] + rot6 + [qpos_dict[name] for name in joint_names],
        dtype=torch.float,
        device=device
    )
    return hand_pose


def get_grasp_center_point(
    object_points: torch.Tensor,
    hand_surface_points: torch.Tensor,
    distance_threshold: float = None,
    use_weighted_centroid: bool = True,
) -> torch.Tensor:
    """
    Compute a representative grasp center point on the object point cloud for a given hand pose.
    """
    if object_points.ndim != 2 or object_points.shape[1] != 3:
        raise ValueError("object_points must be (N,3)")
    if hand_surface_points.ndim != 2 or hand_surface_points.shape[1] != 3:
        raise ValueError("hand_surface_points must be (M,3)")

    if distance_threshold is None:
        bbox_min = torch.min(object_points, dim=0).values
        bbox_max = torch.max(object_points, dim=0).values
        bbox_diag = torch.norm(bbox_max - bbox_min)
        distance_threshold = 0.02 * float(bbox_diag.item() if hasattr(bbox_diag, "item") else bbox_diag)
        if distance_threshold <= 0:
            distance_threshold = 1e-2

    dists = torch.cdist(object_points.unsqueeze(0), hand_surface_points.unsqueeze(0), p=2)
    dmin = dists.min(dim=-1).values.squeeze(0)

    neighbor_mask = dmin <= distance_threshold
    if torch.any(neighbor_mask):
        neighbors = object_points[neighbor_mask]
        if not use_weighted_centroid:
            center = neighbors.mean(dim=0)
        else:
            eps = 1e-6
            w = 1.0 / (dmin[neighbor_mask] + eps)
            w = w / (w.sum() + eps)
            center = (neighbors * w.unsqueeze(-1)).sum(dim=0)
        return center

    idx = torch.argmin(dmin)
    return object_points[idx]


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


def sample_object_points_with_trimesh(mesh: tm.Trimesh, base_n: int, scale: float, x_shift: float, device: str) -> torch.Tensor:
    # Deprecated: no longer used by the unified visibility sampling pipeline.
    pts = mesh.sample(base_n)
    if scale is not None:
        pts = pts * float(scale)
    if x_shift != 0.0:
        pts[:, 0] += float(x_shift)
    return torch.tensor(pts, dtype=torch.float, device=device)


def _grid_from_bounds(center: np.ndarray, corners: np.ndarray, right: np.ndarray, up: np.ndarray, per_view_samples: int):
    """
    Build a regular 2D grid on the image plane that covers the mesh AABB projection.
    """
    rel = corners - center[None, :]
    pr = rel @ right  # (8,)
    pu = rel @ up     # (8,)
    rmin, rmax = pr.min(), pr.max()
    umin, umax = pu.min(), pu.max()
    g = max(1, int(np.ceil(np.sqrt(max(1, per_view_samples)))))
    rs = np.linspace(rmin, rmax, g)
    us = np.linspace(umin, umax, g)
    R, U = np.meshgrid(rs, us, indexing='xy')
    return R.reshape(-1), U.reshape(-1)


def sample_object_points_visible_ortho(
    mesh: tm.Trimesh,
    target_n: int,
    scale: float,
    x_shift: float,
    device: str,
    base_n: int = 30000,
    n_views: int = 26,
    margin_ratio: float = 0.05,
    min_views: int = 2,
) -> torch.Tensor:
    """
    Sample camera-visible surface points using multiple orthographic views (no perspective).
    - Place orthographic cameras along axis-aligned directions (±X, ±Y, ±Z) outside the mesh AABB.
    - Cast parallel rays from a regular image-plane grid; keep the first intersections only per ray.
    - Combine all hits and downsample to target_n via FPS.
    This mimics reconstructing a point cloud from depth maps captured by orthographic cameras,
    which avoids sampling internal/occluded surfaces.
    """
    mesh_scaled = mesh.copy()
    if scale is not None:
        mesh_scaled.apply_scale(float(scale))
    if x_shift != 0.0:
        mesh_scaled.apply_translation([float(x_shift), 0.0, 0.0])

    bounds = mesh_scaled.bounds
    center = bounds.mean(axis=0)
    diag = np.linalg.norm(bounds[1] - bounds[0])
    if not np.isfinite(diag) or diag <= 0:
        diag = 1.0
    cam_offset = 0.6 * diag * (1.0 + margin_ratio)

    # Build view directions: 6 faces, 12 edges, 8 corners of a cube (total 26)
    dirs = []
    axes = [np.array([1,0,0]), np.array([-1,0,0]), np.array([0,1,0]), np.array([0,-1,0]), np.array([0,0,1]), np.array([0,0,-1])]
    edges = [
        np.array([1,1,0]), np.array([1,-1,0]), np.array([-1,1,0]), np.array([-1,-1,0]),
        np.array([1,0,1]), np.array([1,0,-1]), np.array([-1,0,1]), np.array([-1,0,-1]),
        np.array([0,1,1]), np.array([0,1,-1]), np.array([0,-1,1]), np.array([0,-1,-1])
    ]
    corners = [
        np.array([1,1,1]), np.array([1,1,-1]), np.array([1,-1,1]), np.array([1,-1,-1]),
        np.array([-1,1,1]), np.array([-1,1,-1]), np.array([-1,-1,1]), np.array([-1,-1,-1])
    ]
    for v in axes + edges + corners:
        v = v.astype(float)
        n = np.linalg.norm(v)
        if n < 1e-8:
            continue
        dirs.append(v / n)
    # build views (vn, up_hint)
    views = []
    for vn in dirs:
        up_hint = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(up_hint, vn)) > 0.9:
            up_hint = np.array([1.0, 0.0, 0.0])
        views.append((vn, up_hint))
    if n_views and n_views < len(views):
        views = views[:n_views]
    per_view = max(1, int(np.ceil(float(base_n) / float(len(views)))))

    # Try to use pyembree intersector; fall back to pure triangle intersector
    intersector = None
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector as RMI
        intersector = RMI(mesh_scaled)
    except Exception:
        try:
            from trimesh.ray.ray_triangle import RayMeshIntersector as RMI
            intersector = RMI(mesh_scaled)
        except Exception:
            intersector = None

    corners = np.array([
        [bounds[0,0], bounds[0,1], bounds[0,2]],
        [bounds[0,0], bounds[0,1], bounds[1,2]],
        [bounds[0,0], bounds[1,1], bounds[0,2]],
        [bounds[0,0], bounds[1,1], bounds[1,2]],
        [bounds[1,0], bounds[0,1], bounds[0,2]],
        [bounds[1,0], bounds[0,1], bounds[1,2]],
        [bounds[1,0], bounds[1,1], bounds[0,2]],
        [bounds[1,0], bounds[1,1], bounds[1,2]],
    ], dtype=float)

    hits_loc_tid = []  # list of (loc, tri_id)
    tri_vis_count = {}
    for vn, up_hint in views:
        vn = vn / (np.linalg.norm(vn) + 1e-12)
        right = np.cross(up_hint, vn)
        rl = np.linalg.norm(right)
        if rl < 1e-8:
            alt = np.array([1.0, 0.0, 0.0]) if abs(vn[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            right = np.cross(alt, vn)
            rl = np.linalg.norm(right)
        right = right / (rl + 1e-12)
        up = np.cross(vn, right)
        up = up / (np.linalg.norm(up) + 1e-12)

        plane_center = center + vn * cam_offset
        R, U = _grid_from_bounds(center, corners, right, up, per_view)
        origins = plane_center[None, :] + R[:, None] * right[None, :] + U[:, None] * up[None, :]
        directions = np.repeat((-vn)[None, :], origins.shape[0], axis=0)

        if intersector is None:
            continue
        try:
            # get all intersections with triangle ids (ensure correct kwarg names)
            try:
                locations, index_ray, index_tri = intersector.intersects_location(
                    ray_origins=origins, ray_directions=directions, multiple_hits=True
                )
            except TypeError:
                locations, index_ray, index_tri = intersector.intersects_location(
                    ray_origins=origins, ray_directions=directions
                )
            if locations is None or len(locations) == 0:
                continue
            # Front-face culling: keep intersections where face normal faces camera
            fns = mesh_scaled.face_normals[np.asanyarray(index_tri, dtype=int)]
            # Use per-ray direction (all equal to directions[0]) for clarity; front-face means N·D < 0
            ray_dir = directions[0].reshape(1, 3)
            facing = (np.einsum('ij,ij->i', fns, np.repeat(ray_dir, fns.shape[0], axis=0)) < 0.0)

            # keep first front-facing hit per ray
            ray_to_first = {}
            deltas = locations - origins[index_ray]
            dists = np.einsum('ij,ij->i', deltas, deltas)
            for loc, ridx, tri_id, dsq, face_ok in zip(locations, index_ray, index_tri, dists, facing):
                if not face_ok:
                    continue
                prev = ray_to_first.get(ridx)
                if (prev is None) or (dsq < prev[1]):
                    ray_to_first[ridx] = (loc, dsq, int(tri_id))
            if ray_to_first:
                for loc, _, tid in ray_to_first.values():
                    hits_loc_tid.append((loc, tid))
                    tri_vis_count[tid] = tri_vis_count.get(tid, 0) + 1
        except Exception:
            continue

    if len(hits_loc_tid) == 0:
        # fallback: return empty tensor to indicate failure
        return torch.zeros((0, 3), dtype=torch.float, device=device)

    keep_tris = {tid for tid, cnt in tri_vis_count.items() if cnt >= max(1, int(min_views))}
    if not keep_tris:
        # if nothing passes threshold, lower to 1
        keep_tris = set(tri_vis_count.keys())
    pts = np.array([loc for (loc, tid) in hits_loc_tid if tid in keep_tris], dtype=float)
    if pts.shape[0] == 0:
        # ultimate fallback: convex hull surface
        hull = mesh_scaled.convex_hull
        pts, _ = tm.sample.sample_surface(hull, max(target_n, 1024))
    pts_t = torch.tensor(pts, dtype=torch.float, device=device)
    if pts_t.shape[0] >= target_n:
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
    oversample_ratio: int = 10,
) -> torch.Tensor:
    # Deprecated: kept for backward compatibility; not used by the current pipeline.
    mesh_scaled = mesh.copy()
    if scale is not None:
        mesh_scaled.apply_scale(float(scale))
    if x_shift != 0.0:
        mesh_scaled.apply_translation([float(x_shift), 0.0, 0.0])
    bounds = mesh_scaled.bounds
    center = bounds.mean(axis=0)
    count = max(target_n * max(1, oversample_ratio), target_n)
    pts, face_idx = tm.sample.sample_surface(mesh_scaled, count)
    face_normals = mesh_scaled.face_normals[face_idx]
    outward_vec = pts - center
    ov_len = np.linalg.norm(outward_vec, axis=1) + 1e-12
    ov_dir = (outward_vec.T / ov_len).T
    keep = (np.einsum('ij,ij->i', face_normals, ov_dir) > 0.0)
    pts_filtered = pts[keep]
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
        try:
            vox_filled = vox.fill(method='holes')
        except Exception:
            vox_filled = vox
        shell = vox_filled.hollow()
        shell_mesh = shell.as_boxes()
        if shell_mesh.is_empty:
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
        return torch.zeros((0, 3), dtype=torch.float, device=device)


def load_object_points_and_models(
    object_name: str,
    result_path: str,
    sapien_mesh_root: str,
    device: str,
    k: int,
    base_n: int,
    x_shift: float,
    outer_only: bool = True,
):
    """
    Load object mesh from SAPIEN preprocessed meshes, sample object points, and build hand models.
    Scaling is read from the converted per-object npy (pose 0).
    """
    # Read scale from converted data
    npy_path = os.path.join(result_path, object_name + '.npy')
    data_all = np.load(npy_path, allow_pickle=True)
    data_dict = data_all[0]
    scale_val = float(data_dict.get('scale', 1.0))

    # Load mesh
    mesh_path = os.path.join(sapien_mesh_root, object_name, 'coacd', 'decomposed.obj')
    mesh = tm.load(mesh_path, force='mesh')
    # New unified sampling strategy: multi-view orthographic visibility sampling
    pts = sample_object_points_visible_ortho(
        mesh=mesh,
        target_n=k,
        scale=scale_val,
        x_shift=x_shift,
        device=device,
        base_n=base_n,
        n_views=6,
        margin_ratio=0.05,
    )
    if pts.shape[0] == 0:
        # ultimate fallback to previous outer surface voxel shell to avoid empty result
        pts = sample_object_points_outer_surface_voxel(
            mesh=mesh, target_n=k, scale=scale_val, x_shift=x_shift, device=device,
            resolution=96, oversample_ratio=max(2, base_n // max(1, k))
        )

    left_model, right_model = make_hand_models(device)
    return pts, left_model, right_model

def compute_hand_keypoints(
    hand_model: HandModel,
    pose_29: torch.Tensor,
    object_points: torch.Tensor = None,
    max_k: int = 64,
    distance_ratio: float = 0.02,
) -> torch.Tensor:
    # Deprecated: mesh keypoints are no longer used by the current pipeline.
    hand_model.set_parameters(pose_29.unsqueeze(0))
    return torch.zeros((0, 3), dtype=torch.float, device=hand_model.device if hasattr(hand_model, 'device') else 'cpu')


def make_hand_models(device: str):
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
    return left_hand_model, right_hand_model



import os
import sys
import argparse
import numpy as np
import torch

# Make local and third_party modules importable
CUR_DIR = os.path.dirname(__file__)
PROJ_ROOT = os.path.dirname(CUR_DIR)
THIRD_PARTY_DIR = os.path.join(PROJ_ROOT, 'third_party', 'BimanGrasp-Dataset')
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
if THIRD_PARTY_DIR not in sys.path:
    sys.path.insert(0, THIRD_PARTY_DIR)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # Only for type hints; avoid importing heavy deps at runtime
    from hand_model import HandModel  # noqa: F401


def gaussian_field(points_xyz: torch.Tensor, centers_xyz: torch.Tensor, sigma: float) -> torch.Tensor:
    """Compute Gaussian field values on object points given centers."""
    if points_xyz.ndim != 2 or points_xyz.shape[-1] != 3:
        raise ValueError("points_xyz must be (N, 3)")
    if centers_xyz.ndim != 2 or centers_xyz.shape[-1] != 3:
        raise ValueError("centers_xyz must be (M, 3)")

    points = points_xyz.unsqueeze(1)  # (N, 1, 3)
    centers = centers_xyz.unsqueeze(0)  # (1, M, 3)
    dist2 = ((points - centers) ** 2).sum(dim=-1)  # (N, M)
    vals = torch.exp(-dist2 / (2.0 * sigma ** 2))
    values, _ = vals.max(dim=1)  # (N,)
    return values


def compute_hand_keypoints(hand_model, pose_29: torch.Tensor, max_k: int = 64) -> torch.Tensor:
    """
    Sample a set of fingertip-related mesh vertices as keypoints in world coordinates and downsample.
    """
    hand_model.set_parameters(pose_29.unsqueeze(0))
    fingertip_names = [
        'robot0:FFDist', 'robot0:MFDist', 'robot0:RFDist', 'robot0:LFDist', 'robot0:THDist'
    ]
    verts_world = []
    for link_name, link_data in hand_model.mesh.items():
        if any(ft in link_name for ft in fingertip_names):
            v_local = link_data['vertices']
            v = hand_model.current_status[link_name].transform_points(v_local)
            if len(v.shape) == 3:
                v = v[0]
            v = v @ hand_model.global_rotation[0].T + hand_model.global_translation[0]
            verts_world.append(v)
    if len(verts_world) == 0:
        # Fallback to all links' vertices if fingertip heuristic fails
        for link_name, link_data in hand_model.mesh.items():
            v_local = link_data['vertices']
            v = hand_model.current_status[link_name].transform_points(v_local)
            if len(v.shape) == 3:
                v = v[0]
            v = v @ hand_model.global_rotation[0].T + hand_model.global_translation[0]
            verts_world.append(v)
    kp = torch.cat(verts_world, dim=0)
    K = min(max_k, kp.shape[0])
    idx = torch.randperm(kp.shape[0])[:K]
    return kp[idx]


def build_hand_pose_tensor(qpos_dict: dict, device: str) -> torch.Tensor:
    """
    Convert qpos dict to hand_pose tensor: [tx,ty,tz, rot6d(6), joint(20)].
    Lazily import transforms3d to avoid heavy imports on default path.
    """
    import transforms3d
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


def get_grasp_center_point(
    object_points: torch.Tensor,
    hand_surface_points: torch.Tensor,
    distance_threshold: float = None,
    use_weighted_centroid: bool = True,
) -> torch.Tensor:
    """
    Compute a representative grasp center point on object point cloud for a given hand surface.
    Self-contained to avoid importing preprocess.utils (which pulls heavy deps).
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


def load_pairs_db(obj_dir: str, num_hint: int = 0):
    """
    Load aggregated pairs db if available; support legacy single-pose files.
    """
    obj_pts_path = os.path.join(obj_dir, 'obj_points.npy')
    pairs_db_path = os.path.join(obj_dir, 'grasp_pairs.npy')
    if not os.path.exists(obj_pts_path):
        raise FileNotFoundError(f'Missing obj_points.npy in {obj_dir}')
    if not os.path.exists(pairs_db_path):
        # legacy fallbacks
        legacy_path = os.path.join(obj_dir, f'grasp_pairs_{num_hint}.npy')
        if os.path.exists(legacy_path):
            pairs_db = np.load(legacy_path, allow_pickle=True).item()
            pairs_db = {
                'object_name': os.path.basename(obj_dir.rstrip('/')),
                'pairs': {int(num_hint): pairs_db}
            }
        else:
            legacy2 = os.path.join(obj_dir, 'grasp_pairs.npy')
            if os.path.exists(legacy2):
                pairs_db = np.load(legacy2, allow_pickle=True).item()
                if 'pairs' not in pairs_db:
                    pairs_db = {
                        'object_name': os.path.basename(obj_dir.rstrip('/')),
                        'pairs': {int(num_hint): pairs_db}
                    }
            else:
                raise FileNotFoundError(f'No grasp_pairs found in {obj_dir}')
    else:
        pairs_db = np.load(pairs_db_path, allow_pickle=True).item()
    obj_points = np.load(obj_pts_path)
    return obj_points, pairs_db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='preprocess/results/<obj_name> directory')
    parser.add_argument('--hand', type=str, default='left', choices=['left', 'right'], help='which hand is the first hand')
    parser.add_argument('--save', type=str, default=None, help='output npy path for first-hand pairs')
    parser.add_argument('--include_kps_from_mesh', action='store_true', help='include mesh keypoints for the first hand')
    parser.add_argument('--kps_max', type=int, default=64, help='max number of mesh keypoints to sample when enabled')
    parser.add_argument('--recompute_center', action='store_true', help='recompute center from hand surface instead of using stored center_point')
    parser.add_argument('--device', type=str, default='auto', help="'cpu', 'cuda', or 'auto'")
    parser.add_argument('--sigma', type=float, default=0.02, help='Gaussian sigma (meters) for affordance field')
    parser.add_argument('--normalize_aff', action='store_true', default=True, help='normalize affordance to [0,1] (default: enabled)')
    args = parser.parse_args()

    # Resolve device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    obj_points, pairs_db = load_pairs_db(args.dir)
    pts_cpu = torch.from_numpy(obj_points.astype(np.float32))

    # Prepare optional hand model if recompute_center or include_kps_from_mesh
    need_models = args.recompute_center or args.include_kps_from_mesh
    if need_models:
        models_dir = os.path.join(THIRD_PARTY_DIR, 'models')
        meshes_dir = os.path.join(models_dir, 'meshes')
        if args.hand == 'left':
            mjcf_path = os.path.join(models_dir, 'left_shadow_hand_wrist_free.xml')
            contact_json = os.path.join(models_dir, 'left_hand_contact_points.json')
            handedness = 'left_hand'
        else:
            mjcf_path = os.path.join(models_dir, 'right_shadow_hand_wrist_free.xml')
            contact_json = os.path.join(models_dir, 'right_hand_contact_points.json')
            handedness = 'right_hand'
        penetration_json = os.path.join(models_dir, 'penetration_points.json')
        from hand_model import HandModel  # noqa: F811 (import only when needed)

        hand_model = HandModel(
            mjcf_path=mjcf_path,
            mesh_path=meshes_dir,
            contact_points_path=contact_json,
            penetration_points_path=penetration_json,
            device=device,
            n_surface_points=4096,
            handedness=handedness,
        )
        pts_t = pts_cpu.to(device)
    else:
        hand_model = None
        pts_t = None

    # Iterate all poses and extract first-hand information
    pairs = pairs_db.get('pairs', {})
    if not isinstance(pairs, dict) or len(pairs) == 0:
        raise ValueError('No pairs found in grasp_pairs.npy')

    out_pairs = {}
    pose_indices = sorted(list(map(int, pairs.keys())))
    for idx in pose_indices:
        pair = pairs[idx]
        hand_entry = pair[args.hand]
        qpos = hand_entry['qpos']

        # Center point: either reuse saved or recompute
        if args.recompute_center and hand_model is not None:
            pose = build_hand_pose_tensor(qpos, device)
            hand_model.set_parameters(pose.unsqueeze(0))
            hand_surface = hand_model.get_surface_points()[0]
            center_t = get_grasp_center_point(pts_t, hand_surface)
            center_np = center_t.detach().cpu().numpy().astype(np.float32)
        else:
            center_np = np.asarray(hand_entry['center_point'], dtype=np.float32)

        # Optional mesh keypoints
        if args.include_kps_from_mesh and hand_model is not None:
            pose = build_hand_pose_tensor(qpos, device)
            kps_t = compute_hand_keypoints(hand_model, pose, max_k=args.kps_max)
            kps_np = kps_t.detach().cpu().numpy().astype(np.float32)
            kps_np = np.concatenate([center_np.reshape(1, 3), kps_np], axis=0)
        else:
            kps_np = center_np.reshape(1, 3).astype(np.float32)

        # Compute Gaussian affordance field around first-hand keypoints
        aff = gaussian_field(pts_cpu, torch.from_numpy(kps_np).float(), sigma=float(args.sigma))
        aff_np = aff.numpy().astype(np.float32)
        if args.normalize_aff and aff_np.max() > 0:
            aff_np = aff_np / aff_np.max()

        out_pairs[idx] = {
            'qpos': qpos,
            'center_point': center_np,
            'first_kps': kps_np,  # 1x3 if center-only, Kx3 if mesh sampling enabled
            f'aff_scores_{args.hand}': aff_np,
        }

    # Output
    obj_name = pairs_db.get('object_name', os.path.basename(args.dir.rstrip('/')))
    save_path = args.save if args.save is not None else os.path.join(args.dir, f'aff_first_pairs_{args.hand}.npy')
    out_obj = {
        'object_name': obj_name,
        'points': obj_points.astype(np.float32),
        'hand': args.hand,
        'pairs': out_pairs,
        'meta': {
            'include_kps_from_mesh': bool(args.include_kps_from_mesh),
            'kps_max': int(args.kps_max),
            'recompute_center': bool(args.recompute_center),
            'sigma': float(args.sigma),
            'normalize_aff': bool(args.normalize_aff),
        }
    }
    np.save(save_path, out_obj, allow_pickle=True)
    print(f'Saved first-hand pairs to: {save_path}')


if __name__ == '__main__':
    main()

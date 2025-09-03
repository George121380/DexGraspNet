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

from preprocess.utils import build_hand_pose_tensor, get_grasp_center_point
from hand_model import HandModel


def gaussian_field(points_xyz: torch.Tensor, centers_xyz: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Compute Gaussian field values on points given multiple centers (right-hand keypoints).
    Args:
        points_xyz: (N, 3) float tensor, object points
        centers_xyz: (M, 3) float tensor, keypoint centers
        sigma: standard deviation of Gaussian kernel
    Returns:
        values: (N,) tensor, aggregated Gaussian (max over centers)
    """
    diff = points_xyz.unsqueeze(1) - centers_xyz.unsqueeze(0)  # (N, M, 3)
    dist2 = (diff * diff).sum(dim=-1)  # (N, M)
    vals = torch.exp(-dist2 / (2.0 * (sigma ** 2)))
    values, _ = vals.max(dim=1)
    return values


def compute_hand_keypoints(hand_model: HandModel, pose_29: torch.Tensor, max_k: int = 64) -> torch.Tensor:
    """
    Get a set of hand keypoints (e.g., fingertip mesh vertices) in world coordinates and downsample.
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


def load_pairs_db(obj_dir: str, num_hint: int = 0):
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


def process_object(obj_name: str,
                   in_root: str,
                   out_root: str,
                   sigma: float,
                   right_kps_use_center_only: bool,
                   left_kps_use_center_only: bool,
                   left_kps_max: int,
                   right_kps_max: int):
    obj_dir = os.path.join(in_root, obj_name)
    try:
        obj_points, pairs_db = load_pairs_db(obj_dir)
    except Exception as e:
        print(f"[Skip] {obj_name}: {e}")
        return False

    device = 'cpu'
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

    pairs = pairs_db.get('pairs', {})
    if isinstance(pairs, dict):
        pose_indices = sorted(list(map(int, pairs.keys())))
    else:
        # unexpected format
        print(f"[Skip] {obj_name}: invalid pairs format")
        return False

    pts = torch.from_numpy(obj_points).float()

    out_pairs = {}
    for idx in pose_indices:
        pair = pairs[idx]
        left_pose = build_hand_pose_tensor(pair['left']['qpos'], device)
        right_pose = build_hand_pose_tensor(pair['right']['qpos'], device)
        left_hand_model.set_parameters(left_pose.unsqueeze(0))
        right_hand_model.set_parameters(right_pose.unsqueeze(0))

        # Left-hand keypoints
        if left_kps_use_center_only:
            left_surface = left_hand_model.get_surface_points()[0]
            left_center = get_grasp_center_point(pts, left_surface)
            left_kps = left_center.view(1, 3)
        else:
            left_kps = compute_hand_keypoints(left_hand_model, left_pose, max_k=left_kps_max)

        # Right-hand centers for Gaussian
        centers = []
        right_surface = right_hand_model.get_surface_points()[0]
        right_center = get_grasp_center_point(pts, right_surface)
        centers.append(right_center.view(1, 3))
        if not right_kps_use_center_only:
            right_mesh_kps = compute_hand_keypoints(right_hand_model, right_pose, max_k=right_kps_max)
            centers.append(right_mesh_kps)
        right_centers = torch.cat(centers, dim=0)

        aff = gaussian_field(pts, right_centers, sigma=sigma)
        aff_np = aff.detach().cpu().numpy()
        if aff_np.max() > 0:
            aff_np = aff_np / aff_np.max()

        out_pairs[idx] = {
            'left_kps': left_kps.detach().cpu().numpy().astype(np.float32),
            'aff_scores_right': aff_np.astype(np.float32),
        }

    save_dir = os.path.join(out_root, obj_name)
    os.makedirs(save_dir, exist_ok=True)
    out_obj = {
        'object_name': obj_name,
        'points': obj_points.astype(np.float32),
        'pairs': out_pairs,
        'meta': {
            'sigma': float(sigma),
            'right_kps_use_center_only': bool(right_kps_use_center_only),
            'left_kps_use_center_only': bool(left_kps_use_center_only),
            'left_kps_max': int(left_kps_max),
            'right_kps_max': int(right_kps_max),
        }
    }
    np.save(os.path.join(save_dir, 'aff_sec_pairs.npy'), out_obj, allow_pickle=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_root', type=str, default=os.path.join(PROJ_ROOT, 'preprocess', 'results'))
    parser.add_argument('--out_root', type=str, default=os.path.join(PROJ_ROOT, 'preprocess', 'aff_sec_result'))
    parser.add_argument('--sigma', type=float, default=0.02)
    parser.add_argument('--right_kps_use_center_only', action='store_true')
    parser.add_argument('--left_kps_use_center_only', action='store_true', help='default True keeps a single left grasp center')
    parser.add_argument('--left_kps_max', type=int, default=64)
    parser.add_argument('--right_kps_max', type=int, default=32)
    args = parser.parse_args()

    # Default behavior: left center only if flag not provided
    left_center_only = True if not args.left_kps_use_center_only else True

    in_root = args.in_root
    out_root = args.out_root
    os.makedirs(out_root, exist_ok=True)

    if not os.path.isdir(in_root):
        raise NotADirectoryError(f'in_root not found: {in_root}')

    obj_names = [d for d in os.listdir(in_root) if os.path.isdir(os.path.join(in_root, d))]
    obj_names.sort()
    total = 0
    success = 0
    for obj_name in obj_names:
        total += 1
        ok = process_object(
            obj_name=obj_name,
            in_root=in_root,
            out_root=out_root,
            sigma=args.sigma,
            right_kps_use_center_only=args.right_kps_use_center_only,
            left_kps_use_center_only=left_center_only,
            left_kps_max=args.left_kps_max,
            right_kps_max=args.right_kps_max,
        )
        if ok:
            success += 1
    print(f"Done. {success}/{total} objects processed. Results saved to {out_root}")


if __name__ == '__main__':
    main()



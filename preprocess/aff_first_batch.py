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

# Reuse lightweight helpers from aff_first without triggering heavy deps at import time
from preprocess.aff_first import (
    load_pairs_db,
    build_hand_pose_tensor,
    get_grasp_center_point,
    compute_hand_keypoints,
    gaussian_field,
)


def process_one_object(obj_dir: str,
                       hand: str,
                       include_kps_from_mesh: bool,
                       kps_max: int,
                       recompute_center: bool,
                       device: str,
                       sigma: float,
                       normalize_aff: bool) -> bool:
    try:
        obj_points, pairs_db = load_pairs_db(obj_dir)
    except Exception as e:
        print(f"[Skip] {obj_dir}: {e}")
        return False

    pts_cpu = torch.from_numpy(obj_points.astype(np.float32))

    # Prepare optional hand model only when needed
    if include_kps_from_mesh or recompute_center:
        from hand_model import HandModel  # heavy deps -> import lazily
        models_dir = os.path.join(THIRD_PARTY_DIR, 'models')
        meshes_dir = os.path.join(models_dir, 'meshes')
        if hand == 'left':
            mjcf_path = os.path.join(models_dir, 'left_shadow_hand_wrist_free.xml')
            contact_json = os.path.join(models_dir, 'left_hand_contact_points.json')
            handedness = 'left_hand'
        else:
            mjcf_path = os.path.join(models_dir, 'right_shadow_hand_wrist_free.xml')
            contact_json = os.path.join(models_dir, 'right_hand_contact_points.json')
            handedness = 'right_hand'
        penetration_json = os.path.join(models_dir, 'penetration_points.json')
        hand_model = HandModel(
            mjcf_path=mjcf_path,
            mesh_path=meshes_dir,
            contact_points_path=contact_json,
            penetration_points_path=penetration_json,
            device=device,
            n_surface_points=4096,
            handedness=handedness,
        )
        pts_t = torch.from_numpy(obj_points).to(device).float()
    else:
        hand_model = None
        pts_t = None

    pairs = pairs_db.get('pairs', {})
    if not isinstance(pairs, dict) or len(pairs) == 0:
        print(f"[Skip] {obj_dir}: empty pairs")
        return False

    out_pairs = {}
    pose_indices = sorted(list(map(int, pairs.keys())))
    for idx in pose_indices:
        pair = pairs[idx]
        hand_entry = pair[hand]
        qpos = hand_entry['qpos']

        if recompute_center and hand_model is not None:
            pose = build_hand_pose_tensor(qpos, device)
            hand_model.set_parameters(pose.unsqueeze(0))
            hand_surface = hand_model.get_surface_points()[0]
            center_t = get_grasp_center_point(pts_t, hand_surface)
            center_np = center_t.detach().cpu().numpy().astype(np.float32)
        else:
            center_np = np.asarray(hand_entry['center_point'], dtype=np.float32)

        if include_kps_from_mesh and hand_model is not None:
            pose = build_hand_pose_tensor(qpos, device)
            kps_t = compute_hand_keypoints(hand_model, pose, max_k=kps_max)
            kps_np = kps_t.detach().cpu().numpy().astype(np.float32)
            kps_np = np.concatenate([center_np.reshape(1, 3), kps_np], axis=0)
        else:
            kps_np = center_np.reshape(1, 3).astype(np.float32)

        aff = gaussian_field(pts_cpu, torch.from_numpy(kps_np).float(), sigma=float(sigma))
        aff_np = aff.numpy().astype(np.float32)
        if normalize_aff and aff_np.max() > 0:
            aff_np = aff_np / aff_np.max()

        out_pairs[idx] = {
            'qpos': qpos,
            'center_point': center_np,
            'first_kps': kps_np,
            f'aff_scores_{hand}': aff_np,
        }

    obj_name = pairs_db.get('object_name', os.path.basename(obj_dir.rstrip('/')))
    save_path = os.path.join(obj_dir, f'aff_first_pairs_{hand}.npy')
    out_obj = {
        'object_name': obj_name,
        'points': obj_points.astype(np.float32),
        'hand': hand,
        'pairs': out_pairs,
        'meta': {
            'include_kps_from_mesh': bool(include_kps_from_mesh),
            'kps_max': int(kps_max),
            'recompute_center': bool(recompute_center),
            'sigma': float(sigma),
            'normalize_aff': bool(normalize_aff),
        }
    }
    np.save(save_path, out_obj, allow_pickle=True)
    print(f"[OK] {obj_name} -> {save_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_root', type=str, default=os.path.join(PROJ_ROOT, 'preprocess', 'results'))
    parser.add_argument('--hand', type=str, default='left', choices=['left', 'right'])
    parser.add_argument('--include_kps_from_mesh', action='store_true')
    parser.add_argument('--kps_max', type=int, default=64)
    parser.add_argument('--recompute_center', action='store_true')
    parser.add_argument('--device', type=str, default='auto', help="'cpu', 'cuda', or 'auto'")
    parser.add_argument('--sigma', type=float, default=0.02)
    parser.add_argument('--normalize_aff', action='store_true', default=True)
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    in_root = args.in_root
    if not os.path.isdir(in_root):
        raise NotADirectoryError(f'in_root not found: {in_root}')

    obj_dirs = [os.path.join(in_root, d) for d in os.listdir(in_root) if os.path.isdir(os.path.join(in_root, d))]
    obj_dirs.sort()

    total = 0
    success = 0
    for d in obj_dirs:
        total += 1
        ok = process_one_object(
            obj_dir=d,
            hand=args.hand,
            include_kps_from_mesh=args.include_kps_from_mesh,
            kps_max=args.kps_max,
            recompute_center=args.recompute_center,
            device=device,
            sigma=args.sigma,
            normalize_aff=args.normalize_aff,
        )
        if ok:
            success += 1
    print(f'Done. {success}/{total} objects processed for hand={args.hand}.')


if __name__ == '__main__':
    main()



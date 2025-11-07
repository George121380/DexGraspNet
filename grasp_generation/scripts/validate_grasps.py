"""
Last modified date: 2023.02.23
Author: Ruicheng Wang
Description: validate grasps on Isaac simulator
"""

import os
import sys

sys.path.append(os.path.realpath('.'))

from utils.isaac_validator import IsaacValidator
import argparse
import torch
import numpy as np
import transforms3d
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
import json as _json
from datetime import datetime as _dt

# Optional progress bar support
try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None

class _SimpleProgress:
    """Lightweight progress indicator when tqdm is unavailable."""
    def __init__(self, total, desc=''):
        self.total = max(int(total or 0), 0)
        self.desc = desc
        self.count = 0

    def update(self, n=1):
        self.count += int(n)
        if self.total > 0:
            percent = int(self.count * 100 / self.total)
            sys.stdout.write(f"\r{self.desc} {self.count}/{self.total} ({percent}%)")
        else:
            sys.stdout.write(f"\r{self.desc} {self.count}")
        sys.stdout.flush()
        if self.total > 0 and self.count >= self.total:
            sys.stdout.write("\n")

def _create_progress(total, desc):
    return _tqdm(total=total, desc=desc) if _tqdm is not None else _SimpleProgress(total, desc)

try:
    from utils.isaac_bimanual_validator import BimanualIsaacValidator
except ImportError:
    BimanualIsaacValidator = None


def _dict_to_joint_vector(qpos_dict, joint_names):
    """Convert a joint dictionary into a joint vector using the provided ordering."""
    return [float(qpos_dict[name]) for name in joint_names]


def _extract_rotation(qpos_dict, rot_names):
    """Return quaternion (w, x, y, z) derived from Euler angles stored in qpos."""
    rot = [qpos_dict[name] for name in rot_names]
    quat = transforms3d.euler.euler2quat(*rot)
    return quat


def _extract_translation(qpos_dict, translation_names):
    """Return translation vector extracted from qpos."""
    return [float(qpos_dict[name]) for name in translation_names]


def _build_pair_indices(num_left, num_right, strategy, pair_limit, rng):
    """Create pairing indices for left/right grasp sets."""
    if num_left == 0 or num_right == 0:
        return []
    if strategy == 'index':
        count = min(num_left, num_right)
        indices = [(i, i) for i in range(count)]
    elif strategy == 'random':
        count = min(num_left, num_right)
        left_perm = rng.permutation(num_left)[:count]
        right_perm = rng.permutation(num_right)[:count]
        indices = list(zip(left_perm.tolist(), right_perm.tolist()))
    else:  # cartesian
        indices = [(i, j) for i in range(num_left) for j in range(num_right)]
    if pair_limit is not None:
        indices = indices[:pair_limit]
    return indices


def _maybe_mirror_qpos(qpos_dict, enabled):
    """Placeholder for mirroring support. Currently returns the input dictionary unchanged."""
    if enabled:
        print('mirror_left_hand flag is enabled; using identity mirroring (placeholder).')
    return qpos_dict


def _save_bimanual_results(save_path, entries):
    """Persist bimanual validation results to disk."""
    import numpy as _np

    _np.save(save_path, entries, allow_pickle=True)


_ASSET_RESOLVE_CACHE = {}


def _resolve_object_asset_root(mesh_path, primary_code, secondary_code=None):
    """Locate a coacd asset folder for the object, with support for category subfolders.

    Tries the following in order (with simple memoization):
    1) <mesh_path>/<code>/coacd/coacd.urdf
    2) <mesh_path>/*/<code>/coacd/coacd.urdf (one category level)
    3) Recursive walk under mesh_path for a folder named 'coacd' containing 'coacd.urdf'
    """
    # simple cache to avoid repeated directory scans across chunks
    cache_key = (os.path.abspath(mesh_path), primary_code, secondary_code)
    cached = _ASSET_RESOLVE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    candidates = [primary_code]
    if secondary_code and secondary_code not in candidates:
        candidates.append(secondary_code)

    # direct and one-level category checks
    try:
        entries = os.listdir(mesh_path)
    except Exception:
        entries = []

    for code in candidates:
        # direct: <mesh_path>/<code>/coacd/coacd.urdf
        candidate_root = os.path.join(mesh_path, code, 'coacd')
        candidate_file = os.path.join(candidate_root, 'coacd.urdf')
        if os.path.exists(candidate_file):
            _ASSET_RESOLVE_CACHE[cache_key] = (candidate_root, 'coacd.urdf')
            return candidate_root, 'coacd.urdf'
        # one-level category: <mesh_path>/*/<code>/coacd/coacd.urdf
        for entry in entries:
            candidate_root = os.path.join(mesh_path, entry, code, 'coacd')
            candidate_file = os.path.join(candidate_root, 'coacd.urdf')
            if os.path.exists(candidate_file):
                _ASSET_RESOLVE_CACHE[cache_key] = (candidate_root, 'coacd.urdf')
                return candidate_root, 'coacd.urdf'

    # Fallback: recursive search for any 'coacd/coacd.urdf' that sits under a folder named with the code
    for dirpath, dirnames, filenames in os.walk(mesh_path):
        base = os.path.basename(dirpath)
        if base == 'coacd' and 'coacd.urdf' in filenames:
            # ensure its parent dir name matches a candidate code if possible
            parent_name = os.path.basename(os.path.dirname(dirpath))
            if parent_name in candidates:
                _ASSET_RESOLVE_CACHE[cache_key] = (dirpath, 'coacd.urdf')
                return dirpath, 'coacd.urdf'

    # As last resort, accept any coacd folder (kept for backward compatibility)
    for dirpath, dirnames, filenames in os.walk(mesh_path):
        if os.path.basename(dirpath) == 'coacd' and 'coacd.urdf' in filenames:
            _ASSET_RESOLVE_CACHE[cache_key] = (dirpath, 'coacd.urdf')
            return dirpath, 'coacd.urdf'

    _ASSET_RESOLVE_CACHE[cache_key] = (None, None)
    return None, None


def run_bimanual_validation(args, translation_names, rot_names, joint_names):
    import numpy as _np

    object_code_left = args.object_code_left or args.object_code
    object_code_right = args.object_code_right or args.object_code
    grasp_path_left = args.grasp_path_left or args.grasp_path
    grasp_path_right = args.grasp_path_right or args.grasp_path

    use_bundle = args.grasp_file_bimanual is not None

    if use_bundle:
        bundle_path = args.grasp_file_bimanual
        if not os.path.exists(bundle_path):
            print(f'Bimanual bundle file not found: {bundle_path}')
            os._exit(0)
        bundle_data = _np.load(bundle_path, allow_pickle=True)
        num_pairs = len(bundle_data)
        pair_indices = [(i, i) for i in range(num_pairs)]
        # optional limit before slicing
        if args.pair_limit is not None:
            pair_indices = pair_indices[:args.pair_limit]
        # segment slicing
        start_index = int(getattr(args, 'pair_start', 0) or 0)
        count = getattr(args, 'pair_count', None)
        if count is not None:
            pair_indices = pair_indices[start_index:start_index + int(count)]
        else:
            pair_indices = pair_indices[start_index:]
        left_data = [entry['qpos_left'] for entry in bundle_data]
        right_data = [entry['qpos_right'] for entry in bundle_data]
        scale_bundle = [entry.get('scale', 0.1) for entry in bundle_data]
        epen_left_bundle = [entry.get('E_pen_left', entry.get('E_pen', 0.0)) for entry in bundle_data]
        epen_right_bundle = [entry.get('E_pen_right', entry.get('E_pen', 0.0)) for entry in bundle_data]
    else:
        left_file = os.path.join(grasp_path_left, object_code_left + '.npy')
        right_file = os.path.join(grasp_path_right, object_code_right + '.npy')

        if not os.path.exists(left_file) or not os.path.exists(right_file):
            print(f'Bimanual mode requires grasp files on both sides. Missing: {left_file if not os.path.exists(left_file) else right_file}')
            os._exit(0)

        left_data = _np.load(left_file, allow_pickle=True)
        right_data = _np.load(right_file, allow_pickle=True)
        num_left = len(left_data)
        num_right = len(right_data)

        rng = _np.random.default_rng(0)
        pair_indices = _build_pair_indices(num_left, num_right, args.pairing, args.pair_limit, rng)
        # segment slicing
        start_index = int(getattr(args, 'pair_start', 0) or 0)
        count = getattr(args, 'pair_count', None)
        if count is not None:
            pair_indices = pair_indices[start_index:start_index + int(count)]
        else:
            pair_indices = pair_indices[start_index:]
        if not pair_indices:
            print('No bimanual pairs available for evaluation.')
            os._exit(0)

    if BimanualIsaacValidator is None:
        raise RuntimeError('BimanualIsaacValidator import failed; please ensure utils/isaac_bimanual_validator.py is available.')

    mode = 'gui' if args.gui else 'direct'

    validator = BimanualIsaacValidator(
        hand_asset_left_path=args.hand_asset_left,
        hand_asset_right_path=args.hand_asset_right,
        mode='gui' if getattr(args, 'gui', False) else 'direct',
        gpu=args.gpu,
        viewer_width=args.gui_width,
        viewer_height=args.gui_height,
    )

    if args.bimanual_views < len(validator.test_rotations):
        validator.test_rotations = validator.test_rotations[: max(1, args.bimanual_views)]
    elif args.bimanual_views > len(validator.test_rotations):
        print(f'Requested {args.bimanual_views} views; using available {len(validator.test_rotations)} views.')

    obj_root, obj_file = _resolve_object_asset_root(args.mesh_path, object_code_left, object_code_right)
    if obj_root is None:
        print('Failed to locate coacd assets for the object in bimanual mode.')
        os._exit(0)

    total_pairs = len(pair_indices)
    views_per_pair = len(validator.test_rotations)
    simulated_flags = _np.zeros(total_pairs, dtype=bool)
    estimated_flags = _np.zeros(total_pairs, dtype=bool)
    scales = _np.zeros(total_pairs, dtype=float)
    e_pen_left = _np.zeros(total_pairs, dtype=float)
    e_pen_right = _np.zeros(total_pairs, dtype=float)

    offset = 0
    global_index = 0
    results_flat = []

    progress = _create_progress(total_pairs, 'Bimanual validation')
    while offset < total_pairs:
        batch_pairs = pair_indices[offset : offset + args.val_batch_bimanual]
        validator.set_assets(obj_root, obj_file)

        for local_pair_index, (left_idx, right_idx) in enumerate(batch_pairs):
            if use_bundle:
                left_qpos_dict = _maybe_mirror_qpos(left_data[left_idx], args.mirror_left_hand)
                right_qpos_dict = right_data[right_idx]
                scale_value = float(scale_bundle[left_idx])
                left_pen = float(epen_left_bundle[left_idx])
                right_pen = float(epen_right_bundle[right_idx])
            else:
                left_entry = left_data[left_idx]
                right_entry = right_data[right_idx]
                left_qpos_dict = _maybe_mirror_qpos(left_entry['qpos'], args.mirror_left_hand)
                right_qpos_dict = right_entry['qpos']
                scale_left = float(left_entry.get('scale', 0.1))
                scale_right = float(right_entry.get('scale', 0.1))
                scale_value = max(scale_left, scale_right)
                left_pen = float(left_entry.get('E_pen', 0.0))
                right_pen = float(right_entry.get('E_pen', 0.0))

            left_rot = _extract_rotation(left_qpos_dict, rot_names)
            right_rot = _extract_rotation(right_qpos_dict, rot_names)
            left_trans = _extract_translation(left_qpos_dict, translation_names)
            right_trans = _extract_translation(right_qpos_dict, translation_names)
            left_joint_vec = _dict_to_joint_vector(left_qpos_dict, joint_names)
            right_joint_vec = _dict_to_joint_vector(right_qpos_dict, joint_names)

            validator.add_env(left_rot, left_trans, left_joint_vec, right_rot, right_trans, right_joint_vec, scale_value)

            scales[global_index + local_pair_index] = scale_value
            e_pen_left[global_index + local_pair_index] = left_pen
            e_pen_right[global_index + local_pair_index] = right_pen
            estimated_flags[global_index + local_pair_index] = (
                left_pen < args.penetration_threshold and right_pen < args.penetration_threshold
            )

        batch_results = validator.run_sim()
        results_flat.extend(batch_results)
        for local_pair_index in range(len(batch_pairs)):
            start = local_pair_index * views_per_pair
            end = start + views_per_pair
            simulated_flags[global_index + local_pair_index] = all(batch_results[start:end])

        # Per-round metrics (cumulative up to current batch)
        processed_total = global_index + len(batch_pairs)
        est_so_far = int(estimated_flags[:processed_total].sum())
        sim_so_far = int(simulated_flags[:processed_total].sum())
        valid_so_far = int((estimated_flags[:processed_total] & simulated_flags[:processed_total]).sum())
        try:
            if _tqdm is not None:
                progress.set_postfix({
                    'est': f"{est_so_far}/{processed_total}",
                    'sim': f"{sim_so_far}/{processed_total}",
                    'valid': f"{valid_so_far}/{processed_total}"
                })
            else:
                print(f"Round {processed_total}: estimated {est_so_far}/{processed_total}, simulated {sim_so_far}/{processed_total}, valid {valid_so_far}/{processed_total}")
        except Exception:
            pass

        offset += len(batch_pairs)
        global_index += len(batch_pairs)
        try:
            progress.update(len(batch_pairs))
        except Exception:
            pass
        if offset < total_pairs:
            validator.reset_simulator()

    validator.destroy()

    valid_flags = simulated_flags & estimated_flags
    est_cnt = int(estimated_flags.sum())
    sim_cnt = int(simulated_flags.sum())
    val_cnt = int(valid_flags.sum())
    print(
        f"estimated: {est_cnt}/{total_pairs}, simulated: {sim_cnt}/{total_pairs}, valid: {val_cnt}/{total_pairs}"
    )
    # Short summary line for logging/grep convenience
    print(f"est={est_cnt}/{total_pairs}, sim={sim_cnt}/{total_pairs}, valid={val_cnt}/{total_pairs}")

    save_entries = []
    for pair_position, (left_idx, right_idx) in enumerate(pair_indices):
        if not valid_flags[pair_position]:
            continue
        if use_bundle:
            save_entries.append(
                {
                    'qpos_left': left_data[left_idx],
                    'qpos_right': right_data[right_idx],
                    'scale': float(scales[pair_position]),
                    'index_left': int(left_idx),
                    'index_right': int(right_idx),
                    'E_pen_left': float(e_pen_left[pair_position]),
                    'E_pen_right': float(e_pen_right[pair_position]),
                }
            )
        else:
            left_entry = left_data[left_idx]
            right_entry = right_data[right_idx]
            save_entries.append(
                {
                    'qpos_left': left_entry['qpos'],
                    'qpos_right': right_entry['qpos'],
                    'scale': float(scales[pair_position]),
                    'index_left': int(left_idx),
                    'index_right': int(right_idx),
                    'E_pen_left': float(e_pen_left[pair_position]),
                    'E_pen_right': float(e_pen_right[pair_position]),
                }
            )

    # Segment-aware output dir and filename
    base_name = f"{object_code_left}_{object_code_right}" if object_code_left != object_code_right else object_code_left
    segment_dir = os.path.join(args.result_path, 'segments', 'bimanual', base_name)
    os.makedirs(segment_dir, exist_ok=True)
    seg_start = int(getattr(args, 'pair_start', 0) or 0)
    seg_count = int(total_pairs)
    seg_end = seg_start + seg_count
    save_name = f"{base_name}_bimanual_chunk_{seg_start}_{seg_end}.npy"
    save_path = os.path.join(segment_dir, save_name)

    _save_bimanual_results(save_path, save_entries)
    print(f'Saved {len(save_entries)} valid bimanual pairs to {save_path}')

    # Update stats.json in the same directory
    stats_path = os.path.join(segment_dir, 'stats.json')
    stats = {}
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r') as f:
                stats = _json.load(f) or {}
        except Exception:
            stats = {}
    segment_id = save_name
    stats[segment_id] = {
        'mode': 'bimanual',
        'object': base_name,
        'start': seg_start,
        'count': seg_count,
        'estimated': est_cnt,
        'simulated': sim_cnt,
        'valid': val_cnt,
        'timestamp': _dt.now().isoformat(timespec='seconds')
    }
    with open(stats_path, 'w') as f:
        _json.dump(stats, f, indent=2)
    os._exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=3, type=int)
    parser.add_argument('--val_batch', default=500, type=int)
    parser.add_argument('--mesh_path', default="../data/meshdata", type=str)
    parser.add_argument('--grasp_path', default="../data/graspdata", type=str)
    parser.add_argument('--result_path', default="../data/dataset", type=str)
    parser.add_argument('--object_code',
                        default="sem-Xbox360-d0dff348985d4f8e65ca1b579a4b8d2",
                        type=str)
    # if index is received, then the debug mode is on
    parser.add_argument('--index', type=int)
    parser.add_argument('--no_force', action='store_true')
    parser.add_argument('--thres_cont', default=0.001, type=float)
    parser.add_argument('--dis_move', default=0.001, type=float)
    parser.add_argument('--grad_move', default=500, type=float)
    parser.add_argument('--penetration_threshold', default=0.001, type=float)
    parser.add_argument('--gui', action='store_true', help='Enable Isaac Gym viewer (slower, but visualizes simulation)')
    parser.add_argument('--gui_width', type=int, default=1280, help='Viewer window width when --gui is set')
    parser.add_argument('--gui_height', type=int, default=960, help='Viewer window height when --gui is set')
    parser.add_argument('--bimanual', action='store_true', help='Enable dual-hand validation flow')
    # segment options for single-hand mode
    parser.add_argument('--start', type=int, default=0, help='Start index for single-hand validation segment')
    parser.add_argument('--count', type=int, help='Number of items to validate in single-hand segment')
    # segment options for bimanual mode
    parser.add_argument('--pair_start', type=int, default=0, help='Start index for bimanual pair segment')
    parser.add_argument('--pair_count', type=int, help='Number of pairs to validate in bimanual segment')
    parser.add_argument('--object_code_left', type=str, help='Override object code for left hand (default: object_code)')
    parser.add_argument('--object_code_right', type=str, help='Override object code for right hand (default: object_code)')
    parser.add_argument('--grasp_path_left', type=str, help='Directory for left-hand grasp npy files (default: grasp_path)')
    parser.add_argument('--grasp_path_right', type=str, help='Directory for right-hand grasp npy files (default: grasp_path)')
    parser.add_argument('--grasp_file_bimanual', type=str, help='Single npy file containing paired left/right grasps')
    parser.add_argument('--hand_asset_left', type=str, default='open_ai_assets/hand/shadow_hand.xml', help='Asset path for left hand')
    parser.add_argument('--hand_asset_right', type=str, default='open_ai_assets/hand/shadow_hand.xml', help='Asset path for right hand')
    parser.add_argument('--mirror_left_hand', action='store_true', help='Mirror right-hand joint ordering to synthesize left-hand joint angles')
    parser.add_argument('--pairing', choices=['index', 'random', 'cartesian'], default='index', help='Pairing strategy for left/right grasp sets')
    parser.add_argument('--pair_limit', type=int, help='Max number of paired grasps to evaluate for bimanual mode')
    parser.add_argument('--val_batch_bimanual', type=int, default=32, help='Batch size (number of pairs) for bimanual evaluation')
    parser.add_argument('--bimanual_views', type=int, default=4, help='Number of test orientations per pair in bimanual mode')

    args = parser.parse_args()

    translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
    rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
    joint_names = [
        'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
        'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
        'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
        'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
        'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
    ]

    # Bimanual path bypasses single-hand preparation and handles CUDA_VISIBLE_DEVICES internally.
    if args.bimanual:
        run_bimanual_validation(args, translation_names, rot_names, joint_names)

    # Guard against missing key in environments where CUDA_VISIBLE_DEVICES is unset
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.makedirs(args.result_path, exist_ok=True)

    if not args.no_force:
        device = torch.device(
            f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        # load grasp npy with segment slicing
        full_data = np.load(os.path.join(
            args.grasp_path, args.object_code + '.npy'), allow_pickle=True)
        start_index = int(getattr(args, 'start', 0) or 0)
        count = getattr(args, 'count', None)
        if count is not None:
            data_dict = full_data[start_index:start_index + int(count)]
        else:
            data_dict = full_data[start_index:]
        batch_size = data_dict.shape[0]
        hand_state = []
        scale_tensor = []
        for i in range(batch_size):
            qpos = data_dict[i]['qpos']
            scale = data_dict[i]['scale']
            rot = np.array(transforms3d.euler.euler2mat(
                *[qpos[name] for name in rot_names]))
            rot = rot[:, :2].T.ravel().tolist()
            hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [
                qpos[name] for name in joint_names], dtype=torch.float, device=device)
            hand_state.append(hand_pose)
            scale_tensor.append(scale)
        hand_state = torch.stack(hand_state).to(device).requires_grad_()
        scale_tensor = torch.tensor(scale_tensor).reshape(1, -1).to(device)
        # print(scale_tensor.dtype)
        hand_model = HandModel(
            mjcf_path='mjcf/shadow_hand_wrist_free.xml',
            mesh_path='mjcf/meshes',
            contact_points_path='mjcf/contact_points.json',
            penetration_points_path='mjcf/penetration_points.json',
            n_surface_points=2000,
            device=device
        )
        hand_model.set_parameters(hand_state)
        # object model
        object_model = ObjectModel(
            data_root_path=args.mesh_path,
            batch_size_each=batch_size,
            num_samples=0,
            device=device
        )
        object_model.initialize(args.object_code)
        object_model.object_scale_tensor = scale_tensor

        # calculate contact points and contact normals
        contact_points_hand = torch.zeros((batch_size, 19, 3)).to(device)
        contact_normals = torch.zeros((batch_size, 19, 3)).to(device)

        for i, link_name in enumerate(hand_model.mesh):
            if len(hand_model.mesh[link_name]['surface_points']) == 0:
                continue
            surface_points = hand_model.current_status[link_name].transform_points(
                hand_model.mesh[link_name]['surface_points']).expand(batch_size, -1, 3)
            surface_points = surface_points @ hand_model.global_rotation.transpose(
                1, 2) + hand_model.global_translation.unsqueeze(1)
            distances, normals = object_model.cal_distance(
                surface_points)
            nearest_point_index = distances.argmax(dim=1)
            nearest_distances = torch.gather(
                distances, 1, nearest_point_index.unsqueeze(1))
            nearest_points_hand = torch.gather(
                surface_points, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3))
            nearest_normals = torch.gather(
                normals, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3))
            admited = -nearest_distances < args.thres_cont
            admited = admited.reshape(-1, 1, 1).expand(-1, 1, 3)
            contact_points_hand[:, i:i+1, :] = torch.where(
                admited, nearest_points_hand, contact_points_hand[:, i:i+1, :])
            contact_normals[:, i:i+1, :] = torch.where(
                admited, nearest_normals, contact_normals[:, i:i+1, :])

        target_points = contact_points_hand + contact_normals * args.dis_move
        loss = (target_points.detach().clone() -
                contact_points_hand).square().sum()
        loss.backward()
        with torch.no_grad():
            hand_state[:, 9:] += hand_state.grad[:, 9:] * args.grad_move
            hand_state.grad.zero_()

    sim = IsaacValidator(gpu=args.gpu, mode='gui' if args.gui else 'direct', viewer_width=args.gui_width, viewer_height=args.gui_height)
    if (args.index is not None):
        sim = IsaacValidator(gpu=args.gpu, mode="gui", viewer_width=args.gui_width, viewer_height=args.gui_height)

    # reload grasp npy for simulation segment
    full_data = np.load(os.path.join(
        args.grasp_path, args.object_code + '.npy'), allow_pickle=True)
    start_index = int(getattr(args, 'start', 0) or 0)
    count = getattr(args, 'count', None)
    if count is not None:
        data_dict = full_data[start_index:start_index + int(count)]
    else:
        data_dict = full_data[start_index:]
    batch_size = data_dict.shape[0]
    scale_array = []
    hand_poses = []
    rotations = []
    translations = []
    E_pen_array = []
    for i in range(batch_size):
        qpos = data_dict[i]['qpos']
        scale = data_dict[i]['scale']
        rot = [qpos[name] for name in rot_names]
        rot = transforms3d.euler.euler2quat(*rot)
        rotations.append(rot)
        translations.append(np.array([qpos[name]
                            for name in translation_names]))
        hand_poses.append(np.array([qpos[name] for name in joint_names]))
        scale_array.append(scale)
        E_pen_array.append(data_dict[i]["E_pen"])
    E_pen_array = np.array(E_pen_array)
    if not args.no_force:
        hand_poses = hand_state[:, 9:]

    if (args.index is not None):
        obj_root, obj_file = _resolve_object_asset_root(args.mesh_path, args.object_code)
        if obj_root is None:
            print('Failed to locate coacd assets for the object.')
            os._exit(0)
        sim.set_asset("open_ai_assets", "hand/shadow_hand.xml", obj_root, obj_file)
        index = args.index
        sim.add_env_single(rotations[index], translations[index], hand_poses[index],
                           scale_array[index], 0)
        result = sim.run_sim()
        print(result)
    else:
        simulated = np.zeros(batch_size, dtype=np.bool8)
        offset = 0
        result = []
        progress = _create_progress(batch_size, 'Validation')
        for batch in range(batch_size // args.val_batch):
            offset_ = min(offset + args.val_batch, batch_size)
            obj_root, obj_file = _resolve_object_asset_root(args.mesh_path, args.object_code)
            if obj_root is None:
                print('Failed to locate coacd assets for the object.')
                os._exit(0)
            sim.set_asset("open_ai_assets", "hand/shadow_hand.xml", obj_root, obj_file)
            for index in range(offset, offset_):
                sim.add_env(rotations[index], translations[index], hand_poses[index],
                            scale_array[index])
            result = [*result, *sim.run_sim()]
            sim.reset_simulator()
            processed = offset_ - offset
            try:
                progress.update(processed)
            except Exception:
                pass

            # Per-round metrics (cumulative up to current batch)
            try:
                views_per_item = 6
                num_processed = len(result) // views_per_item
                if num_processed > 0:
                    simulated_partial = np.zeros(num_processed, dtype=np.bool8)
                    for i in range(num_processed):
                        start = i * views_per_item
                        end = start + views_per_item
                        simulated_partial[i] = np.array(sum(result[start:end]) == views_per_item)
                    estimated_partial = E_pen_array[:num_processed] < args.penetration_threshold
                    valid_partial = simulated_partial & estimated_partial
                    est_so_far = int(estimated_partial.sum())
                    sim_so_far = int(simulated_partial.sum())
                    valid_so_far = int(valid_partial.sum())
                    if _tqdm is not None:
                        progress.set_postfix({
                            'est': f"{est_so_far}/{num_processed}",
                            'sim': f"{sim_so_far}/{num_processed}",
                            'valid': f"{valid_so_far}/{num_processed}"
                        })
                    else:
                        print(f"Round {num_processed}: estimated {est_so_far}/{num_processed}, simulated {sim_so_far}/{num_processed}, valid {valid_so_far}/{num_processed}")
            except Exception:
                pass
            offset = offset_
        for i in range(batch_size):
            simulated[i] = np.array(sum(result[i * 6:(i + 1) * 6]) == 6)

        estimated = E_pen_array < args.penetration_threshold
        valid = simulated * estimated
        print(
            f'estimated: {estimated.sum().item()}/{batch_size}, '
            f'simulated: {simulated.sum().item()}/{batch_size}, '
            f'valid: {valid.sum().item()}/{batch_size}')
        result_list = []
        for i in range(batch_size):
            if (valid[i]):
                new_data_dict = {}
                new_data_dict["qpos"] = data_dict[i]["qpos"]
                new_data_dict["scale"] = data_dict[i]["scale"]
                result_list.append(new_data_dict)
        # segment-aware save and stats.json update
        segment_dir = os.path.join(args.result_path, 'segments', 'single', args.object_code)
        os.makedirs(segment_dir, exist_ok=True)
        seg_start = int(getattr(args, 'start', 0) or 0)
        seg_count = int(batch_size)
        seg_end = seg_start + seg_count
        save_name = f"{args.object_code}_chunk_{seg_start}_{seg_end}.npy"
        save_path = os.path.join(segment_dir, save_name)
        np.save(save_path, result_list, allow_pickle=True)
        # update stats
        est_cnt = int((E_pen_array < args.penetration_threshold).sum())
        sim_cnt = int(simulated.sum())
        val_cnt = int(valid.sum())
        stats_path = os.path.join(segment_dir, 'stats.json')
        stats = {}
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    stats = _json.load(f) or {}
            except Exception:
                stats = {}
        stats[save_name] = {
            'mode': 'single',
            'object': args.object_code,
            'start': seg_start,
            'count': seg_count,
            'estimated': est_cnt,
            'simulated': sim_cnt,
            'valid': val_cnt,
            'timestamp': _dt.now().isoformat(timespec='seconds')
        }
        with open(stats_path, 'w') as f:
            _json.dump(stats, f, indent=2)
    # Avoid interpreter teardown crashes from Isaac Gym/PhysX by exiting immediately.
    # Result files are already saved at this point.
    import os as _os
    _os._exit(0)

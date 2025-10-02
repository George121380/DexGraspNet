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


def _resolve_object_asset_root(mesh_path, primary_code, secondary_code=None):
    """Locate a coacd asset folder for the object, with fallbacks."""
    candidates = [primary_code]
    if secondary_code and secondary_code not in candidates:
        candidates.append(secondary_code)
    for code in candidates:
        candidate_root = os.path.join(mesh_path, code, 'coacd')
        candidate_file = os.path.join(candidate_root, 'coacd.urdf')
        if os.path.exists(candidate_file):
            return candidate_root, 'coacd.urdf'
    # Fallback: scan mesh_path for any directory containing coacd/coacd.urdf
    for entry in os.listdir(mesh_path):
        candidate_root = os.path.join(mesh_path, entry, 'coacd')
        candidate_file = os.path.join(candidate_root, 'coacd.urdf')
        if os.path.exists(candidate_file):
            print(f'Warning: object assets for {primary_code} not found. Using {entry} as fallback.')
            return candidate_root, 'coacd.urdf'
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
        if args.pair_limit is not None:
            pair_indices = pair_indices[:args.pair_limit]
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

        offset += len(batch_pairs)
        global_index += len(batch_pairs)
        if offset < total_pairs:
            validator.reset_simulator()

    validator.destroy()

    valid_flags = simulated_flags & estimated_flags
    print(
        f"estimated: {estimated_flags.sum()}/{total_pairs}, simulated: {simulated_flags.sum()}/{total_pairs}, valid: {valid_flags.sum()}/{total_pairs}"
    )

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

    suffix = '_bimanual.npy'
    if object_code_left != object_code_right:
        save_name = f"{object_code_left}_{object_code_right}{suffix}"
    else:
        save_name = f"{object_code_left}{suffix}"
    save_path = os.path.join(args.result_path, save_name)

    _save_bimanual_results(save_path, save_entries)
    print(f'Saved {len(save_entries)} valid bimanual pairs to {save_path}')
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
    parser.add_argument('--val_batch_bimanual', type=int, default=16, help='Batch size (number of pairs) for bimanual evaluation')
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
        data_dict = np.load(os.path.join(
            args.grasp_path, args.object_code + '.npy'), allow_pickle=True)
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

    data_dict = np.load(os.path.join(
        args.grasp_path, args.object_code + '.npy'), allow_pickle=True)
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
        sim.set_asset("open_ai_assets", "hand/shadow_hand.xml",
                       os.path.join(args.mesh_path, args.object_code, "coacd"), "coacd.urdf")
        index = args.index
        sim.add_env_single(rotations[index], translations[index], hand_poses[index],
                           scale_array[index], 0)
        result = sim.run_sim()
        print(result)
    else:
        simulated = np.zeros(batch_size, dtype=np.bool8)
        offset = 0
        result = []
        for batch in range(batch_size // args.val_batch):
            offset_ = min(offset + args.val_batch, batch_size)
            sim.set_asset("open_ai_assets", "hand/shadow_hand.xml",
                           os.path.join(args.mesh_path, args.object_code, "coacd"), "coacd.urdf")
            for index in range(offset, offset_):
                sim.add_env(rotations[index], translations[index], hand_poses[index],
                            scale_array[index])
            result = [*result, *sim.run_sim()]
            sim.reset_simulator()
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
        np.save(os.path.join(args.result_path, args.object_code +
                '.npy'), result_list, allow_pickle=True)
    # Avoid interpreter teardown crashes from Isaac Gym/PhysX by exiting immediately.
    # Result files are already saved at this point.
    import os as _os
    _os._exit(0)

import os
import sys
import argparse
import numpy as np
import torch

# Ensure local imports
CUR_DIR = os.path.dirname(__file__)
PROJ_ROOT = os.path.dirname(CUR_DIR)
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

from utils import (
    load_models_and_data,
    build_hand_pose_tensor,
    get_grasp_center_point,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_name', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--k', type=int, default=8192)
    parser.add_argument('--base_n', type=int, default=30000)
    parser.add_argument('--pre_rand_n', type=int, default=12000)
    parser.add_argument('--use_fps', action='store_true', default=True)
    parser.add_argument('--x_shift', type=float, default=0.0)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--out_dir', type=str, default=os.path.join('preprocess', 'results'))
    parser.add_argument('--outer_only', action='store_true', default=True)
    args = parser.parse_args()

    # Resolve device
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.out_dir, exist_ok=True)
    save_dir = os.path.join(args.out_dir, f'{args.object_name}')
    os.makedirs(save_dir, exist_ok=True)

    # Load object npy to know number of poses
    npy_path = os.path.join(args.result_path, args.object_name + '.npy')
    data_all = np.load(npy_path, allow_pickle=True)
    num_poses = len(data_all)

    # Build once: object points, hand models, object model
    # We will initialize using pose 0, then only update parameters per pose
    obj_points, right_pts, left_pts, data_dict, right_hand_model, left_hand_model, object_model = load_models_and_data(
        object_name=args.object_name,
        result_path=args.result_path,
        device=device,
        base_n=args.base_n,
        pre_rand_n=args.pre_rand_n,
        use_fps=args.use_fps,
        k=args.k,
        x_shift=args.x_shift,
        num_idx=0,
        outer_only=args.outer_only,
    )

    # Save obj_points once (reuse if exists)
    obj_pts_file = os.path.join(save_dir, 'obj_points.npy')
    if not os.path.exists(obj_pts_file):
        np.save(obj_pts_file, obj_points.detach().cpu().numpy())

    # Load or init aggregated pairs DB
    pairs_db_path = os.path.join(save_dir, 'grasp_pairs.npy')
    if os.path.exists(pairs_db_path):
        try:
            pairs_db = np.load(pairs_db_path, allow_pickle=True).item()
        except Exception:
            pairs_db = {}
    else:
        pairs_db = {}
    if 'object_name' not in pairs_db:
        pairs_db['object_name'] = args.object_name
    if 'pairs' not in pairs_db or not isinstance(pairs_db['pairs'], dict):
        pairs_db['pairs'] = {}

    # Tensor version of object points on device for fast per-pose computation
    pts_t = obj_points.to(device).float() if isinstance(obj_points, torch.Tensor) else torch.from_numpy(obj_points).to(device).float()

    # Iterate all poses; reuse hand models and just set parameters
    for i in range(num_poses):
        data_dict_i = data_all[i]
        right_qpos = data_dict_i['qpos_right']
        left_qpos = data_dict_i['qpos_left']

        right_pose = build_hand_pose_tensor(right_qpos, device)
        left_pose = build_hand_pose_tensor(left_qpos, device)

        right_hand_model.set_parameters(right_pose.unsqueeze(0))
        left_hand_model.set_parameters(left_pose.unsqueeze(0))

        # Compute representative centers (reference method) per pose
        right_surface = right_hand_model.get_surface_points()[0]
        left_surface = left_hand_model.get_surface_points()[0]
        right_center = get_grasp_center_point(pts_t, right_surface)
        left_center = get_grasp_center_point(pts_t, left_surface)

        pairs_db['pairs'][int(i)] = {
            'num': int(i),
            'right': {
                'qpos': right_qpos,
                'center_point': right_center.detach().cpu().numpy().tolist(),
            },
            'left': {
                'qpos': left_qpos,
                'center_point': left_center.detach().cpu().numpy().tolist(),
            }
        }

        # Periodically save (protect against long runs)
        if (i % 10) == 0:
            np.save(pairs_db_path, pairs_db, allow_pickle=True)

    # Final save
    np.save(pairs_db_path, pairs_db, allow_pickle=True)

    print('Saved object points:', obj_pts_file)
    print('Updated grasp pairs DB:', pairs_db_path)


if __name__ == '__main__':
    main()





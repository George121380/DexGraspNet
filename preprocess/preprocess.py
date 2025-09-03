import os
import sys
import argparse
import numpy as np
import torch

# Ensure local imports from preprocess/ work when running this file directly
CUR_DIR = os.path.dirname(__file__)
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)
from utils import (
    load_models_and_data,
    get_grasp_center_point,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_name', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--k', type=int, default=8192)
    parser.add_argument('--base_n', type=int, default=30000)
    parser.add_argument('--pre_rand_n', type=int, default=12000)
    parser.add_argument('--use_fps', action='store_true', default=True)
    parser.add_argument('--x_shift', type=float, default=0.0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out_dir', type=str, default=os.path.join('preprocess', 'results'))
    parser.add_argument('--outer_only', action='store_true', default=True, help='sample only outer surface when building obj_points')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data and models, sample object points and hand surface points
    obj_points, right_pts, left_pts, data_dict, right_hand_model, left_hand_model, object_model = load_models_and_data(
        object_name=args.object_name,
        result_path=args.result_path,
        device=args.device,
        base_n=args.base_n,
        pre_rand_n=args.pre_rand_n,
        use_fps=args.use_fps,
        k=args.k,
        x_shift=args.x_shift,
        num_idx=args.num,
        outer_only=args.outer_only,
    )

    # Compute representative center points per hand
    right_center = get_grasp_center_point(obj_points, right_pts.to(dtype=torch.float, device=args.device))
    left_center = get_grasp_center_point(obj_points, left_pts.to(dtype=torch.float, device=args.device))

    # Save directory per object; save object point cloud once
    save_dir = os.path.join(args.out_dir, f'{args.object_name}')
    os.makedirs(save_dir, exist_ok=True)
    obj_pts_np = obj_points.detach().cpu().numpy()
    obj_pts_file = os.path.join(save_dir, 'obj_points.npy')
    if not os.path.exists(obj_pts_file):
        np.save(obj_pts_file, obj_pts_np)

    # Save hand poses and center pairs
    # Restore original qpos dicts
    right_qpos = data_dict['qpos_right']
    left_qpos = data_dict['qpos_left']

    # Aggregate all pairs into a single file per object
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

    pairs_db['pairs'][int(args.num)] = {
        'num': int(args.num),
        'right': {
            'qpos': right_qpos,
            'center_point': right_center.detach().cpu().numpy().tolist(),
        },
        'left': {
            'qpos': left_qpos,
            'center_point': left_center.detach().cpu().numpy().tolist(),
        }
    }

    np.save(pairs_db_path, pairs_db, allow_pickle=True)

    print('Saved object points:', obj_pts_file)
    print('Updated grasp pairs DB:', pairs_db_path)


if __name__ == '__main__':
    main()



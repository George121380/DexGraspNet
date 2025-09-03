import os
import sys
import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

# Make local and third_party modules importable
CUR_DIR = os.path.dirname(__file__)
PROJ_ROOT = os.path.dirname(CUR_DIR)
THIRD_PARTY_DIR = os.path.join(PROJ_ROOT, 'third_party', 'BimanGrasp-Dataset')
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
if THIRD_PARTY_DIR not in sys.path:
    sys.path.insert(0, THIRD_PARTY_DIR)

from preprocess.utils import build_hand_pose_tensor
from hand_model import HandModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='preprocess/results/<obj_name> directory')
    parser.add_argument('--num', type=int, default=0, help='pose index to visualize')
    parser.add_argument('--save_html', type=str, default=None)
    args = parser.parse_args()

    obj_pts_path = os.path.join(args.dir, 'obj_points.npy')
    # aggregated pairs file
    pairs_db_path = os.path.join(args.dir, 'grasp_pairs.npy')
    if not os.path.exists(obj_pts_path):
        raise FileNotFoundError('Expected obj_points.npy in the directory.')
    if not os.path.exists(pairs_db_path):
        # legacy support
        legacy_path = os.path.join(args.dir, f'grasp_pairs_{args.num}.npy')
        if not os.path.exists(legacy_path):
            legacy_path2 = os.path.join(args.dir, 'grasp_pairs.npy')
        else:
            legacy_path2 = legacy_path
        if not os.path.exists(legacy_path2):
            raise FileNotFoundError('Expected grasp_pairs.npy (aggregated) or grasp_pairs_<num>.npy (legacy).')
        pairs_db = np.load(legacy_path2, allow_pickle=True).item()
        if 'pairs' in pairs_db:
            # already in aggregated format
            pass
        else:
            # single pair file -> wrap to aggregated
            pairs_db = {
                'object_name': os.path.basename(args.dir.rstrip('/')),
                'pairs': {int(args.num): pairs_db}
            }
    else:
        pairs_db = np.load(pairs_db_path, allow_pickle=True).item()

    if 'pairs' not in pairs_db or len(pairs_db['pairs']) == 0:
        raise ValueError('No pairs found in the aggregated grasp_pairs.npy')

    if int(args.num) not in pairs_db['pairs']:
        available = sorted(list(map(int, pairs_db['pairs'].keys())))
        raise KeyError(f"Pose index {args.num} not found. Available indices: {available}")

    pair = pairs_db['pairs'][int(args.num)]

    obj_points = np.load(obj_pts_path)
    right_center = np.array(pair['right']['center_point'])
    left_center = np.array(pair['left']['center_point'])

    # Build hand models from recorded qpos and overlay them
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
        handedness='left_hand'
    )
    right_hand_model = HandModel(
        mjcf_path=right_mjcf,
        mesh_path=meshes_dir,
        contact_points_path=right_contact_json,
        penetration_points_path=penetration_json,
        device=device,
        handedness='right_hand'
    )

    right_pose = build_hand_pose_tensor(pair['right']['qpos'], device)
    left_pose = build_hand_pose_tensor(pair['left']['qpos'], device)
    right_hand_model.set_parameters(right_pose.unsqueeze(0))
    left_hand_model.set_parameters(left_pose.unsqueeze(0))

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]])

    # Left panel: point cloud with both centers and hands
    fig.add_trace(go.Scatter3d(x=obj_points[:,0], y=obj_points[:,1], z=obj_points[:,2],
                               mode='markers', marker=dict(size=2, color='lightgray')), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=[right_center[0]], y=[right_center[1]], z=[right_center[2]],
                               mode='markers', marker=dict(size=6, color='red'), name='right_center'), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=[left_center[0]], y=[left_center[1]], z=[left_center[2]],
                               mode='markers', marker=dict(size=6, color='blue'), name='left_center'), row=1, col=1)
    # Add hand meshes (regenerate traces for each subplot to avoid reuse issues)
    right_traces_c1 = right_hand_model.get_plotly_data(i=0, opacity=0.8, color='lightslategray', with_contact_points=False)
    left_traces_c1 = left_hand_model.get_plotly_data(i=0, opacity=0.8, color='powderblue', with_contact_points=False)
    for t in right_traces_c1 + left_traces_c1:
        fig.add_trace(t, row=1, col=1)

    # Right panel: centers and hands
    fig.add_trace(go.Scatter3d(x=[right_center[0]], y=[right_center[1]], z=[right_center[2]],
                               mode='markers', marker=dict(size=6, color='red'), name='right_center'), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=[left_center[0]], y=[left_center[1]], z=[left_center[2]],
                               mode='markers', marker=dict(size=6, color='blue'), name='left_center'), row=1, col=2)
    right_traces_c2 = right_hand_model.get_plotly_data(i=0, opacity=0.8, color='lightslategray', with_contact_points=False)
    left_traces_c2 = left_hand_model.get_plotly_data(i=0, opacity=0.8, color='powderblue', with_contact_points=False)
    for t in right_traces_c2 + left_traces_c2:
        fig.add_trace(t, row=1, col=2)

    for c in [1, 2]:
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data', row=1, col=c)

    fig.update_layout(title='Preprocessed Grasp Visualization', title_x=0.5)

    # Default save path: save into the input directory if not specified
    if args.save_html is None:
        default_path = os.path.join(args.dir, f'vis_{args.num}.html')
        fig.write_html(default_path)
    else:
        os.makedirs(os.path.dirname(args.save_html), exist_ok=True)
        fig.write_html(args.save_html)
    fig.show()


if __name__ == '__main__':
    main()



import os
import sys
import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Make local imports possible
CUR_DIR = os.path.dirname(__file__)
PROJ_ROOT = os.path.dirname(CUR_DIR)
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='directory that contains aff_first_pairs_<hand>.npy')
    parser.add_argument('--hand', type=str, default='left', choices=['left', 'right'])
    parser.add_argument('--num', type=int, default=0, help='pose index to visualize')
    parser.add_argument('--save_html', type=str, default=None)
    args = parser.parse_args()

    aff_path = os.path.join(args.dir, f'aff_first_pairs_{args.hand}.npy')
    if not os.path.exists(aff_path):
        raise FileNotFoundError(f'Missing file: {aff_path}. Run aff_first.py or aff_first_batch.py first.')

    data = np.load(aff_path, allow_pickle=True).item()
    obj_points = data['points']
    pairs = data['pairs']
    if int(args.num) not in pairs:
        available = sorted(list(map(int, pairs.keys())))
        raise KeyError(f'Pose {args.num} not found. Available: {available}')

    rec = pairs[int(args.num)]
    center = np.asarray(rec['center_point']).reshape(3)
    kps = np.asarray(rec['first_kps']).reshape(-1, 3)

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]])

    # Left: object points + center
    fig.add_trace(go.Scatter3d(
        x=obj_points[:, 0], y=obj_points[:, 1], z=obj_points[:, 2],
        mode='markers', marker=dict(size=2, color='lightgray'), name='object'
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[center[0]], y=[center[1]], z=[center[2]],
        mode='markers', marker=dict(size=6, color='red'), name='center'
    ), row=1, col=1)

    # Right: object points + keypoints
    fig.add_trace(go.Scatter3d(
        x=obj_points[:, 0], y=obj_points[:, 1], z=obj_points[:, 2],
        mode='markers', marker=dict(size=2, color='lightgray'), name='object'
    ), row=1, col=2)
    fig.add_trace(go.Scatter3d(
        x=kps[:, 0], y=kps[:, 1], z=kps[:, 2],
        mode='markers', marker=dict(size=4, color='blue'), name='first_kps'
    ), row=1, col=2)

    for c in [1, 2]:
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data', row=1, col=c)
    fig.update_layout(title=f'Aff First Visualization (hand={args.hand}, pose={args.num})', title_x=0.5)

    if args.save_html is None:
        default_path = os.path.join(args.dir, f'vis_aff_first_{args.hand}_{args.num}.html')
        fig.write_html(default_path)
    else:
        os.makedirs(os.path.dirname(args.save_html), exist_ok=True)
        fig.write_html(args.save_html)
    fig.show()


if __name__ == '__main__':
    main()



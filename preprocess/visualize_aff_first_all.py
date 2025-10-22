import os
import sys
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Visualize all first-hand keypoints across poses from aff_first_pairs_left.npy')
    parser.add_argument('--dir', type=str, required=True, help='Directory containing obj_points.npy and aff_first_pairs_left.npy')
    parser.add_argument('--save_html', type=str, required=True, help='Output HTML path')
    parser.add_argument('--vis_sample_k', type=int, default=0, help='Subsample this many object points for speed (0 = use all)')
    args = parser.parse_args()

    obj_pts_path = os.path.join(args.dir, 'obj_points.npy')
    aff_first_path = os.path.join(args.dir, 'aff_first_pairs_left.npy')
    if not os.path.isfile(obj_pts_path):
        raise FileNotFoundError('obj_points.npy not found in ' + args.dir)
    if not os.path.isfile(aff_first_path):
        raise FileNotFoundError('aff_first_pairs_left.npy not found in ' + args.dir)

    obj_points = np.load(obj_pts_path)
    data = np.load(aff_first_path, allow_pickle=True).item()
    pairs = data.get('pairs', {})
    if not isinstance(pairs, dict) or len(pairs) == 0:
        raise ValueError('pairs empty in aff_first_pairs_left.npy')

    # accumulate all keypoints across poses
    all_kps = []
    for idx in sorted(map(int, pairs.keys())):
        rec = pairs[idx]
        kps = np.asarray(rec['first_kps']).reshape(-1, 3)
        all_kps.append(kps)
    if len(all_kps) == 0:
        all_kps = np.zeros((0, 3), dtype=np.float32)
    else:
        all_kps = np.concatenate(all_kps, axis=0).astype(np.float32)

    # optional subsampling of points
    points_xyz = obj_points.astype(np.float32)
    if isinstance(args.vis_sample_k, int) and args.vis_sample_k > 0 and points_xyz.shape[0] > args.vis_sample_k:
        sel = np.random.choice(points_xyz.shape[0], args.vis_sample_k, replace=False)
        points_xyz = points_xyz[sel]

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=points_xyz[:, 0], y=points_xyz[:, 1], z=points_xyz[:, 2],
        mode='markers', marker=dict(size=2, color='lightgray'), name='object'))
    if all_kps.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=all_kps[:, 0], y=all_kps[:, 1], z=all_kps[:, 2],
            mode='markers', marker=dict(size=3, color='red'), name='all_left_kps'))
    fig.update_layout(title='All left-hand keypoints across poses', title_x=0.5, scene=dict(aspectmode='data'))
    os.makedirs(os.path.dirname(args.save_html), exist_ok=True)
    fig.write_html(args.save_html)
    print('[vis_aff_first_all] saved:', args.save_html)


if __name__ == '__main__':
    main()



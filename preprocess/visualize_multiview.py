import os
import sys
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Multi-pose preview: left keypoints and right affordance for several pose indices')
    parser.add_argument('--dir', type=str, required=True, help='Directory with obj_points.npy, aff_first_pairs_left.npy, aff_sec_pairs.npy')
    parser.add_argument('--pose_indices', type=str, default='0,1,2', help='Comma-separated pose indices to visualize')
    parser.add_argument('--save_html', type=str, required=True, help='Output HTML path')
    parser.add_argument('--vis_sample_k', type=int, default=0, help='Subsample object points for speed (0 = use all)')
    args = parser.parse_args()

    obj_pts_path = os.path.join(args.dir, 'obj_points.npy')
    aff_first_path = os.path.join(args.dir, 'aff_first_pairs_left.npy')
    aff_sec_path = os.path.join(args.dir, 'aff_sec_pairs.npy')
    if not os.path.isfile(obj_pts_path):
        raise FileNotFoundError('obj_points.npy not found: ' + obj_pts_path)
    if not os.path.isfile(aff_first_path):
        raise FileNotFoundError('aff_first_pairs_left.npy not found: ' + aff_first_path)
    if not os.path.isfile(aff_sec_path):
        raise FileNotFoundError('aff_sec_pairs.npy not found: ' + aff_sec_path)

    obj_points = np.load(obj_pts_path)
    first = np.load(aff_first_path, allow_pickle=True).item()
    sec = np.load(aff_sec_path, allow_pickle=True).item()
    pairs_first = first.get('pairs', {})
    pairs_sec = sec.get('pairs', {})
    if not pairs_first or not pairs_sec:
        raise ValueError('pairs missing in aff_first or aff_sec')

    indices = [int(x.strip()) for x in args.pose_indices.split(',') if x.strip()!='']
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    rows = len(indices)
    fig = make_subplots(rows=rows, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}] for _ in range(rows)],
                        subplot_titles=[f'pose={i} left_kps' if c==0 else f'pose={i} right_aff' for i in indices for c in range(2)])

    # optional subsampling once
    points_xyz_full = obj_points.astype(np.float32)
    if isinstance(args.vis_sample_k, int) and args.vis_sample_k > 0 and points_xyz_full.shape[0] > args.vis_sample_k:
        sel = np.random.choice(points_xyz_full.shape[0], args.vis_sample_k, replace=False)
        points_vis = points_xyz_full[sel]
    else:
        points_vis = points_xyz_full

    for r, i in enumerate(indices, start=1):
        if i not in pairs_first or i not in pairs_sec:
            continue
        # left keypoints
        kps = np.asarray(pairs_first[i]['first_kps']).reshape(-1, 3)
        fig.add_trace(go.Scatter3d(x=points_xyz_full[:,0], y=points_xyz_full[:,1], z=points_xyz_full[:,2],
                                   mode='markers', marker=dict(size=2, color='lightgray'), name=f'obj_{i}'), row=r, col=1)
        if kps.size > 0:
            fig.add_trace(go.Scatter3d(x=kps[:,0], y=kps[:,1], z=kps[:,2],
                                       mode='markers', marker=dict(size=3, color='blue'), name=f'kps_{i}'), row=r, col=1)
        # right affordance
        aff = np.asarray(pairs_sec[i]['aff_scores_right']).reshape(-1)
        aff_points = points_vis
        aff_vals = aff if aff_points.shape[0] == obj_points.shape[0] else aff[np.random.choice(obj_points.shape[0], aff_points.shape[0], replace=False)]
        fig.add_trace(go.Scatter3d(x=aff_points[:,0], y=aff_points[:,1], z=aff_points[:,2],
                                   mode='markers', marker=dict(size=2, color=aff_vals, colorscale='Viridis', showscale=True),
                                   name=f'aff_{i}'), row=r, col=2)

    for r in range(1, rows+1):
        for c in [1,2]:
            fig.update_scenes(aspectmode='data', xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, row=r, col=c)
    os.makedirs(os.path.dirname(args.save_html), exist_ok=True)
    fig.update_layout(title='Multi-pose preview (left kps & right affordance)', title_x=0.5)
    fig.write_html(args.save_html)
    print('[visualize_multiview] saved:', args.save_html)


if __name__ == '__main__':
    main()



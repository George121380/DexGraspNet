import os
import sys
import argparse
import numpy as np
import torch

# Allow importing project-local and third_party modules
CUR_DIR = os.path.dirname(__file__)
PROJ_ROOT = os.path.dirname(CUR_DIR)
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from utils import (
    ensure_object_symlink,
    gaussian_field,
    make_hand_models,
    build_hand_pose_tensor,
    get_grasp_center_point,
    load_object_points_and_models,
)


def run_all_in_one(
    object_name: str,
    result_path: str,
    output_root: str,
    sapien_mesh_root: str,
    k: int,
    base_n: int,
    x_shift: float,
    device: str,
    outer_only: bool,
    sigma: float,
    visualize: bool,
    log_every: int,
    vis_sample_k: int,
    debug: bool,
    vis_pose_indices: str,
):
    """
    All-in-one preprocessing pipeline for a single object.

    Args:
        object_name: Object identifier (directory name) used across datasets.
        result_path: Root holding converted per-object npy (e.g., converted_biman_format/<OBJ>.npy).
        output_root: Root directory where artifacts will be written (one subdir per object).
        sapien_mesh_root: Root to SAPIEN preprocessed meshes (<root>/<OBJ>/coacd/decomposed.obj).
        k: Number of object point samples to produce for obj_points.npy.
        base_n: Base sample count used by surface sampling before downsampling.
        x_shift: Optional translation on x-axis applied to object mesh before sampling.
        device: 'cpu', 'cuda', or 'auto' for automatic selection.
        outer_only: When True, sample outer surface (voxel shell fallback) to avoid internal surfaces.
        sigma: Gaussian kernel sigma (meters) for affordance fields.
        visualize: When True, write a small preview HTML (pose 0) with left keypoints and right affordance.
        log_every: Print progress every N poses for long runs.
        vis_sample_k: When > 0, randomly subsample this many object points for the preview visualization.
        debug: If True, stop right after writing obj_points.npy (for quick point cloud debugging).
        vis_pose_indices: Comma-separated pose indices for multi-pose preview (e.g., '0,10,50').
    """
    os.makedirs(output_root, exist_ok=True)
    print(f"[AIO] object={object_name}  device={device}  k={k}  outer_only={outer_only}")
    # Ensure mesh symlink for this object under Object-Release-v1
    obj_dir_rel = ensure_object_symlink(object_name, sapien_mesh_root)
    print(f"[AIO] mesh symlink ready: {obj_dir_rel}")

    # 1) Sample object points and build hand models without relying on preprocess/ module
    obj_points, left_hand_model, right_hand_model = load_object_points_and_models(
        object_name=object_name,
        result_path=result_path,
        sapien_mesh_root=sapien_mesh_root,
        device=device,
        k=k,
        base_n=base_n,
        x_shift=x_shift,
        outer_only=outer_only,
    )
    print(f"[AIO] object points sampled: {obj_points.shape}")

    save_dir = os.path.join(output_root, object_name)
    os.makedirs(save_dir, exist_ok=True)
    obj_pts_file = os.path.join(save_dir, 'obj_points.npy')
    if not os.path.exists(obj_pts_file):
        np.save(obj_pts_file, obj_points.detach().cpu().numpy())
        print(f"[AIO] saved obj_points: {obj_pts_file}")
    else:
        print(f"[AIO] obj_points exists: {obj_pts_file}")

    if debug:
        # Auto-write a quick point cloud preview HTML (print path for manual opening)
        try:
            pts_np = obj_points.detach().cpu().numpy() if isinstance(obj_points, torch.Tensor) else np.asarray(obj_points)
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Scatter3d(
                x=pts_np[:, 0], y=pts_np[:, 1], z=pts_np[:, 2],
                mode='markers', marker=dict(size=2, color='steelblue'))])
            fig.update_layout(title=f'Debug Preview: object={object_name} (points={pts_np.shape[0]})', title_x=0.5)
            debug_html = os.path.join(save_dir, 'debug_obj_points.html')
            fig.write_html(debug_html)
            print(f"[AIO] debug preview saved (open in browser): {debug_html}")
        except Exception as e:
            print(f"[AIO] debug preview failed: {e}")
        print("[AIO] debug mode: stop after point cloud sampling.")
        return

    # Load original converted npy to iterate all poses
    npy_path = os.path.join(result_path, object_name + '.npy')
    data_all = np.load(npy_path, allow_pickle=True)
    num_poses = len(data_all)
    print(f"[AIO] poses found: {num_poses}")

    pairs_db_path = os.path.join(save_dir, 'grasp_pairs.npy')
    pairs_db = {'object_name': object_name, 'pairs': {}}

    pts_t = obj_points.to(device).float() if isinstance(obj_points, torch.Tensor) else torch.from_numpy(obj_points).to(device).float()

    for i in range(num_poses):
        rec = data_all[i]
        right_qpos = rec['qpos_right']
        left_qpos = rec['qpos_left']
        right_pose = build_hand_pose_tensor(right_qpos, device)
        left_pose = build_hand_pose_tensor(left_qpos, device)

        right_hand_model.set_parameters(right_pose.unsqueeze(0))
        left_hand_model.set_parameters(left_pose.unsqueeze(0))

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
        if (i % max(1, log_every)) == 0:
            print(f"[AIO] pairs written: {i}/{num_poses}")
            np.save(pairs_db_path, pairs_db, allow_pickle=True)

    np.save(pairs_db_path, pairs_db, allow_pickle=True)
    print(f"[AIO] saved grasp_pairs: {pairs_db_path}")

    # 2) First-hand (left by default) affordance using grasp center only
    left_model, _ = make_hand_models(device)
    pts_cpu = pts_t.detach().cpu()

    out_pairs = {}
    for i in range(num_poses):
        pair = pairs_db['pairs'][int(i)]['left']
        center_np = np.asarray(pair['center_point'], dtype=np.float32)

        # keypoints: grasp center only
        kps_np = center_np.reshape(1, 3).astype(np.float32)

        aff = gaussian_field(pts_cpu, torch.from_numpy(kps_np).float(), sigma=float(sigma))
        aff_np = aff.numpy().astype(np.float32)
        if aff_np.max() > 0:
            aff_np = aff_np / aff_np.max()

        out_pairs[int(i)] = {
            'qpos': pair['qpos'],
            'center_point': center_np,
            'first_kps': kps_np,
            'aff_scores_left': aff_np,
        }
        if (i % max(1, log_every)) == 0:
            print(f"[AIO] aff_first progress: {i}/{num_poses}  kps={kps_np.shape[0]}")

    out_first = {
        'object_name': object_name,
        'points': pts_cpu.numpy().astype(np.float32),
        'hand': 'left',
        'pairs': out_pairs,
        'meta': {
            'sigma': float(sigma),
        }
    }
    first_path = os.path.join(save_dir, 'aff_first_pairs_left.npy')
    np.save(first_path, out_first, allow_pickle=True)
    print(f"[AIO] saved aff_first_pairs_left: {first_path}")

    # 3) Second-stage (right) affordance pairs file using right grasp center only
    _, right_model = make_hand_models(device)
    out_sec_pairs = {}
    for i in range(num_poses):
        left_entry = pairs_db['pairs'][int(i)]['left']
        right_entry = pairs_db['pairs'][int(i)]['right']
        right_pose = build_hand_pose_tensor(right_entry['qpos'], device)
        right_model.set_parameters(right_pose.unsqueeze(0))
        right_surface = right_model.get_surface_points()[0]
        right_center = get_grasp_center_point(pts_t, right_surface)
        centers = right_center.view(1, 3)
        aff = gaussian_field(pts_t, centers, sigma=sigma)
        aff_np = aff.detach().cpu().numpy()
        if aff_np.max() > 0:
            aff_np = aff_np / aff_np.max()
        out_sec_pairs[int(i)] = {
            'left_kps': np.asarray(out_pairs[int(i)]['first_kps']).astype(np.float32),
            'aff_scores_right': aff_np.astype(np.float32),
        }
        if (i % max(1, log_every)) == 0:
            print(f"[AIO] aff_sec progress: {i}/{num_poses}  centers={centers.shape[0]}")

    out_sec = {
        'object_name': object_name,
        'points': pts_cpu.numpy().astype(np.float32),
        'pairs': out_sec_pairs,
        'meta': {
            'sigma': float(sigma),
        }
    }
    sec_path = os.path.join(save_dir, 'aff_sec_pairs.npy')
    np.save(sec_path, out_sec, allow_pickle=True)
    print(f"[AIO] saved aff_sec_pairs: {sec_path}")

    # Optional visualization
    if visualize:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            # Simple one-pose visualizations (pose 0): centers and affordances
            pose_idx = 0
            points_xyz = pts_cpu.numpy()
            first = out_pairs[pose_idx]
            # left keypoints
            kps = np.asarray(first['first_kps']).reshape(-1, 3)
            fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]])
            fig.add_trace(go.Scatter3d(x=points_xyz[:,0], y=points_xyz[:,1], z=points_xyz[:,2],
                                       mode='markers', marker=dict(size=2, color='lightgray'), name='object'), row=1, col=1)
            fig.add_trace(go.Scatter3d(x=kps[:,0], y=kps[:,1], z=kps[:,2],
                                       mode='markers', marker=dict(size=4, color='blue'), name='left_kps'), row=1, col=1)
            # right affordance (optionally subsampled for faster rendering)
            sec0 = out_sec_pairs[pose_idx]['aff_scores_right']
            sec0 = np.asarray(sec0)
            if isinstance(vis_sample_k, int) and vis_sample_k > 0 and points_xyz.shape[0] > vis_sample_k:
                sel = np.random.choice(points_xyz.shape[0], vis_sample_k, replace=False)
                points_vis = points_xyz[sel]
                sec0_vis = sec0[sel]
                print(f"[AIO] preview subsample: {vis_sample_k}/{points_xyz.shape[0]} points")
            else:
                points_vis = points_xyz
                sec0_vis = sec0
            fig.add_trace(go.Scatter3d(x=points_vis[:,0], y=points_vis[:,1], z=points_vis[:,2],
                                       mode='markers', marker=dict(size=2, color=sec0_vis, colorscale='Viridis', showscale=True),
                                       name='right_aff'), row=1, col=2)
            fig.update_layout(title=f'All-in-one preview (object={object_name}, pose=0)', title_x=0.5)
            preview_path = os.path.join(save_dir, 'all_in_one_preview_0.html')
            fig.write_html(preview_path)
            print(f"[AIO] saved preview: {preview_path}")

            # Extra 1: aggregate all left keypoints across poses
            try:
                all_kps = []
                for idx in sorted(out_pairs.keys()):
                    rec = out_pairs[idx]
                    kk = np.asarray(rec['first_kps']).reshape(-1, 3)
                    if kk.size > 0:
                        all_kps.append(kk)
                if len(all_kps) > 0:
                    all_kps = np.concatenate(all_kps, axis=0)
                else:
                    all_kps = np.zeros((0, 3), dtype=np.float32)
                fig_all = go.Figure()
                fig_all.add_trace(go.Scatter3d(x=points_xyz[:,0], y=points_xyz[:,1], z=points_xyz[:,2],
                                                mode='markers', marker=dict(size=2, color='lightgray'), name='object'))
                if all_kps.shape[0] > 0:
                    fig_all.add_trace(go.Scatter3d(x=all_kps[:,0], y=all_kps[:,1], z=all_kps[:,2],
                                                   mode='markers', marker=dict(size=3, color='red'), name='all_left_kps'))
                fig_all.update_layout(title='All left-hand keypoints across poses', title_x=0.5, scene=dict(aspectmode='data'))
                all_path = os.path.join(save_dir, 'vis_aff_first_all.html')
                fig_all.write_html(all_path)
                print(f"[AIO] saved: {all_path}")
            except Exception as e:
                print(f"[AIO] vis_aff_first_all failed: {e}")

            # Extra 2: multi-pose double scene preview (left kps & right affordance)
            try:
                idx_list = [int(x.strip()) for x in str(vis_pose_indices).split(',') if x.strip()!='']
                rows = len(idx_list)
                fig_m = make_subplots(rows=rows, cols=2,
                                      specs=[[{"type": "scene"}, {"type": "scene"}] for _ in range(rows)],
                                      subplot_titles=[f'pose={i} left_kps' if c==0 else f'pose={i} right_aff' for i in idx_list for c in range(2)])
                # precompute subsample set for aff
                points_xyz_full = points_xyz
                if isinstance(vis_sample_k, int) and vis_sample_k > 0 and points_xyz_full.shape[0] > vis_sample_k:
                    sel = np.random.choice(points_xyz_full.shape[0], vis_sample_k, replace=False)
                    points_vis_m = points_xyz_full[sel]
                    sel_idx = sel
                else:
                    points_vis_m = points_xyz_full
                    sel_idx = None
                for r, i in enumerate(idx_list, start=1):
                    if i not in out_pairs or i not in out_sec_pairs:
                        continue
                    kps_i = np.asarray(out_pairs[i]['first_kps']).reshape(-1, 3)
                    fig_m.add_trace(go.Scatter3d(x=points_xyz_full[:,0], y=points_xyz_full[:,1], z=points_xyz_full[:,2],
                                                 mode='markers', marker=dict(size=2, color='lightgray'), name=f'obj_{i}'), row=r, col=1)
                    if kps_i.size > 0:
                        fig_m.add_trace(go.Scatter3d(x=kps_i[:,0], y=kps_i[:,1], z=kps_i[:,2],
                                                     mode='markers', marker=dict(size=3, color='blue'), name=f'kps_{i}'), row=r, col=1)
                    aff_i = np.asarray(out_sec_pairs[i]['aff_scores_right']).reshape(-1)
                    if sel_idx is None:
                        aff_vals = aff_i
                    else:
                        aff_vals = aff_i[sel_idx]
                    fig_m.add_trace(go.Scatter3d(x=points_vis_m[:,0], y=points_vis_m[:,1], z=points_vis_m[:,2],
                                                 mode='markers', marker=dict(size=2, color=aff_vals, colorscale='Viridis', showscale=True),
                                                 name=f'aff_{i}'), row=r, col=2)
                for r in range(1, rows+1):
                    for c in [1,2]:
                        fig_m.update_scenes(aspectmode='data', xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, row=r, col=c)
                mpath = os.path.join(save_dir, 'vis_multi_poses.html')
                fig_m.update_layout(title='Multi-pose preview (left kps & right affordance)', title_x=0.5)
                fig_m.write_html(mpath)
                print(f"[AIO] saved: {mpath}")
            except Exception as e:
                print(f"[AIO] visualize_multiview failed: {e}")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description='All-in-one preprocessing for bimanual grasp data (SAPIEN→Biman).')
    parser.add_argument('--object_name', type=str, required=True, help='Object directory/name to process (e.g., 100017).')
    parser.add_argument('--result_path', type=str, required=True, help='Root of converted per-object npy files (e.g., .../converted_biman_format).')
    parser.add_argument('--sapien_mesh_root', type=str, required=True, help='Root of SAPIEN preprocessed meshes (<root>/<OBJ>/coacd/decomposed.obj).')
    parser.add_argument('--output_root', type=str, required=True, help='Output root; artifacts will be saved under <output_root>/<OBJ>/.')
    parser.add_argument('--k', type=int, default=8192, help='Number of points to sample for the object point cloud (obj_points.npy).')
    parser.add_argument('--base_n', type=int, default=30000, help='Base number of raw samples used before downsampling (surface sampling).')
    # removed unused args: pre_rand_n, use_fps
    parser.add_argument('--x_shift', type=float, default=0.0, help='Translate object mesh along +X before sampling (meters).')
    parser.add_argument('--device', type=str, default='auto', help="Computation device: 'cpu', 'cuda', or 'auto' to choose automatically.")
    parser.add_argument('--outer_only', action='store_true', default=True, help='Sample only the outer surface (voxel-shell fallback) to avoid internal faces.')
    parser.add_argument('--sigma', type=float, default=0.02, help='Gaussian sigma (meters) for affordance fields (left/right).')
    # removed mesh keypoints options: include_kps_from_mesh, kps_max
    parser.add_argument('--visualize', action='store_true', help='Write a small HTML preview (pose 0) with left keypoints and right affordance.')
    parser.add_argument('--vis_sample_k', type=int, default=0, help='Subsample this many points for preview visualization (0 = use all).')
    parser.add_argument('--log_every', type=int, default=100, help='print progress every N poses')
    parser.add_argument('--debug', action='store_true', help='If set, stop after writing obj_points.npy (skip pairs/affordance).')
    parser.add_argument('--vis_pose_indices', type=str, default='0,1,2', help="Comma-separated pose indices for multi-pose preview (e.g., '0,10,50')")
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    run_all_in_one(
        object_name=args.object_name,
        result_path=args.result_path,
        output_root=args.output_root,
        sapien_mesh_root=args.sapien_mesh_root,
        k=args.k,
        base_n=args.base_n,
        x_shift=args.x_shift,
        device=device,
        outer_only=args.outer_only,
        sigma=args.sigma,
        visualize=args.visualize,
        log_every=int(args.log_every),
        vis_sample_k=int(args.vis_sample_k),
        debug=bool(args.debug),
        vis_pose_indices=str(args.vis_pose_indices),
    )


if __name__ == '__main__':
    main()

#python tools/sapien_convert_to_biman.py --in_root /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/sapien_data/bimanual-pose-data --out_root /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/sapien_data/converted_biman_format


#python /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/preprocess/preprocess.py --object_name 100025 --result_path /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/sapien_data/converted_biman_format --sapien_mesh_root /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/sapien_data/preprocessed_meshes --output_root /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/preprocess_data/pot_data --k 8192 --device cuda --outer_only --sigma 0.02 --visualize --vis_sample_k 0 --log_every 500 --debug
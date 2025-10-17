import os
import sys
import argparse
import numpy as np
import torch
import trimesh as tm

# Allow importing project-local packages
CUR_DIR = os.path.dirname(__file__)
PROJ_ROOT = os.path.dirname(CUR_DIR)
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from preprocess.utils import (
    sample_object_points_outer_surface_voxel,
    sample_object_points_outer_surface,
    sample_object_points_with_trimesh,
)


def load_mesh(mesh_path: str) -> tm.Trimesh:
    """
    Load a mesh from file and return a single Trimesh instance.
    If the file contains a scene with multiple geometries, concatenate them.
    """
    mesh_or_scene = tm.load(mesh_path, force='mesh')
    if isinstance(mesh_or_scene, tm.Trimesh):
        return mesh_or_scene
    if isinstance(mesh_or_scene, tm.Scene):
        geos = list(mesh_or_scene.geometry.values())
        if len(geos) == 0:
            return tm.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))
        return tm.util.concatenate(geos)
    # Fallback: try to coerce
    return tm.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))


def main():
    parser = argparse.ArgumentParser(description='Convert a mesh to point cloud using voxel-shell sampling.')
    parser.add_argument('--mesh_path', type=str, required=True, help='Path to the input mesh file (e.g., .obj/.stl/.ply).')
    parser.add_argument('--out', type=str, default=None, help='Output .npy path to save Nx3 float32 point cloud.')
    parser.add_argument('--k', type=int, default=8192, help='Target number of points (match preprocess defaults).')
    parser.add_argument('--base_n', type=int, default=30000, help='Base sample count for plain surface sampling (match utils).')
    parser.add_argument('--pre_rand_n', type=int, default=12000, help='Random pre-downsample before FPS when needed.')
    parser.add_argument('--use_fps', action='store_true', default=True, help='Use FPS to downsample to k (match utils).')
    parser.add_argument('--resolution', type=int, default=96, help='Voxel resolution (higher -> finer).')
    parser.add_argument('--oversample_ratio', type=int, default=None, help='If None, computed as base_n // k (>=2).')
    parser.add_argument('--x_shift', type=float, default=0.0, help='Translate along x-axis before sampling.')
    parser.add_argument('--scale', type=float, default=1.0, help='Uniform scale to apply to the mesh (1.0 keeps original).')
    parser.add_argument('--device', type=str, default='cpu', help="'cpu' or 'cuda'")
    args = parser.parse_args()

    mesh = load_mesh(args.mesh_path)
    if mesh.is_empty:
        raise ValueError(f'Loaded empty mesh from {args.mesh_path}')

    # Prefer voxel hollow-shell sampling; fallback to normal-based outer surface; then to plain surface.
    # Compute oversample ratio like utils: max(2, base_n // k)
    over_ratio = int(args.oversample_ratio) if args.oversample_ratio is not None else max(2, int(args.base_n) // max(1, int(args.k)))

    pts = sample_object_points_outer_surface_voxel(
        mesh=mesh,
        target_n=int(args.k),
        scale=float(args.scale),
        x_shift=float(args.x_shift),
        device=str(args.device),
        resolution=int(args.resolution),
        oversample_ratio=over_ratio,
    )

    if pts is None or pts.shape[0] == 0:
        pts = sample_object_points_outer_surface(
            mesh=mesh,
            target_n=int(args.k),
            scale=float(args.scale),
            x_shift=float(args.x_shift),
            device=str(args.device),
            object_center=None,
            oversample_ratio=over_ratio,
        )

    if pts is None or pts.shape[0] == 0:
        pts = sample_object_points_with_trimesh(
            mesh=mesh,
            base_n=int(args.base_n),
            scale=float(args.scale),
            x_shift=float(args.x_shift),
            device=str(args.device),
        )

    # Match utils: optional random pre-downsample + FPS/_ensure_k to exactly k
    if bool(args.use_fps):
        if pts.shape[0] > int(args.pre_rand_n) > 0:
            idx = torch.randperm(pts.shape[0], device=pts.device)[: int(args.pre_rand_n)]
            pts = pts[idx]
        # Simple FPS from utils._fps logic (reimplemented here to avoid import)
        num_points = pts.shape[0]
        m = min(int(args.k), num_points)
        sampled_indices = torch.zeros(m, dtype=torch.long, device=pts.device)
        distances = torch.full((num_points,), float('inf'), device=pts.device, dtype=pts.dtype)
        farthest = torch.tensor(0, device=pts.device, dtype=torch.long)
        for i in range(m):
            sampled_indices[i] = farthest
            centroid = pts[farthest].unsqueeze(0)
            dist = torch.sum((pts - centroid) ** 2, dim=1)
            distances = torch.minimum(distances, dist)
            farthest = torch.argmax(distances)
        pts = pts[sampled_indices]
    else:
        n = pts.shape[0]
        if n >= int(args.k):
            idx = torch.randperm(n, device=pts.device)[: int(args.k)]
            pts = pts[idx]
        else:
            repeat = (int(args.k) + n - 1) // n
            pts = pts.repeat(repeat, 1)[: int(args.k)]

    pts_np = pts.detach().cpu().numpy().astype(np.float32)

    out_path = args.out
    if out_path is None:
        base, _ = os.path.splitext(args.mesh_path)
        out_path = f"{base}_pc_voxel_{args.k}.npy"
    out_dir = os.path.dirname(out_path)
    if len(out_dir) > 0:
        os.makedirs(out_dir, exist_ok=True)
    np.save(out_path, pts_np)
    print(f'Saved point cloud to: {out_path}  (N={pts_np.shape[0]})')


if __name__ == '__main__':
    main()



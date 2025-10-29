# python /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/sapien_data/sapien_data_to_watertight_mesh.py   --input_dir /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/sapien_data/partnet-mobility-dataset/3386   --output /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/sapien_data/partnet-mobility-dataset/3386/watertight/mesh_wt.obj   --resolution 256

# python /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/tools/check_and_render_mesh.py --mesh /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/sapien_data/partnet-mobility-dataset/100015/watertight/mesh_wt.obj --out /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/sapien_data/partnet-mobility-dataset/100015/watertight/preview.png --html /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/sapien_data/partnet-mobility-dataset/100015/watertight/preview.html

import os
import sys
import argparse
from typing import List, Optional, Tuple

import numpy as np
import trimesh as tm


def _find_urdf_file(root_dir: str) -> Optional[str]:
    """
    Recursively search for a URDF file under the given directory.
    Prefer common name 'mobility.urdf' if present, otherwise return the first .urdf found.
    """
    mob = None
    first = None
    for cur, _dirs, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith('.urdf'):
                full = os.path.join(cur, f)
                if f == 'mobility.urdf' or f == 'model.urdf':
                    mob = full
                if first is None:
                    first = full
        # Fast-exit if preferred found in this dir
        if mob is not None:
            return mob
    return mob if mob is not None else first


def _load_mesh_from_file(path: str) -> Optional[tm.Trimesh]:
    """
    Load a mesh from a file robustly. Returns a single Trimesh or None on failure.
    """
    try:
        if not os.path.isfile(path):
            return None
        m = tm.load(path, force='mesh')
        if isinstance(m, tm.Trimesh) and (not m.is_empty):
            return m
        return None
    except Exception:
        return None


def _collect_meshes_from_urdf(urdf_path: str) -> List[tm.Trimesh]:
    """
    Collect meshes from a URDF using urdfpy if available, including link/visual transforms and per-visual scales.
    Falls back to an empty list if urdfpy is not installed or parsing fails.
    """
    meshes: List[tm.Trimesh] = []
    try:
        from urdfpy import URDF
    except Exception:
        return meshes

    try:
        urdf = URDF.load(urdf_path)
        link_to_T = urdf.link_fk(cfg={})  # default configuration (all zeros)
        base_dir = os.path.dirname(urdf_path)

        for link in urdf.links:
            T_link = link_to_T.get(link, np.eye(4))
            if getattr(link, 'visuals', None) is None:
                continue
            for vis in link.visuals:
                geom = getattr(vis, 'geometry', None)
                if geom is None:
                    continue
                # Handle mesh geometry
                mesh_attr = getattr(geom, 'mesh', None)
                if mesh_attr is not None:
                    filenames: List[str] = []
                    # urdfpy may expose either 'filename' or 'filenames'
                    fname = getattr(mesh_attr, 'filename', None)
                    fnames = getattr(mesh_attr, 'filenames', None)
                    if isinstance(fnames, (list, tuple)) and len(fnames) > 0:
                        filenames.extend(list(fnames))
                    elif isinstance(fname, str) and len(fname) > 0:
                        filenames.append(fname)
                    # scale may be a 3-vector or None
                    scale = getattr(mesh_attr, 'scale', None)
                    if scale is not None:
                        try:
                            scale = np.asarray(scale, dtype=float).reshape(3)
                        except Exception:
                            scale = None

                    # visual origin transform
                    T_vis = np.eye(4)
                    try:
                        origin = getattr(vis, 'origin', None)
                        if origin is not None:
                            T_vis = origin.matrix  # urdfpy.URDFPose -> 4x4
                    except Exception:
                        T_vis = np.eye(4)

                    T = (T_link @ T_vis).astype(float)
                    for rel in filenames:
                        src = rel
                        if not os.path.isabs(src):
                            src = os.path.join(base_dir, rel)
                        mm = _load_mesh_from_file(src)
                        if mm is None:
                            continue
                        if scale is not None:
                            try:
                                mm.apply_scale(scale)
                            except Exception:
                                pass
                        try:
                            mm.apply_transform(T)
                        except Exception:
                            pass
                        meshes.append(mm)
                # TODO: optionally handle primitive geometry (box/cylinder/sphere) if needed
    except Exception:
        # Parsing failure: return what we collected so far (likely empty)
        return meshes

    return meshes


def _collect_meshes_by_scanning(root_dir: str) -> List[tm.Trimesh]:
    """
    Fallback: recursively scan common mesh file types and load them all as absolute meshes.
    This ignores URDF transforms but still merges the geometry, which is often acceptable
    when individual part files are already positioned in a consistent frame.
    """
    exts = {'.obj', '.stl', '.ply', '.glb', '.gltf', '.dae'}
    skip_dirs = {'texture', 'textures', 'materials', 'images'}
    meshes: List[tm.Trimesh] = []
    for cur, dirs, files in os.walk(root_dir):
        # prune common texture/material folders
        dirs[:] = [d for d in dirs if d.lower() not in skip_dirs]
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                p = os.path.join(cur, f)
                mm = _load_mesh_from_file(p)
                if mm is not None and not mm.is_empty:
                    meshes.append(mm)
    return meshes


def _merge_meshes(meshes: List[tm.Trimesh]) -> tm.Trimesh:
    """
    Merge a list of Trimesh into a single Trimesh with basic cleanup.
    """
    if len(meshes) == 0:
        raise ValueError("No meshes to merge.")
    merged = tm.util.concatenate(meshes)
    # Basic cleanup before voxelization
    try:
        # Prefer modern API to avoid deprecation warnings
        faces_keep = getattr(merged, 'nondegenerate_faces', None)
        if callable(faces_keep):
            merged.update_faces(faces_keep())
        else:
            merged.remove_degenerate_faces()
    except Exception:
        pass
    try:
        merged.remove_unreferenced_vertices()
    except Exception:
        pass
    try:
        merged.merge_vertices()
    except Exception:
        pass
    return merged


def _to_watertight(mesh: tm.Trimesh, resolution: int = 256) -> tm.Trimesh:
    """
    Convert an arbitrary mesh to a watertight mesh by voxelization + marching cubes.
    The output is inherently closed as an isosurface of a solid occupancy grid.
    """
    if mesh.is_empty:
        raise ValueError("Input mesh is empty.")

    bounds = mesh.bounds
    diag = float(np.linalg.norm(bounds[1] - bounds[0]))
    if not np.isfinite(diag) or diag <= 0:
        diag = 1.0
    # pitch so the longest side ~ resolution cells
    longest = float(np.max(bounds[1] - bounds[0]))
    if not np.isfinite(longest) or longest <= 0:
        longest = diag
    pitch = max(longest / float(max(8, resolution)), 1e-4)

    # Voxelize and try to fill small holes before surface reconstruction
    vox = mesh.voxelized(pitch)
    try:
        vox = vox.fill(method='holes')
    except Exception:
        pass

    wt_mesh: Optional[tm.Trimesh] = None
    # Try API variants across trimesh versions
    try:
        m = getattr(vox, 'marching_cubes', None)
        wt_mesh = m() if callable(m) else (m if isinstance(m, tm.Trimesh) else None)
    except Exception:
        wt_mesh = None

    if wt_mesh is None:
        try:
            from trimesh.voxel.ops import matrix_to_marching_cubes
            wt_mesh = matrix_to_marching_cubes(vox.matrix, pitch=vox.pitch, origin=vox.origin)
        except Exception:
            wt_mesh = None

    if wt_mesh is None or wt_mesh.is_empty:
        # Last resort: convex hull (guaranteed watertight but loses concavities)
        wt_mesh = mesh.convex_hull

    # Final cleanup
    try:
        faces_keep = getattr(wt_mesh, 'nondegenerate_faces', None)
        if callable(faces_keep):
            wt_mesh.update_faces(faces_keep())
        else:
            wt_mesh.remove_degenerate_faces()
        wt_mesh.remove_unreferenced_vertices()
        wt_mesh.merge_vertices()
    except Exception:
        pass
    return wt_mesh


def _try_subdivide(mesh: tm.Trimesh, target_edge_length: Optional[float] = None, iterations: int = 0) -> tm.Trimesh:
    """
    Optionally refine mesh by subdividing to increase vertex density which helps smoothing.
    Tries to use trimesh.remesh.subdivide_to_size when available, otherwise falls back
    to uniform subdivision of all faces for a small number of iterations.
    """
    try:
        import trimesh.remesh as remesh  # local import to avoid hard dependency at import-time
    except Exception:
        return mesh

    refined = mesh
    # Prefer target-edge-length driven remesh if requested and available
    if target_edge_length is not None:
        try:
            fn = getattr(remesh, 'subdivide_to_size', None)
            if callable(fn):
                refined = fn(refined, max_edge=target_edge_length)  # type: ignore[arg-type]
        except Exception:
            pass

    # Optional fixed iteration uniform subdivision (mild)
    if iterations and iterations > 0:
        for _ in range(int(iterations)):
            try:
                fn2 = getattr(remesh, 'subdivide', None)
                if callable(fn2):
                    new_vertices, new_faces = fn2(refined.vertices, refined.faces)
                    refined = tm.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
                else:
                    break
            except Exception:
                break

    return refined


def _try_smooth_in_place(mesh: tm.Trimesh, iterations: int = 0, taubin_lambda: float = 0.5, taubin_nu: float = -0.53) -> None:
    """
    Apply in-place smoothing if requested. Prefer Taubin (low-shrinkage). Fallback to
    Humphrey or Laplacian if Taubin is unavailable in this trimesh version.
    """
    if iterations is None or int(iterations) <= 0:
        return
    try:
        from trimesh import smoothing as tms
    except Exception:
        return

    iters = int(max(0, iterations))
    # Try Taubin first
    fn = getattr(tms, 'filter_taubin', None)
    if callable(fn):
        try:
            fn(mesh, lamb=float(taubin_lambda), nu=float(taubin_nu), iterations=iters)
            return
        except Exception:
            pass
    # Fallback: Humphrey's method
    fn = getattr(tms, 'filter_humphrey', None)
    if callable(fn):
        try:
            fn(mesh, alpha=0.1, beta=0.1, iterations=iters)
            return
        except Exception:
            pass
    # Last resort: simple Laplacian
    fn = getattr(tms, 'filter_laplacian', None)
    if callable(fn):
        try:
            fn(mesh, lamb=0.5, iterations=iters)
        except Exception:
            pass


def sapien_data_to_watertight_mesh(
    object_dir: str,
    output_mesh_path: str,
    resolution: int = 256,
    smooth_iterations: int = 0,
    taubin_lambda: float = 0.5,
    taubin_nu: float = -0.53,
    subdivide_iterations: int = 0,
    target_edge_length: Optional[float] = None,
) -> str:
    """
    Build a watertight mesh for a SAPIEN (PartNet-Mobility) object directory.

    Steps:
      1) Try to locate and parse a URDF (e.g., mobility.urdf); merge all visual meshes using link/visual transforms.
      2) If URDF is not available, fall back to scanning common mesh files and merge directly.
      3) Convert the merged mesh to a watertight mesh by voxelization + marching cubes.
      4) Export to output_mesh_path (format inferred by extension, e.g., .obj/.stl/.ply).

    Args:
        object_dir: Directory that contains a SAPIEN object (e.g., with mobility.urdf and meshes/ subfolders).
        output_mesh_path: Destination filepath. Parent directories will be created if needed.
        resolution: Target voxel resolution along the longest side (higher -> finer, slower).

    Returns:
        The output_mesh_path for convenience.
    """
    if not os.path.isdir(object_dir):
        raise NotADirectoryError(f"Not a directory: {object_dir}")

    # 1) Collect meshes via URDF if possible
    meshes: List[tm.Trimesh] = []
    urdf_file = _find_urdf_file(object_dir)
    if urdf_file is not None:
        meshes = _collect_meshes_from_urdf(urdf_file)

    # 2) Fallback to scanning when URDF parsing didn't yield anything
    if len(meshes) == 0:
        meshes = _collect_meshes_by_scanning(object_dir)

    if len(meshes) == 0:
        raise FileNotFoundError(
            f"No meshes found under {object_dir}. Ensure the object folder contains a URDF and meshes."
        )

    merged = _merge_meshes(meshes)
    wt = _to_watertight(merged, resolution=resolution)

    # Optional refinement and smoothing to improve surface quality while preserving watertightness
    original = wt.copy()
    try:
        refined = _try_subdivide(wt, target_edge_length=target_edge_length, iterations=subdivide_iterations)
        if refined is not wt:
            wt = refined
        _try_smooth_in_place(wt, iterations=smooth_iterations, taubin_lambda=taubin_lambda, taubin_nu=taubin_nu)
        # Light cleanup
        try:
            faces_keep = getattr(wt, 'nondegenerate_faces', None)
            if callable(faces_keep):
                wt.update_faces(faces_keep())
            else:
                wt.remove_degenerate_faces()
            wt.remove_unreferenced_vertices()
            wt.merge_vertices()
        except Exception:
            pass
        # Ensure watertight; if broken, revert to original marching-cubes output
        if not bool(wt.is_watertight):
            wt = original
    except Exception:
        wt = original

    os.makedirs(os.path.dirname(os.path.abspath(output_mesh_path)), exist_ok=True)
    wt.export(output_mesh_path)
    return output_mesh_path


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert a SAPIEN object directory into a single watertight mesh.')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the SAPIEN object directory.')
    parser.add_argument('--output', type=str, required=True, help='Output mesh path (.obj/.stl/.ply).')
    parser.add_argument('--resolution', type=int, default=256, help='Voxel resolution along the longest side.')
    parser.add_argument('--smooth-iters', type=int, default=10, help='Surface smoothing iterations (0 to disable).')
    parser.add_argument('--taubin-lambda', type=float, default=0.5, help='Taubin smoothing lambda (passband).')
    parser.add_argument('--taubin-nu', type=float, default=-0.53, help='Taubin smoothing nu (stopband).')
    parser.add_argument('--subdivide-iters', type=int, default=0, help='Uniform subdivision iterations (0 to disable).')
    parser.add_argument('--target-edge-length', type=float, default=None, help='Optional target max edge length for remeshing.')
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    out_path = sapien_data_to_watertight_mesh(
        args.input_dir,
        args.output,
        resolution=args.resolution,
        smooth_iterations=int(getattr(args, 'smooth_iters', 10)),
        taubin_lambda=float(getattr(args, 'taubin_lambda', 0.5)),
        taubin_nu=float(getattr(args, 'taubin_nu', -0.53)),
        subdivide_iterations=int(getattr(args, 'subdivide_iters', 0)),
        target_edge_length=(None if getattr(args, 'target_edge_length', None) in (None, "None", "") else float(args.target_edge_length)),
    )
    print(f"[OK] Watertight mesh saved to: {out_path}")


if __name__ == '__main__':
    main(sys.argv[1:])



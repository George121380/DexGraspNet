#!/usr/bin/env python3
"""
Convert SAPIEN/PartNet-Mobility URDF meshes to watertight meshes.

Features:
- Per-link processing that preserves kinematic structure (exports per-link meshes and updates a new URDF)
- Optional merge-all-links mode that merges every link into one mesh and outputs a single watertight mesh
- Basic geometric repair with optional PyMeshFix, fallback voxel reconstruction via marching cubes
- QA report with key metrics

Notes:
- Comments are in English per user request.
- This tool expects `mobility.urdf` inside --obj-dir.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
import tempfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np

# Third-party libs (soft dependencies). We attempt to import and degrade gracefully.
try:
    from urdfpy import URDF
except Exception:  # pragma: no cover
    URDF = None  # type: ignore

try:
    import trimesh
    from trimesh.transformations import translation_matrix
except Exception as exc:  # pragma: no cover
    print("ERROR: trimesh is required: pip install trimesh", file=sys.stderr)
    raise

try:
    import pymeshfix  # type: ignore
except Exception:  # pragma: no cover
    pymeshfix = None

try:
    import open3d as o3d  # type: ignore
except Exception:  # pragma: no cover
    o3d = None


@dataclass
class QARecord:
    link_name: str
    mode: str  # 'per-link' or 'merged'
    is_watertight: Optional[bool]
    num_vertices: int
    num_faces: int
    num_components: int
    volume: Optional[float]
    bbox_diag: float
    notes: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="URDF -> Watertight mesh converter")
    parser.add_argument("--obj-dir", required=True, help="Directory containing mobility.urdf and meshes")
    parser.add_argument("--out-dir", required=True, help="Output directory to write watertight meshes")
    parser.add_argument("--voxel-size", type=float, default=0.002, help="Voxel size in meters for reconstruction")
    parser.add_argument("--simplify-ratio", type=float, default=0.5, help="Target ratio for mesh simplification (if available)")
    parser.add_argument("--min-component-tris", type=int, default=100, help="Remove connected components smaller than this face count")
    parser.add_argument("--merge-all-links", type=str, default="false", help="true/false: merge all links and output a single mesh")
    parser.add_argument("--update-urdf", action="store_true", help="When per-link mode, also write mobility_watertight.urdf")
    args = parser.parse_args()

    val = args.__dict__.get("merge_all_links", "false")
    args.merge_all_links = str(val).lower() in {"1", "true", "yes", "y"}
    return args


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_urdf_path(obj_dir: str) -> str:
    candidates = [
        os.path.join(obj_dir, "mobility.urdf"),
        os.path.join(obj_dir, "mobility_watertight.urdf"),  # fallback if user re-runs
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"URDF not found in {obj_dir}; expected mobility.urdf")


def sanitize_urdf_limits(src_urdf: str) -> str:
    """Sanitize URDF to satisfy strict parsers: ensure joint limit has effort/velocity.

    - Adds default 'effort' and 'velocity' to <limit> if missing (value: 1000.0).
    - Leaves lower/upper untouched (we avoid inventing kinematic bounds).
    - Writes a temporary sanitized URDF and returns its path.
    """
    tree = ET.parse(src_urdf)
    root = tree.getroot()

    # Iterate all joints and fix limit attributes if present
    for joint in root.findall('joint'):
        limit = joint.find('limit')
        if limit is None:
            continue
        if 'effort' not in limit.attrib:
            limit.set('effort', '1000.0')
        if 'velocity' not in limit.attrib:
            limit.set('velocity', '1000.0')

    # Write next to the original URDF so that relative mesh paths remain valid
    base_dir = os.path.dirname(src_urdf)
    base_name = os.path.splitext(os.path.basename(src_urdf))[0]
    dst_path = os.path.join(base_dir, f"{base_name}_sanitized.urdf")
    tree.write(dst_path)
    return dst_path


def load_trimesh_single(path: str) -> trimesh.Trimesh:
    """Load a single Trimesh from file; if Scene, concatenate geometries."""
    m = trimesh.load(path, force='mesh', process=False)
    if isinstance(m, trimesh.Scene):
        # Concatenate scene geometry into one mesh
        geoms = [g for g in m.geometry.values()]
        if not geoms:
            raise ValueError(f"No geometry in scene: {path}")
        return trimesh.util.concatenate(geoms)
    if not isinstance(m, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type from {path}: {type(m)}")
    return m


def apply_transform_and_scale(mesh: trimesh.Trimesh, transform: np.ndarray, scale: Optional[List[float]]) -> trimesh.Trimesh:
    """Apply local scale then rigid transform to a mesh and return a copy."""
    out = mesh.copy()
    if scale is not None:
        s = np.eye(4)
        s[0, 0], s[1, 1], s[2, 2] = scale[0], scale[1], scale[2]
        out.apply_transform(s)
    out.apply_transform(transform)
    return out


def try_basic_repair(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Perform basic in-memory repairs; attempt PyMeshFix if available."""
    m = mesh.copy()
    try:
        # Remove obvious defects first
        m.remove_duplicate_faces()
        m.remove_degenerate_faces()
        m.remove_infinite_values()
        m.remove_unreferenced_vertices()
        m.process(validate=True)
    except Exception:
        pass

    # Try PyMeshFix which is strong at closing holes and fixing non-manifoldness
    if pymeshfix is not None:
        try:
            fixer = pymeshfix.MeshFix(m.vertices.copy(), m.faces.copy())
            # Common options: join connected components, remove small bits
            fixer.repair(joincomp=True, remove_smallest_components=True)
            v, f = fixer.mesh
            if v is not None and len(v) > 0 and f is not None and len(f) > 0:
                m = trimesh.Trimesh(vertices=v, faces=f, process=True)
        except Exception:
            # Fall back silently
            pass

    # Finalize normals and attempt to fill tiny holes
    try:
        trimesh.repair.fix_normals(m)
    except Exception:
        pass
    try:
        # Fill tiny boundary holes (heuristic)
        trimesh.repair.fill_holes(m)
    except Exception:
        pass
    return m


def remove_small_components(mesh: trimesh.Trimesh, min_faces: int) -> trimesh.Trimesh:
    """Remove connected components smaller than min_faces (by face count)."""
    comps = mesh.split(only_watertight=False)
    comps_filtered = [c for c in comps if len(c.faces) >= min_faces]
    if not comps_filtered:
        # Keep the largest if all are small
        comps_filtered = [max(comps, key=lambda c: len(c.faces))]
    return trimesh.util.concatenate(comps_filtered)


def voxel_reconstruct(mesh: trimesh.Trimesh, voxel_size: float) -> trimesh.Trimesh:
    """Reconstruct a watertight mesh via voxelization + marching cubes.

    This is robust to self-intersections and open surfaces, but may lose fine detail.
    """
    # Ensure mesh is in a sane state (voxelizer expects finite values)
    m = mesh.copy()
    m.remove_infinite_values()
    m.remove_unreferenced_vertices()

    vg = m.voxelized(pitch=voxel_size)
    mat = vg.matrix.astype(np.uint8)
    # Derive origin from VoxelGrid transform: world position of index [0,0,0]
    try:
        origin = (vg.transform @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
    except Exception:
        origin = np.zeros(3, dtype=float)
    # marching cubes expects occupancy grid; use trimesh helper
    res = trimesh.voxel.ops.matrix_to_marching_cubes(mat, pitch=voxel_size)
    if isinstance(res, trimesh.Trimesh):
        recon = res
    else:
        v, f = res
        recon = trimesh.Trimesh(vertices=v, faces=f, process=True)
    # translate vertices to world using derived origin
    try:
        recon.apply_translation(origin)
    except Exception:
        pass
    return recon


def simplify_mesh(mesh: trimesh.Trimesh, simplify_ratio: float) -> trimesh.Trimesh:
    """Simplify mesh if Open3D is available; otherwise return input."""
    if o3d is None:
        return mesh
    try:
        target = max(16, int(len(mesh.faces) * max(0.0, min(1.0, simplify_ratio))))
        if target >= len(mesh.faces):
            return mesh
        o3 = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(mesh.vertices),
            triangles=o3d.utility.Vector3iVector(mesh.faces),
        )
        o3.compute_vertex_normals()
        o3s = o3.simplify_quadric_decimation(target)
        v = np.asarray(o3s.vertices)
        f = np.asarray(o3s.triangles)
        if len(v) > 0 and len(f) > 0:
            return trimesh.Trimesh(vertices=v, faces=f, process=True)
    except Exception:
        pass
    return mesh


def mesh_metrics(mesh: trimesh.Trimesh) -> Tuple[bool, int, int, int, Optional[float], float]:
    """Return key metrics: watertight, num_vertices, num_faces, num_components, volume, bbox_diag."""
    try:
        wt = bool(mesh.is_watertight)
    except Exception:
        wt = False
    try:
        vol = float(mesh.volume) if wt else None
    except Exception:
        vol = None
    comps = mesh.split(only_watertight=False)
    bbox = mesh.bounding_box_oriented.extents if mesh.vertices.size else np.array([0, 0, 0])
    bbox_diag = float(np.linalg.norm(bbox))
    return wt, int(len(mesh.vertices)), int(len(mesh.faces)), int(len(comps)), vol, bbox_diag


def concatenate_meshes(meshes: List[trimesh.Trimesh]) -> trimesh.Trimesh:
    geoms = [m for m in meshes if m is not None and len(m.vertices) > 0 and len(m.faces) > 0]
    if not geoms:
        return trimesh.Trimesh()
    if len(geoms) == 1:
        return geoms[0].copy()
    return trimesh.util.concatenate(geoms)


def get_link_world_transforms(model: URDF) -> Dict[str, np.ndarray]:
    """Compute world transforms for all links at zero configuration."""
    # Zero configuration (all joints at 0)
    cfg = {j.name: 0.0 for j in model.joints}
    tf_map = model.link_fk(cfg=cfg)
    # Map by link name for convenience
    out: Dict[str, np.ndarray] = {}
    for link, T in tf_map.items():
        out[link.name] = T
    return out


def load_link_meshes(model: URDF, obj_dir: str, link_name: str) -> List[trimesh.Trimesh]:
    """Load and transform all visual meshes for a link into world coordinates.

    If no visuals, fall back to collision geometries.
    """
    link = next(l for l in model.links if l.name == link_name)
    world_T = get_link_world_transforms(model)[link_name]

    meshes: List[trimesh.Trimesh] = []

    def _collect_from_visuals(visuals) -> None:
        for vis in visuals:
            geom = getattr(vis, 'geometry', None)
            if geom is None or getattr(geom, 'mesh', None) is None:
                continue
            mesh_attr = geom.mesh
            filename = mesh_attr.filename
            scale = mesh_attr.scale if mesh_attr.scale is not None else None
            local_T = vis.origin if getattr(vis, 'origin', None) is not None else np.eye(4)
            file_path = filename
            if not os.path.isabs(file_path):
                file_path = os.path.join(obj_dir, file_path)
            if not os.path.isfile(file_path):
                # Try to resolve relative to obj_dir anyway
                alt = os.path.join(obj_dir, os.path.basename(filename))
                if os.path.isfile(alt):
                    file_path = alt
                else:
                    print(f"WARN: mesh file not found: {filename}")
                    continue
            try:
                m = load_trimesh_single(file_path)
            except Exception as exc:
                print(f"WARN: failed to load {file_path}: {exc}")
                continue
            T = world_T @ local_T
            m = apply_transform_and_scale(m, T, scale)
            meshes.append(m)

    # Prefer visuals
    if getattr(link, 'visuals', None):
        _collect_from_visuals(link.visuals)
    # Fallback to collisions
    if not meshes and getattr(link, 'collisions', None):
        _collect_from_visuals(link.collisions)

    return meshes


def process_mesh(mesh: trimesh.Trimesh, voxel_size: float, simplify_ratio: float, min_component_tris: int) -> trimesh.Trimesh:
    """Repair the mesh; if still not watertight, fallback to voxel reconstruction; simplify and finalize."""
    # Remove small floating bits before repair to reduce workload
    if len(mesh.faces) > 0:
        mesh = remove_small_components(mesh, min_component_tris)
    mesh = try_basic_repair(mesh)
    wt, *_ = mesh_metrics(mesh)
    if not wt:
        mesh = voxel_reconstruct(mesh, voxel_size=voxel_size)
    # Post-process: remove tiny components again and simplify
    if len(mesh.faces) > 0:
        mesh = remove_small_components(mesh, min_component_tris)
    mesh = simplify_mesh(mesh, simplify_ratio=simplify_ratio)
    return mesh


def export_trimesh(mesh: trimesh.Trimesh, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    # Export as OBJ for portability
    mesh.export(path)


def write_qa(records: List[QARecord], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2, ensure_ascii=False)


def update_urdf_file(src_urdf: str, dst_urdf: str, per_link_visual_paths: Dict[str, str], per_link_collision_paths: Dict[str, str]) -> None:
    """Update URDF mesh filenames to new watertight ones and remove scale attributes."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(src_urdf)
    root = tree.getroot()

    def _update_mesh_nodes(parent_tag: str, mapping: Dict[str, str]):
        # Iterate all link elements
        for link in root.findall('link'):
            link_name = link.attrib.get('name', '')
            for parent in link.findall(parent_tag):
                # Update all geometries under this parent
                geom = parent.find('geometry')
                if geom is None:
                    continue
                mesh_node = geom.find('mesh')
                if mesh_node is None:
                    continue
                new_path = mapping.get(link_name)
                if new_path is None:
                    continue
                mesh_node.set('filename', new_path)
                # Remove scale attribute if present (now baked in)
                if 'scale' in mesh_node.attrib:
                    try:
                        del mesh_node.attrib['scale']
                    except Exception:
                        pass

    _update_mesh_nodes('visual', per_link_visual_paths)
    _update_mesh_nodes('collision', per_link_collision_paths)

    ensure_dir(os.path.dirname(dst_urdf))
    tree.write(dst_urdf)


def per_link_pipeline(model: URDF, obj_dir: str, out_dir: str, voxel_size: float, simplify_ratio: float, min_component_tris: int, update_urdf: bool) -> None:
    qa: List[QARecord] = []

    visual_out_dir = os.path.join(out_dir, 'visual')
    collision_out_dir = os.path.join(out_dir, 'collision')
    ensure_dir(visual_out_dir)
    ensure_dir(collision_out_dir)

    per_link_visual_paths: Dict[str, str] = {}
    per_link_collision_paths: Dict[str, str] = {}

    for link in model.links:
        # Load and merge all meshes for this link in world frame
        meshes = load_link_meshes(model, obj_dir, link.name)
        combined = concatenate_meshes(meshes)
        if len(combined.faces) == 0:
            notes = "no_faces"
            wt, nv, nf, nc, vol, diag = False, 0, 0, 0, None, 0.0
            qa.append(QARecord(link_name=link.name, mode='per-link', is_watertight=wt, num_vertices=nv, num_faces=nf, num_components=nc, volume=vol, bbox_diag=diag, notes=notes))
            continue

        processed = process_mesh(combined, voxel_size=voxel_size, simplify_ratio=simplify_ratio, min_component_tris=min_component_tris)

        wt, nv, nf, nc, vol, diag = mesh_metrics(processed)
        qa.append(QARecord(link_name=link.name, mode='per-link', is_watertight=wt, num_vertices=nv, num_faces=nf, num_components=nc, volume=vol, bbox_diag=diag))

        # Export paths are relative to obj_dir for URDF referencing
        visual_rel = os.path.join('watertight', 'visual', f'link_{link.name}.obj')
        coll_rel = os.path.join('watertight', 'collision', f'link_{link.name}.obj')
        visual_abs = os.path.join(obj_dir, visual_rel)
        coll_abs = os.path.join(obj_dir, coll_rel)

        export_trimesh(processed, visual_abs)
        export_trimesh(processed, coll_abs)

        per_link_visual_paths[link.name] = visual_rel
        per_link_collision_paths[link.name] = coll_rel

    # QA report
    qa_path = os.path.join(out_dir, 'qa.json')
    write_qa(qa, qa_path)

    if update_urdf:
        src_urdf = find_urdf_path(obj_dir)
        dst_urdf = os.path.join(obj_dir, 'mobility_watertight.urdf')
        update_urdf_file(src_urdf, dst_urdf, per_link_visual_paths, per_link_collision_paths)


def merged_pipeline(model: URDF, obj_dir: str, out_dir: str, voxel_size: float, simplify_ratio: float, min_component_tris: int) -> None:
    qa: List[QARecord] = []
    all_meshes: List[trimesh.Trimesh] = []

    # Collect all link meshes transformed to world
    for link in model.links:
        meshes = load_link_meshes(model, obj_dir, link.name)
        if meshes:
            all_meshes.extend(meshes)

    merged = concatenate_meshes(all_meshes)
    if len(merged.faces) == 0:
        raise RuntimeError("Merged mesh is empty; cannot proceed")

    processed = process_mesh(merged, voxel_size=voxel_size, simplify_ratio=simplify_ratio, min_component_tris=min_component_tris)
    wt, nv, nf, nc, vol, diag = mesh_metrics(processed)
    qa.append(QARecord(link_name='ALL', mode='merged', is_watertight=wt, num_vertices=nv, num_faces=nf, num_components=nc, volume=vol, bbox_diag=diag))

    # Export: both visual and collision write the same watertight mesh for simplicity
    merged_visual_abs = os.path.join(out_dir, 'merged_visual.obj')
    merged_collision_abs = os.path.join(out_dir, 'merged_collision.obj')
    export_trimesh(processed, merged_visual_abs)
    export_trimesh(processed, merged_collision_abs)

    qa_path = os.path.join(out_dir, 'qa.json')
    write_qa(qa, qa_path)


def main() -> None:
    args = parse_args()
    obj_dir = os.path.abspath(args.obj_dir)
    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)

    if URDF is None:
        print("ERROR: urdfpy is required: pip install urdfpy", file=sys.stderr)
        sys.exit(2)

    urdf_path = find_urdf_path(obj_dir)
    sanitized_path = sanitize_urdf_limits(urdf_path)
    try:
        model = URDF.load(sanitized_path)
    finally:
        # Best-effort cleanup of temporary file
        try:
            os.remove(sanitized_path)
        except Exception:
            pass

    if args.merge_all_links:
        merged_pipeline(model, obj_dir=obj_dir, out_dir=out_dir, voxel_size=args.voxel_size, simplify_ratio=args.simplify_ratio, min_component_tris=args.min_component_tris)
    else:
        per_link_pipeline(model, obj_dir=obj_dir, out_dir=out_dir, voxel_size=args.voxel_size, simplify_ratio=args.simplify_ratio, min_component_tris=args.min_component_tris, update_urdf=args.update_urdf)


if __name__ == "__main__":
    main()



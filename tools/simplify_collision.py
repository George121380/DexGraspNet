#!/usr/bin/env python3
"""
Simplify a collision mesh to a target number of faces using PyVista/VTK decimation.

Usage:
  python tools/simplify_collision.py \
    --in /path/to/merged_collision.obj \
    --out /path/to/merged_collision_300k.obj \
    --target-faces 300000 \
    --qa /path/to/qa_simplified.json \
    [--watertight-target]

Notes:
- Comments are in English by request.
- Requires pyvista + vtk (already installed as pymeshfix deps).
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import trimesh

try:
    import pyvista as pv  # type: ignore
except Exception as exc:  # pragma: no cover
    print("ERROR: pyvista is required for simplification", file=sys.stderr)
    raise


@dataclass
class QAShort:
    mesh: str
    is_watertight: bool
    num_vertices: int
    num_faces: int
    volume: Optional[float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simplify collision mesh")
    p.add_argument('--in', dest='src', required=True)
    p.add_argument('--out', dest='dst', required=True)
    p.add_argument('--target-faces', type=int, default=300_000)
    p.add_argument('--qa', dest='qa', required=False)
    p.add_argument('--watertight-target', action='store_true', help='Use voxel reconstruction to hit target faces and preserve watertight')
    return p.parse_args()


def trimesh_to_polydata(mesh: trimesh.Trimesh) -> pv.PolyData:
    v = mesh.vertices
    faces = mesh.faces
    # Build faces array with leading 3 for each triangle
    f = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(v, f)


def polydata_to_trimesh(poly: pv.PolyData) -> trimesh.Trimesh:
    v = np.asarray(poly.points)
    faces = np.asarray(poly.faces)
    if faces.ndim == 1:
        faces = faces.reshape(-1, 4)
    tri = faces[:, 1:4].copy()
    return trimesh.Trimesh(vertices=v, faces=tri, process=True)


def qa_of(mesh: trimesh.Trimesh, label: str) -> QAShort:
    wt = bool(mesh.is_watertight)
    vol = float(mesh.volume) if wt else None
    return QAShort(mesh=label, is_watertight=wt, num_vertices=int(len(mesh.vertices)), num_faces=int(len(mesh.faces)), volume=vol)


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.dst), exist_ok=True)

    src_mesh = trimesh.load(args.src, force='mesh', process=True)
    if isinstance(src_mesh, trimesh.Scene):
        src_mesh = trimesh.util.concatenate([g for g in src_mesh.geometry.values()])

    cur_faces = int(len(src_mesh.faces))
    target = int(max(1000, min(args.target_faces, max(1000, cur_faces - 1))))

    if args.watertight_target:
        # Find voxel pitch so that marching cubes returns ~ target faces
        bbox = src_mesh.bounding_box.extents
        max_dim = float(np.max(bbox)) if bbox.size else 1.0
        # Initial guess similar to 256 resolution
        pitch = max_dim / 256.0

        def recon_faces(p: float) -> trimesh.Trimesh:
            vg = src_mesh.voxelized(pitch=p)
            mat = vg.matrix.astype(np.uint8)
            try:
                origin = (vg.transform @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
            except Exception:
                origin = np.zeros(3, dtype=float)
            res = trimesh.voxel.ops.matrix_to_marching_cubes(mat, pitch=p)
            if isinstance(res, trimesh.Trimesh):
                m = res
            else:
                v, f = res
                m = trimesh.Trimesh(vertices=v, faces=f, process=True)
            try:
                m.apply_translation(origin)
            except Exception:
                pass
            return m

        best = None
        for _ in range(10):
            m = recon_faces(pitch)
            nf = int(len(m.faces))
            if best is None or abs(nf - target) < abs(len(best.faces) - target):
                best = m
            if 0.8 * target <= nf <= 1.2 * target:
                best = m
                break
            # Adjust pitch: more pitch -> fewer faces
            if nf > target:
                pitch *= 1.25
            else:
                pitch *= 0.85
        simp_mesh = best if best is not None else recon_faces(pitch)
    else:
        reduction = max(0.0, min(1.0, 1.0 - (target / float(cur_faces))))
        poly = trimesh_to_polydata(src_mesh)
        # Use VTK decimation via PyVista
        simplified = poly.decimate(target_reduction=reduction, volume_preservation=True)
        simp_mesh = polydata_to_trimesh(simplified)

    # Export OBJ
    simp_mesh.export(args.dst)

    # QA
    qa = qa_of(simp_mesh, os.path.basename(args.dst))
    print(json.dumps(asdict(qa), indent=2))
    if args.qa:
        with open(args.qa, 'w', encoding='utf-8') as f:
            json.dump(asdict(qa), f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()



#!/usr/bin/env python3
"""
Unified wrapper to convert a PartNet-Mobility/SAPIEN object (URDF + meshes)
into watertight meshes with a single command.

This CLI orchestrates the existing tools:
- tools/urdf_to_watertight.py
- tools/simplify_collision.py
- tools/check_and_render_mesh.py

Features:
- Mode selection: merged/per-link/auto
- Auto fallback on OOM or failure: merge raw OBJ then voxel reconstruct
- Optional simplification to target faces (with watertight preservation)
- Optional preview image rendering and QA manifest

Notes:
- Comments are in English.
- The wrapper calls existing tools via subprocess to avoid modifying them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class QARecord:
    mesh: str
    is_watertight: bool
    num_vertices: int
    num_faces: int
    volume: Optional[float]
    bbox_extents: Optional[list]


def run_subprocess(cmd: list[str], cwd: Optional[str] = None) -> int:
    env = os.environ.copy()
    # Reduce thread fan-out to limit memory usage peaks.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("TRIMESH_NO_NETWORK", "1")
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    return int(proc.returncode)


def is_oom_returncode(code: int) -> bool:
    # Common Linux OOM/sigkill signatures
    return code in {137, 134, 139, -9}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="URDF -> Watertight wrapper")
    p.add_argument("--obj-dir", required=True, help="Directory containing mobility.urdf and meshes")
    p.add_argument("--out-dir", required=False, help="Output dir (default: <obj-dir>/watertight)")
    p.add_argument("--mode", choices=["merged", "per-link", "auto"], default="auto")
    p.add_argument("--voxel-size", type=float, default=0.002)
    p.add_argument("--simplify-ratio", type=float, default=0.5)
    p.add_argument("--min-component-tris", type=int, default=100)
    p.add_argument("--target-faces", type=int, default=0, help="If >0, simplify to target faces")
    p.add_argument("--watertight-target", action="store_true", help="Preserve watertight in simplification")
    p.add_argument("--no-preview", action="store_true", help="Skip preview image rendering")
    p.add_argument("--force-fallback", action="store_true", help="Skip urdf tool and use fallback directly")
    p.add_argument("--fallback-resolution", type=int, default=64, help="Fallback voxel resolution along bbox max dim")
    p.add_argument("--preview-size", type=int, nargs=2, default=(1024, 768), help="Preview resolution (WxH)")
    return p.parse_args()


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def run_urdf_tool(obj_dir: str, out_dir: str, mode: str, voxel_size: float, simplify_ratio: float, min_component_tris: int) -> int:
    exe = sys.executable
    script = os.path.join(os.path.dirname(__file__), "urdf_to_watertight.py")
    base = [exe, script, "--obj-dir", obj_dir, "--out-dir", out_dir,
            "--voxel-size", str(voxel_size), "--simplify-ratio", str(simplify_ratio),
            "--min-component-tris", str(min_component_tris)]
    if mode == "merged":
        base += ["--merge-all-links", "true"]
    elif mode == "per-link":
        base += ["--update-urdf"]
    else:
        # auto: try merged first, then per-link if merged fails
        code1 = run_subprocess(base + ["--merge-all-links", "true"])
        if code1 == 0:
            return 0
        if is_oom_returncode(code1):
            return code1
        code2 = run_subprocess(base + ["--update-urdf"])  # per-link
        return code2
    return run_subprocess(base)


def fallback_merge_and_voxel(obj_dir: str, out_dir: str, resolution: int) -> tuple[str, QARecord]:
    import numpy as np
    import glob
    import trimesh

    textured = os.path.join(obj_dir, "textured_objs")
    objs = sorted(glob.glob(os.path.join(textured, "*.obj")))
    meshes = []
    for p in objs:
        try:
            m = trimesh.load(p, force="mesh", process=False)
            if isinstance(m, trimesh.Scene):
                geoms = [g for g in m.geometry.values()]
                if not geoms:
                    continue
                m = trimesh.util.concatenate(geoms)
            if isinstance(m, trimesh.Trimesh) and len(m.faces) > 0:
                meshes.append(m)
        except Exception:
            continue
    if not meshes:
        raise RuntimeError("No OBJ meshes found for fallback")
    merged = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]

    # Determine voxel pitch from bbox and requested resolution
    bbox = merged.bounding_box.extents
    max_dim = float(np.max(bbox)) if bbox.size else 1.0
    pitch = max_dim / max(8, int(resolution))

    vg = merged.voxelized(pitch=pitch)
    mat = vg.matrix.astype(np.uint8)
    try:
        origin = (vg.transform @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
    except Exception:
        origin = np.zeros(3, dtype=float)

    res = trimesh.voxel.ops.matrix_to_marching_cubes(mat, pitch=pitch)
    if isinstance(res, trimesh.Trimesh):
        recon = res
    else:
        v, f = res
        recon = trimesh.Trimesh(vertices=v, faces=f, process=True)
    try:
        recon.apply_translation(origin)
    except Exception:
        pass

    ensure_dir(out_dir)
    out_mesh = os.path.join(out_dir, "merged_collision.obj")
    recon.export(out_mesh)

    wt = bool(recon.is_watertight)
    num_v = int(len(recon.vertices))
    num_f = int(len(recon.faces))
    volume = float(recon.volume) if wt else None
    bbox_extents = recon.bounding_box_oriented.extents.tolist()
    qa = QARecord(mesh=os.path.basename(out_mesh), is_watertight=wt, num_vertices=num_v, num_faces=num_f,
                  volume=volume, bbox_extents=bbox_extents)
    return out_mesh, qa


def run_preview(mesh_path: str, out_png: str, size: tuple[int, int]) -> int:
    exe = sys.executable
    script = os.path.join(os.path.dirname(__file__), "check_and_render_mesh.py")
    cmd = [exe, script, "--mesh", mesh_path, "--out", out_png]
    # check_and_render_mesh uses fixed window size internally; size kept for future extension
    return run_subprocess(cmd)


def run_simplify(src_mesh: str, dst_mesh: str, target_faces: int, qa_path: Optional[str], watertight_target: bool) -> int:
    exe = sys.executable
    script = os.path.join(os.path.dirname(__file__), "simplify_collision.py")
    cmd = [exe, script, "--in", src_mesh, "--out", dst_mesh, "--target-faces", str(target_faces)]
    if qa_path:
        cmd += ["--qa", qa_path]
    if watertight_target:
        cmd += ["--watertight-target"]
    return run_subprocess(cmd)


def main() -> None:
    args = parse_args()
    obj_dir = os.path.abspath(args.obj_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(obj_dir, "watertight")
    ensure_dir(out_dir)

    final_mesh: Optional[str] = None
    qa_records: list[QARecord] = []

    if not args.force_fallback:
        rc = run_urdf_tool(
            obj_dir=obj_dir,
            out_dir=out_dir,
            mode=args.mode,
            voxel_size=args.voxel_size,
            simplify_ratio=args.simplify_ratio,
            min_component_tris=args.min_component_tris,
        )
        if rc == 0:
            # Prefer merged_collision.obj if exists, else attempt to pick one file
            merged_collision = os.path.join(out_dir, "merged_collision.obj")
            merged_visual = os.path.join(out_dir, "merged_visual.obj")
            if os.path.isfile(merged_collision):
                final_mesh = merged_collision
            elif os.path.isfile(merged_visual):
                final_mesh = merged_visual
        else:
            if is_oom_returncode(rc):
                print(f"INFO: urdf_to_watertight OOM/failure (rc={rc}), switching to fallback...", file=sys.stderr)
            else:
                print(f"WARN: urdf_to_watertight failed (rc={rc}), switching to fallback...", file=sys.stderr)

    if final_mesh is None:
        # Fallback path: merge raw OBJ and voxel reconstruct
        mesh_path, qa = fallback_merge_and_voxel(obj_dir=obj_dir, out_dir=out_dir, resolution=args.fallback_resolution)
        final_mesh = mesh_path
        qa_records.append(qa)

    # Optional simplification
    if args.target_faces and args.target_faces > 0:
        simp_out = os.path.join(out_dir, f"merged_collision_{args.target_faces}_wt.obj")
        qa_out = os.path.join(out_dir, f"qa_simplified_{args.target_faces}.json")
        rc = run_simplify(final_mesh, simp_out, args.target_faces, qa_out, watertight_target=bool(args.watertight_target))
        if rc == 0 and os.path.isfile(simp_out):
            final_mesh = simp_out

    # Optional preview
    if not args.no_preview and final_mesh and os.path.isfile(final_mesh):
        preview_png = os.path.join(out_dir, "preview.png")
        _ = run_preview(final_mesh, preview_png, size=tuple(args.preview_size))

    # Write wrapper QA if we generated any
    if qa_records:
        qa_path = os.path.join(out_dir, "qa_wrapper.json")
        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump([asdict(q) for q in qa_records], f, indent=2, ensure_ascii=False)

    if final_mesh:
        print(f"DONE: final mesh => {final_mesh}")
    else:
        print("ERROR: No mesh produced", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()




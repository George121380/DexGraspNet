import argparse
import os
import math
import json
from typing import List, Optional, Tuple

import numpy as np
import imageio.v2 as imageio
import trimesh
import pyrender

from baselines.vlm_baseline.utils.io import ensure_dir, write_json
from baselines.vlm_baseline.utils.camera import look_at, intrinsics_from_fov


MESH_EXTS = {".obj", ".ply", ".stl", ".glb", ".gltf"}


def find_mesh_path(object_dir: str) -> Optional[str]:
    for root, _, files in os.walk(object_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in MESH_EXTS:
                return os.path.join(root, f)
    return None


def orbit_poses(n_views: int, radius: float, elevation_deg: float = 30.0) -> List[np.ndarray]:
    poses = []
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    elev_rad = math.radians(elevation_deg)
    for i in range(n_views):
        az = 2.0 * math.pi * i / n_views
        eye = np.array([
            radius * math.cos(az) * math.cos(elev_rad),
            radius * math.sin(az) * math.cos(elev_rad),
            radius * math.sin(elev_rad),
        ], dtype=np.float32)
        poses.append(look_at(eye, target, up))
    return poses


def enumerate_objects(dataset_dir: str) -> List[Tuple[str, str]]:
    # Returns list of (object_name, object_dir)
    if os.path.isdir(dataset_dir):
        # Single-object mode if the directory itself contains a mesh
        mesh_here = find_mesh_path(dataset_dir)
        if mesh_here is not None:
            return [(os.path.basename(os.path.normpath(dataset_dir)), dataset_dir)]
        # Otherwise, treat children as objects
        names = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        names.sort()
        return [(n, os.path.join(dataset_dir, n)) for n in names]
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fov-deg", type=float, default=60.0)
    parser.add_argument("--radius-scale", type=float, default=2.0)
    parser.add_argument("--max-objects", type=int, default=0, help="0 means no limit")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    obj_list = enumerate_objects(args.dataset_dir)
    if args.max_objects > 0:
        obj_list = obj_list[: args.max_objects]

    for obj, obj_dir in obj_list:
        mesh_path = find_mesh_path(obj_dir)
        if mesh_path is None:
            continue

        out_obj = os.path.join(args.out_dir, obj)
        rgb_dir = os.path.join(out_obj, "rgb")
        depth_dir = os.path.join(out_obj, "depth")
        mask_dir = os.path.join(out_obj, "mask")
        ensure_dir(rgb_dir)
        ensure_dir(depth_dir)
        ensure_dir(mask_dir)

        mesh = trimesh.load(mesh_path, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh) and hasattr(mesh, 'dump'):
            mesh = trimesh.util.concatenate(tuple(geom for geom in mesh.dump()))
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        scale = 1.0 / max(1e-6, np.linalg.norm(mesh.extents))
        mesh.apply_scale(scale)

        scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.2, 0.2, 0.2])
        pm = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        scene.add(pm)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light)

        r = pyrender.OffscreenRenderer(viewport_width=args.width, viewport_height=args.height)
        fx, fy, cx, cy = intrinsics_from_fov(args.width, args.height, args.fov_deg)
        cam = pyrender.PerspectiveCamera(yfov=math.radians(args.fov_deg), aspect=args.width / args.height)

        radius = args.radius_scale
        poses = orbit_poses(args.views, radius)

        cameras_meta = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": args.width, "height": args.height, "views": []}

        for i, T_wc in enumerate(poses, start=1):
            node_cam = scene.add(cam, pose=T_wc)
            color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
            scene.remove_node(node_cam)
            rgb = color[:, :, :3]
            msk = (depth > 0).astype(np.uint8) * 255

            imageio.imwrite(os.path.join(rgb_dir, f"{i:04d}.png"), rgb)
            np.save(os.path.join(depth_dir, f"{i:04d}.npy"), depth)
            imageio.imwrite(os.path.join(mask_dir, f"{i:04d}.png"), msk)

            cameras_meta["views"].append({
                "view_id": i,
                "T_wc": T_wc.tolist(),
            })

        write_json(os.path.join(out_obj, "cameras.json"), cameras_meta)


if __name__ == "__main__":
    main()


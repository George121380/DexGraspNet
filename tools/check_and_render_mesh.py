#!/usr/bin/env python3
"""
Check if a mesh is watertight and render a quick preview image.

Usage:
  python tools/check_and_render_mesh.py --mesh /path/to/mesh.obj --out /path/to/out.png

Notes:
- Comments are in English by request.
- Rendering uses PyVista offscreen; falls back to a simple trimesh scene capture if PyVista is unavailable.
"""

import argparse
import os
import sys
import numpy as np

import trimesh

try:
    import pyvista as pv  # type: ignore
    PV_OK = True
except Exception:
    PV_OK = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--mesh', required=True)
    p.add_argument('--out', required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mesh = trimesh.load(args.mesh, force='mesh', process=True)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values()]
        if not geoms:
            print('Empty scene', file=sys.stderr)
            sys.exit(2)
        mesh = trimesh.util.concatenate(geoms)

    wt = bool(mesh.is_watertight)
    num_v = int(len(mesh.vertices))
    num_f = int(len(mesh.faces))
    volume = float(mesh.volume) if wt else float('nan')
    bbox = mesh.bounding_box_oriented.extents
    print(f'watertight={wt}, vertices={num_v}, faces={num_f}, volume={volume}, bbox_extents={bbox}')

    # Render preview offscreen
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if PV_OK:
        try:
            pv.start_xvfb()
        except Exception:
            pass
        pl = pv.Plotter(off_screen=True, window_size=(1024, 768))
        v = mesh.vertices
        f = np.hstack([np.full((len(mesh.faces), 1), 3, dtype=np.int64), mesh.faces]).ravel()
        surf = pv.PolyData(v, f)
        pl.add_mesh(surf, color='lightgray', smooth_shading=True)
        pl.add_axes()
        pl.set_background('white')
        pl.camera_position = 'yz'  # deterministic view
        pl.show(screenshot=args.out)
        print(f'preview saved to: {args.out}')
        return

    # Fallback: use trimesh scene save_image (may require pyglet)
    try:
        scene = mesh.scene()
        png = scene.save_image(resolution=(1024, 768), visible=False)
        with open(args.out, 'wb') as f:
            f.write(png)
        print(f'preview saved to: {args.out}')
    except Exception as exc:
        print(f'Failed to render preview: {exc}', file=sys.stderr)
        sys.exit(3)


if __name__ == '__main__':
    main()





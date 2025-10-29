#!/usr/bin/env python3
"""
Check if a mesh is watertight and render a quick preview image, optionally export an interactive HTML.

Usage:
  python tools/check_and_render_mesh.py --mesh /path/to/mesh.obj --out /path/to/out.png [--html /path/to/out.html]

Notes:
- Comments are in English by request.
- Rendering uses PyVista offscreen; falls back to a simple trimesh scene capture if PyVista is unavailable.
- HTML export prefers PyVista export_html; otherwise falls back to a minimal three.js viewer with embedded GLB.
"""

import argparse
import os
import sys
import numpy as np
import base64
from typing import Optional

import trimesh

try:
    import pyvista as pv  # type: ignore
    PV_OK = True
except Exception:
    PV_OK = False

try:
    import plotly.graph_objects as go  # type: ignore
    import plotly.io as pio  # type: ignore
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--mesh', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--html', required=False, default=None, help='Optional interactive HTML output path.')
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
        # Prefer Plotly for offline self-contained HTML if available
        if args.html and PLOTLY_OK:
            try:
                os.makedirs(os.path.dirname(args.html), exist_ok=True)
                vv = mesh.vertices
                ff = mesh.faces
                fig = go.Figure(data=[go.Mesh3d(x=vv[:,0], y=vv[:,1], z=vv[:,2], i=ff[:,0], j=ff[:,1], k=ff[:,2], color='lightgray', opacity=1.0, flatshading=True)])
                fig.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data'), margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='white')
                pio.write_html(fig, file=args.html, include_plotlyjs='inline', full_html=True, auto_open=False)
                print(f'html saved to: {args.html}')
                return
            except Exception:
                pass
        # If HTML requested, try PyVista export first; if it fails, fallback to three.js here
        if args.html:
            try:
                os.makedirs(os.path.dirname(args.html), exist_ok=True)
                pl.export_html(args.html)
                print(f'html saved to: {args.html}')
                return
            except Exception:
                # Fallback to three.js inline
                try:
                    scene = mesh.scene()
                    glb_bytes = scene.export(file_type='glb')
                    if isinstance(glb_bytes, str):
                        glb_bytes = glb_bytes.encode('utf-8')
                    b64 = base64.b64encode(glb_bytes).decode('ascii')
                    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\"/>
  <title>Mesh Preview</title>
  <style>
    html, body {{ margin:0; padding:0; height:100%; overflow:hidden; background:#ffffff; }}
    #c {{ width:100%; height:100%; display:block; }}
  </style>
  <script src=\"https://unpkg.com/three@0.158.0/build/three.min.js\"></script>
  <script src=\"https://unpkg.com/three@0.158.0/examples/js/controls/OrbitControls.js\"></script>
  <script src=\"https://unpkg.com/three@0.158.0/examples/js/loaders/GLTFLoader.js\"></script>
  <script>
    function init() {{
      const canvas = document.getElementById('c');
      const renderer = new THREE.WebGLRenderer({{canvas, antialias:true}});
      renderer.setSize(window.innerWidth, window.innerHeight);
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xffffff);
      const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.01, 1000);
      camera.position.set(0.8, 0.8, 0.8);
      const controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.target.set(0,0,0);
      controls.update();
      const light1 = new THREE.HemisphereLight(0xffffff, 0x888888, 1.0);
      scene.add(light1);
      const light2 = new THREE.DirectionalLight(0xffffff, 0.8);
      light2.position.set(1,1,1);
      scene.add(light2);
      const grid = new THREE.GridHelper(1, 10, 0x888888, 0xdddddd);
      grid.position.set(0, -0.5, 0);
      scene.add(grid);
      const loader = new THREE.GLTFLoader();
      const dataUri = 'data:model/gltf-binary;base64,{b64}';
      loader.load(dataUri, (gltf) => {{
        const obj = gltf.scene;
        scene.add(obj);
        const box = new THREE.Box3().setFromObject(obj);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const fitDist = maxDim / (2*Math.tan((Math.PI/180)*camera.fov/2));
        const dir = new THREE.Vector3(1,1,1).normalize();
        camera.position.copy(center.clone().add(dir.multiplyScalar(fitDist*1.5)));
        camera.near = Math.max(1e-3, fitDist/100);
        camera.far = Math.max(10, fitDist*100);
        camera.updateProjectionMatrix();
        controls.target.copy(center);
        controls.update();
      }});
      window.addEventListener('resize', () => {{
        camera.aspect = window.innerWidth/window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      }});
      function animate() {{
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
      }}
      animate();
    }}
    window.addEventListener('DOMContentLoaded', init);
  </script>
  </head>
  <body>
    <canvas id=\"c\"></canvas>
  </body>
</html>
"""
                    with open(args.html, 'w', encoding='utf-8') as f:
                        f.write(html)
                    print(f'html saved to: {args.html}')
                    return
                except Exception as exc:
                    print(f'Failed to export HTML: {exc}', file=sys.stderr)
                    # Even if HTML fails, preview is saved; exit gracefully
                    sys.exit(4)
        else:
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

    # If we reach here and HTML was requested, try Plotly first (offline self-contained)
    if args.html:
        if PLOTLY_OK:
            try:
                os.makedirs(os.path.dirname(args.html), exist_ok=True)
                vv = mesh.vertices
                ff = mesh.faces
                fig = go.Figure(data=[go.Mesh3d(x=vv[:,0], y=vv[:,1], z=vv[:,2], i=ff[:,0], j=ff[:,1], k=ff[:,2], color='lightgray', opacity=1.0, flatshading=True)])
                fig.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data'), margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='white')
                pio.write_html(fig, file=args.html, include_plotlyjs='inline', full_html=True, auto_open=False)
                print(f'html saved to: {args.html}')
                return
            except Exception:
                pass
        # Fallback: generate using three.js with embedded GLB (requires internet to fetch scripts)
        try:
            os.makedirs(os.path.dirname(args.html), exist_ok=True)
            scene = mesh.scene()
            glb_bytes = scene.export(file_type='glb')
            if isinstance(glb_bytes, str):
                glb_bytes = glb_bytes.encode('utf-8')
            b64 = base64.b64encode(glb_bytes).decode('ascii')
            html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\"/>
  <title>Mesh Preview</title>
  <style>
    html, body {{ margin:0; padding:0; height:100%; overflow:hidden; background:#ffffff; }}
    #c {{ width:100%; height:100%; display:block; }}
  </style>
  <script src=\"https://unpkg.com/three@0.158.0/build/three.min.js\"></script>
  <script src=\"https://unpkg.com/three@0.158.0/examples/js/controls/OrbitControls.js\"></script>
  <script src=\"https://unpkg.com/three@0.158.0/examples/js/loaders/GLTFLoader.js\"></script>
  <script>
    function init() {{
      const canvas = document.getElementById('c');
      const renderer = new THREE.WebGLRenderer({{canvas, antialias:true}});
      renderer.setSize(window.innerWidth, window.innerHeight);
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xffffff);
      const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.01, 1000);
      camera.position.set(0.8, 0.8, 0.8);
      const controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.target.set(0,0,0);
      controls.update();
      const light1 = new THREE.HemisphereLight(0xffffff, 0x888888, 1.0);
      scene.add(light1);
      const light2 = new THREE.DirectionalLight(0xffffff, 0.8);
      light2.position.set(1,1,1);
      scene.add(light2);
      const grid = new THREE.GridHelper(1, 10, 0x888888, 0xdddddd);
      grid.position.set(0, -0.5, 0);
      scene.add(grid);
      const loader = new THREE.GLTFLoader();
      const dataUri = 'data:model/gltf-binary;base64,{b64}';
      loader.load(dataUri, (gltf) => {{
        const obj = gltf.scene;
        scene.add(obj);
        const box = new THREE.Box3().setFromObject(obj);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const fitDist = maxDim / (2*Math.tan((Math.PI/180)*camera.fov/2));
        const dir = new THREE.Vector3(1,1,1).normalize();
        camera.position.copy(center.clone().add(dir.multiplyScalar(fitDist*1.5)));
        camera.near = Math.max(1e-3, fitDist/100);
        camera.far = Math.max(10, fitDist*100);
        camera.updateProjectionMatrix();
        controls.target.copy(center);
        controls.update();
      }});
      window.addEventListener('resize', () => {{
        camera.aspect = window.innerWidth/window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      }});
      function animate() {{
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
      }}
      animate();
    }}
    window.addEventListener('DOMContentLoaded', init);
  </script>
  </head>
  <body>
    <canvas id=\"c\"></canvas>
  </body>
</html>
"""
            with open(args.html, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f'html saved to: {args.html}')
        except Exception as exc:
            print(f'Failed to export HTML: {exc}', file=sys.stderr)
            sys.exit(4)


if __name__ == '__main__':
    main()





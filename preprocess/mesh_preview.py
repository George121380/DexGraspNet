import os
import sys
import argparse
import numpy as np
import trimesh as tm


def load_mesh_single(mesh_path: str) -> tm.Trimesh:
    mesh = tm.load(mesh_path)
    if isinstance(mesh, tm.Scene):
        # concatenate all geometries into a single mesh
        geos = []
        for name, geom in mesh.geometry.items():
            if isinstance(geom, tm.Trimesh):
                geos.append(geom)
        if len(geos) == 0:
            raise ValueError("No Trimesh geometry found in scene: " + mesh_path)
        mesh = tm.util.concatenate(geos)
    if not isinstance(mesh, tm.Trimesh):
        raise ValueError("Unsupported mesh type for: " + mesh_path)
    return mesh


def write_plotly_html(mesh: tm.Trimesh, out_html: str, title: str = "Mesh Preview"):
    import plotly.graph_objects as go
    v = mesh.vertices
    f = mesh.faces
    fig = go.Figure(data=[go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color='lightgray', opacity=1.0
    )])
    fig.update_layout(title=title, title_x=0.5, scene=dict(aspectmode='data'))
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(out_html)
    print("[mesh_preview] saved:", out_html)


def main():
    CUR_DIR = os.path.dirname(__file__)
    PROJ_ROOT = os.path.dirname(CUR_DIR)

    parser = argparse.ArgumentParser(description="Preview mesh (.obj) as Plotly HTML. Support single or batch mode.")
    parser.add_argument('--object_name', type=str, default=None, help='Object id/dir, e.g., 100017')
    parser.add_argument('--mesh_path', type=str, default=None, help='Direct mesh path to .obj; overrides object_name if set')
    parser.add_argument('--sapien_mesh_root', type=str, default=os.path.join(PROJ_ROOT, 'data', 'sapien_data', 'preprocessed_meshes'))
    parser.add_argument('--output_html', type=str, default=None, help='Output HTML path (single mode)')
    parser.add_argument('--in_root', type=str, default=None, help='Batch mode: root containing <OBJ>/coacd/decomposed.obj')
    parser.add_argument('--out_root', type=str, default=None, help='Batch mode: output root to write <OBJ>/mesh_preview.html and index page')
    args = parser.parse_args()

    # Batch mode
    if args.in_root is not None:
        in_root = args.in_root
        out_root = args.out_root if args.out_root is not None else os.path.join(PROJ_ROOT, 'data', 'mesh_previews')
        os.makedirs(out_root, exist_ok=True)
        entries = []
        for name in sorted(os.listdir(in_root)):
            obj_dir = os.path.join(in_root, name)
            mesh_path = os.path.join(obj_dir, 'coacd', 'decomposed.obj')
            if not os.path.isdir(obj_dir) or not os.path.isfile(mesh_path):
                continue
            try:
                mesh = load_mesh_single(mesh_path)
                out_html = os.path.join(out_root, name, 'mesh_preview.html')
                write_plotly_html(mesh, out_html, title=f"Mesh Preview - {name}")
                entries.append((name, out_html))
            except Exception as e:
                print(f"[mesh_preview] skip {name}: {e}")
                continue
        # write index
        index_path = os.path.join(out_root, 'index_mesh_preview.html')
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write('<html><head><meta charset="utf-8"><title>Mesh Preview Index</title></head><body>')
            f.write('<h2>Mesh Preview Index</h2><ul>')
            for name, path in entries:
                rel = os.path.relpath(path, out_root)
                f.write(f'<li><a href="{rel}" target="_blank">{name}</a></li>')
            f.write('</ul></body></html>')
        print('[mesh_preview] index saved:', index_path)
        return

    # Single mode
    if args.mesh_path is None:
        if args.object_name is None:
            raise ValueError('Either --mesh_path or --object_name must be provided (or use --in_root for batch).')
        mesh_path = os.path.join(args.sapien_mesh_root, args.object_name, 'coacd', 'decomposed.obj')
    else:
        mesh_path = args.mesh_path
    if args.output_html is None:
        raise ValueError('--output_html must be provided in single mode')

    if not os.path.isfile(mesh_path):
        raise FileNotFoundError('Mesh not found: ' + mesh_path)

    mesh = load_mesh_single(mesh_path)
    title = f"Mesh Preview - {os.path.basename(os.path.dirname(os.path.dirname(mesh_path)))}"
    write_plotly_html(mesh, args.output_html, title=title)


if __name__ == '__main__':
    main()



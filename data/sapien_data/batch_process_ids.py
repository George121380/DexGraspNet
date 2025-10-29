"""
Batch download SAPIEN PartNet-Mobility objects by IDs, convert to watertight meshes,
and render preview images (and optional HTML viewers).

Usage examples:
  conda run -n urdf2mesh python data/sapien_data/batch_process_ids.py \
    --ids 3386,3393,4094,4529,4533,4541,4542,4552,4562,4563,4564,4566,4571,4574,4576,4578,4586,4589,4590,4592,4594,4627,4628,4633,4681,4853,5050,5306,5477 \
    --dataset_root /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/sapien_data/partnet-mobility-dataset \
    --resolution 256 --html

  conda run -n urdf2mesh python data/sapien_data/batch_process_ids.py \
    --ids_file /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/sapien_data/obj_list.txt \
    --dataset_root /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/sapien_data/partnet-mobility-dataset

Notes:
- Comments are in English.
- No persistent temp files are created.
"""

import os
import sys
import argparse
import subprocess
from typing import List

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

try:
    import sapien  # type: ignore
    SAPIEN_OK = True
except Exception:
    SAPIEN_OK = False


def parse_ids(ids_str: str) -> List[int]:
    parts = ids_str.replace('\n', ' ').replace('\t', ' ').replace(',', ' ').split()
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            pass
    return sorted(list(set(out)))


def read_ids_file(path: str) -> List[int]:
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            try:
                out.append(int(s))
            except Exception:
                continue
    return sorted(list(set(out)))


def ensure_symlink(src_dir: str, dst_dir: str) -> str:
    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    if os.path.islink(dst_dir) or os.path.exists(dst_dir):
        return dst_dir
    os.symlink(src_dir, dst_dir)
    return dst_dir


def run_convert(object_dir: str, output_mesh_path: str, resolution_candidates: List[int]) -> str:
    from data.sapien_data.sapien_data_to_watertight_mesh import sapien_data_to_watertight_mesh
    last_err = None
    for r in resolution_candidates:
        try:
            return sapien_data_to_watertight_mesh(object_dir, output_mesh_path, resolution=r)
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err is not None else RuntimeError('Conversion failed for unknown reason')


def run_preview(mesh_path: str, out_png: str, out_html: str = None) -> None:
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tool = os.path.join(proj_root, 'tools', 'check_and_render_mesh.py')
    cmd = [sys.executable, tool, '--mesh', mesh_path, '--out', out_png]
    if out_html:
        cmd += ['--html', out_html]
    subprocess.run(cmd, check=True)


def process_one(object_id: int, dataset_root: str, token: str, res_list: List[int], with_html: bool) -> None:
    # Resolve object directory path
    dst_dir = os.path.join(dataset_root, str(object_id))
    if SAPIEN_OK and token:
        try:
            # Download object (returns the path to the URDF inside a local folder)
            urdf_path = sapien.asset.download_partnet_mobility(object_id, token)  # type: ignore
            dl_dir = os.path.dirname(urdf_path)
            try:
                dst_dir = ensure_symlink(dl_dir, dst_dir)
            except Exception:
                dst_dir = dl_dir
        except Exception:
            # If download fails, attempt to use dataset_root/<ID>
            pass
    if not os.path.isdir(dst_dir):
        raise FileNotFoundError(f"Object directory not found: {dst_dir} (and download unavailable)")

    wt_dir = os.path.join(dst_dir, 'watertight')
    os.makedirs(wt_dir, exist_ok=True)
    wt_mesh = os.path.join(wt_dir, 'mesh_wt.obj')
    png_path = os.path.join(wt_dir, 'preview.png')
    html_path = os.path.join(wt_dir, 'preview.html') if with_html else None

    if not os.path.isfile(wt_mesh):
        out = run_convert(dst_dir, wt_mesh, res_list)
        print(f"[OK] {object_id} watertight -> {out}")
    else:
        print(f"[SKIP] {object_id} watertight exists: {wt_mesh}")

    try:
        run_preview(wt_mesh, png_path, html_path)
        print(f"[OK] {object_id} preview -> {png_path}{' and html' if with_html else ''}")
    except Exception as e:
        print(f"[WARN] preview failed for {object_id}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', type=str, default=None, help='Comma/space separated IDs, e.g., "3386,3393 4094".')
    parser.add_argument('--ids_file', type=str, default=None, help='Text file with one ID per line.')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root for dataset objects (<root>/<ID>/...).')
    parser.add_argument('--resolution', type=int, default=256, help='Preferred voxel resolution (fallback: 128,64).')
    parser.add_argument('--token', type=str, default='', help='SAPIEN PartNet-Mobility token (overrides env SAPIEN_TOKEN).')
    parser.add_argument('--html', action='store_true', help='Also export interactive HTML.')
    args = parser.parse_args()

    token = args.token or os.environ.get('SAPIEN_TOKEN', '')
    if not token:
        # Best-effort: fallback to the token used in existing workflow, if present
        token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InBlaXFpNjIxQGJlcmtlbGV5LmVkdSIsImlwIjoiMTcyLjIwLjAuMSIsInByaXZpbGVnZSI6MSwiZmlsZU9ubHkiOnRydWUsImlhdCI6MTc1OTkyOTk3MywiZXhwIjoxNzkxNDY1OTczfQ.vWAY5J1HZ8bxqEsqwvkMiQXxBMhBXKjUkhxlzqRoqrk"

    ids: List[int] = []
    if args.ids:
        ids.extend(parse_ids(args.ids))
    if args.ids_file:
        ids.extend(read_ids_file(args.ids_file))
    ids = sorted(list(set(ids)))
    if not ids:
        print('No IDs specified. Use --ids or --ids_file.', file=sys.stderr)
        sys.exit(1)

    res_list = [int(args.resolution), 128, 64]
    for oid in ids:
        try:
            process_one(oid, args.dataset_root, token, res_list, args.html)
        except Exception as e:
            print(f"[ERR] {oid} failed: {e}")


if __name__ == '__main__':
    main()



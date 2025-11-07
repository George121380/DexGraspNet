#!/usr/bin/env python3
"""
Batch validator for bimanual grasp bundles.

This script recursively scans a root directory for .npy files (bimanual grasp bundles),
derives the object code from each file name (e.g., 100015.npy -> object_code 100015),
and runs validate_grasps.py in bimanual mode in chunks to avoid memory growth.

Usage example:
  python scripts/valiation_grasps_all.py \
    --experiments_root /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/experiments \
    --mesh_path /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/third_party/DexGraspNet/asset_process/data/meshdata_sapien \
    --result_root /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/third_party/DexGraspNet/data/dataset \
    --gpu 0 --chunk_size 500

Notes:
- Bimanual only. Single-hand paths are intentionally not handled here.
- This script executes validate_grasps.py with cwd set to the grasp_generation directory
  so its relative asset paths resolve correctly.
"""

import argparse
import os
import sys
import subprocess
from typing import List
import signal
import time

# Global interrupt/child tracking for clean Ctrl+C handling
_STOP_REQUESTED = False
_CHILD_PROC = None


def _on_signal(signum, frame):
    global _STOP_REQUESTED, _CHILD_PROC
    _STOP_REQUESTED = True
    try:
        if _CHILD_PROC is not None:
            # Terminate the whole child process group (Linux)
            try:
                os.killpg(os.getpgid(_CHILD_PROC.pid), signal.SIGTERM)
            except Exception:
                try:
                    _CHILD_PROC.terminate()
                except Exception:
                    pass
    finally:
        # Give the child a moment to exit gracefully
        time.sleep(0.2)

try:
    from tqdm import tqdm  # optional, for progress bar
except Exception:
    tqdm = None


def find_npy_files(root_dir: str) -> List[str]:
    """Recursively find all .npy files under root_dir."""
    npy_paths: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith('.npy'):
                npy_paths.append(os.path.join(dirpath, name))
    npy_paths.sort()
    return npy_paths


def get_object_code_from_filename(npy_path: str) -> str:
    """Derive object code from file name. Example: /a/b/100015.npy -> 100015."""
    base = os.path.basename(npy_path)
    return os.path.splitext(base)[0]


def count_pairs_in_bundle(npy_path: str) -> int:
    """Return number of pairs in a bimanual bundle (.npy with allow_pickle)."""
    import numpy as np  # local import to avoid global hard dependency at import-time
    data = np.load(npy_path, allow_pickle=True)
    try:
        return int(len(data))
    finally:
        # help GC immediately
        del data


def is_object_already_processed(result_root: str, object_code: str) -> bool:
    """Return True if output folder for this object already exists with results.

    We consider it processed if directory result_root/segments/bimanual/<object_code>
    exists and contains either stats.json or any chunk npy file.
    """
    segment_dir = os.path.join(result_root, 'segments', 'bimanual', object_code)
    if not os.path.isdir(segment_dir):
        return False
    try:
        entries = os.listdir(segment_dir)
    except Exception:
        return False
    if 'stats.json' in entries:
        return True
    for name in entries:
        if name.startswith(f"{object_code}_bimanual_chunk_") and name.endswith('.npy'):
            return True
    return False


def run_validation_chunks(
    grasp_bundle_path: str,
    object_code: str,
    mesh_path: str,
    result_root: str,
    gpu: int,
    chunk_size: int,
    extra_args: List[str],
) -> None:
    """Run validate_grasps.py in chunks for a single bundle file."""
    # Resolve working directory to ensure relative assets in validate_grasps.py work
    grasp_generation_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(grasp_generation_dir, 'scripts', 'validate_grasps.py')

    total_pairs = count_pairs_in_bundle(grasp_bundle_path)
    if total_pairs <= 0:
        print(f"[Skip] Empty bundle: {grasp_bundle_path}")
        return

    # Prepare chunk loop
    indices = list(range(0, total_pairs, chunk_size))
    progress_iter = tqdm(indices, desc=f"{object_code}") if tqdm else indices

    for start_idx in progress_iter:
        if _STOP_REQUESTED:
            break
        this_count = min(chunk_size, total_pairs - start_idx)
        cmd = [
            sys.executable, script_path,
            '--gpu', str(gpu),
            '--bimanual',
            '--grasp_file_bimanual', grasp_bundle_path,
            '--mesh_path', mesh_path,
            '--result_path', result_root,
            '--object_code', object_code,
            '--pair_start', str(start_idx),
            '--pair_count', str(this_count),
            # increase or tune if needed
            '--val_batch_bimanual', '512',
        ]
        if extra_args:
            cmd.extend(extra_args)

        # Inherit environment; ensure working directory for relative assets
        print(f"[Run] {object_code} chunk {start_idx}..{start_idx + this_count} -> validate_grasps.py")
        # Start child in its own process group so we can kill the whole group on Ctrl+C
        proc = subprocess.Popen(cmd, cwd=grasp_generation_dir, preexec_fn=os.setsid)
        try:
            global _CHILD_PROC
            _CHILD_PROC = proc
            rc = proc.wait()
        finally:
            _CHILD_PROC = None
        if _STOP_REQUESTED:
            break
        if rc != 0:
            print(f"[Warn] validate_grasps.py exited with code {rc} for chunk {start_idx}..{start_idx + this_count}")
            # continue to next chunk; do not abort whole batch


def main() -> None:
    parser = argparse.ArgumentParser(description='Batch validate all bimanual bundles under experiments root.')
    parser.add_argument('--experiments_root', type=str, required=True, help='Root directory containing .npy bundles (recursive).')
    parser.add_argument('--mesh_path', type=str, required=True, help='Path to mesh folders (supports <obj_code>/coacd and <category>/<obj_code>/coacd).')
    parser.add_argument('--result_root', type=str, required=True, help='Directory to store results and stats.json (validate_grasps.py result_path).')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use.')
    parser.add_argument('--chunk_size', type=int, default=1024, help='Number of pairs per validation chunk.')
    parser.add_argument('--max_files', type=int, help='Optional: limit number of .npy files to process.')
    parser.add_argument('--pass_args', type=str, nargs='*', default=[], help='Additional args to pass to validate_grasps.py (e.g., --gui)')
    args = parser.parse_args()

    # Register signal handlers for clean interruption
    try:
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
    except Exception:
        pass

    npy_files = find_npy_files(args.experiments_root)
    if args.max_files is not None:
        npy_files = npy_files[: max(0, int(args.max_files))]

    if not npy_files:
        print(f"No .npy files found under: {args.experiments_root}")
        return

    print(f"Found {len(npy_files)} bundle(s) under: {args.experiments_root}")

    iterable = tqdm(npy_files, desc='Bundles') if tqdm else npy_files
    for npy_path in iterable:
        if _STOP_REQUESTED:
            break
        object_code = get_object_code_from_filename(npy_path)
        # Skip object if already processed in result_root
        if is_object_already_processed(os.path.abspath(args.result_root), object_code):
            print(f"[Skip existing] {object_code} already has outputs under segments/bimanual/{object_code}")
            continue
        try:
            run_validation_chunks(
                grasp_bundle_path=os.path.abspath(npy_path),
                object_code=object_code,
                mesh_path=os.path.abspath(args.mesh_path),
                result_root=os.path.abspath(args.result_root),
                gpu=args.gpu,
                chunk_size=args.chunk_size,
                extra_args=args.pass_args,
            )
        except Exception as e:
            print(f"[Error] Failed on {npy_path}: {e}")
        if _STOP_REQUESTED:
            break


if __name__ == '__main__':
    main()



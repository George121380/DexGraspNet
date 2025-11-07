#!/usr/bin/env python3
"""
Merge bimanual chunk files into a single npy per object.

Given a root directory, this script recursively searches for files matching
"*_bimanual_chunk_<start>_<end>.npy" and merges them by object code.

The merged file name is:
  <object_code>_merged_<posenum>.npy
where <posenum> is the total number of entries after merging.

Usage examples:
  # Merge chunks inside a specific object directory
  python scripts/validaed_data_merge.py \
    --root /path/to/result_root/segments/bimanual/100015

  # Merge all objects under bimanual segments, writing outputs next to each object dir
  python scripts/validaed_data_merge.py \
    --root //media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/third_party/DexGraspNet/data/dataset/segments/bimanual/100058

  # Merge all objects and place outputs under a separate directory (replicates object dirs)
  python scripts/validaed_data_merge.py \
    --root /path/to/result_root/segments/bimanual \
    --output_root /path/to/result_root/merged
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np


CHUNK_PATTERN = re.compile(r"^(?P<object>.+?)_bimanual_chunk_(?P<start>\d+)_(?P<end>\d+)\.npy$")


def find_chunk_files(root: str) -> List[str]:
    """Recursively find chunk files under root."""
    matches: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith('.npy'):
                continue
            if CHUNK_PATTERN.match(name):
                matches.append(os.path.join(dirpath, name))
    matches.sort()
    return matches


def group_by_object(chunk_paths: List[str]) -> Dict[str, List[Tuple[int, int, str]]]:
    """Group chunk file paths by object code.

    Returns mapping: object_code -> list of (start, end, path), sorted by start.
    """
    groups: Dict[str, List[Tuple[int, int, str]]] = {}
    for p in chunk_paths:
        m = CHUNK_PATTERN.match(os.path.basename(p))
        if not m:
            continue
        obj = m.group('object')
        start = int(m.group('start'))
        end = int(m.group('end'))
        groups.setdefault(obj, []).append((start, end, p))
    for obj, items in groups.items():
        items.sort(key=lambda t: (t[0], t[1]))
    return groups


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def merge_object_chunks(object_code: str, chunks: List[Tuple[int, int, str]], output_dir: str, overwrite: bool) -> str:
    """Merge chunk npys for one object and write merged file.

    Returns the output file path.
    """
    # Load and concatenate
    merged: List[object] = []
    total = 0
    for start, end, path in chunks:
        data = np.load(path, allow_pickle=True)
        # Expect data to be a list-like of entries
        if isinstance(data, np.ndarray) and data.dtype == object:
            data_list = list(data)
        else:
            # Fallback: attempt to coerce into list
            try:
                data_list = list(data)
            except Exception:
                raise RuntimeError(f"Unsupported chunk format: {path}")
        merged.extend(data_list)
        total += len(data_list)

    # Build output path
    ensure_dir(output_dir)
    out_name = f"{object_code}_merged_{total}.npy"
    out_path = os.path.join(output_dir, out_name)

    if os.path.exists(out_path) and not overwrite:
        print(f"[Skip existing] {out_path}")
        return out_path

    np.save(out_path, merged, allow_pickle=True)
    print(f"[Merged] {object_code}: {len(chunks)} chunk(s) -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Merge bimanual chunk npy files into a single npy per object.')
    parser.add_argument('--root', type=str, required=True, help='Root directory containing *_bimanual_chunk_<start>_<end>.npy files (recursive).')
    parser.add_argument('--output_root', type=str, help='Optional root for merged outputs; defaults to the object chunk directory.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing merged files if present.')
    parser.add_argument('--max_objects', type=int, help='Optional limit on number of objects to merge.')
    args = parser.parse_args()

    chunk_paths = find_chunk_files(os.path.abspath(args.root))
    if not chunk_paths:
        print(f"No chunk files found under: {args.root}")
        return

    groups = group_by_object(chunk_paths)
    object_codes = sorted(groups.keys())
    if args.max_objects is not None:
        object_codes = object_codes[: max(0, int(args.max_objects))]

    print(f"Found {len(object_codes)} object(s) to merge under: {args.root}")

    for obj in object_codes:
        chunks = groups[obj]
        # Determine output directory
        if args.output_root:
            out_dir = os.path.join(os.path.abspath(args.output_root), obj)
        else:
            # Default: use the directory containing the chunks (assume all in same folder)
            # Fall back to the directory of the first chunk
            out_dir = os.path.dirname(chunks[0][2])
        try:
            merge_object_chunks(obj, chunks, out_dir, overwrite=args.overwrite)
        except Exception as e:
            print(f"[Error] Failed merging {obj}: {e}")


if __name__ == '__main__':
    main()




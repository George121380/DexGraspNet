#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copy valid pose files (.npy chunks) for seen objects into a classified folder.

Assumptions:
- Per-object directory under --bimanual-dir contains files like
  "<object>_bimanual_chunk_<k>_<n>.npy" and a stats.json.
- The user-provided example indicates chunk files are the desired "valid pose files".
- Missing objects are already excluded from the split, but code is robust to skips.

Inputs:
- --bimanual-dir   Root of bimanual objects (each object is a subfolder)
- --split-file     seen_unseen_split.json generated earlier
- --output-dir     Destination root; files stored under <output>/<category>/<object>/
- --glob-pattern   Glob for files to copy (default "*_bimanual_*_*.npy")

Notes:
- English comments only while coding (per user preference).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import json
from typing import Dict, Any, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy seen objects' .npy chunk files classified by category.")
    parser.add_argument("--bimanual-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--glob-pattern", type=str, default="*_bimanual_*_*.npy")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination files if exist")
    return parser.parse_args()


def load_split(split_path: Path) -> Dict[str, Any]:
    with split_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    split = load_split(args.split_file)
    by_category = split.get("by_category", {})

    total_objects = 0
    total_files_copied = 0
    missing_objects: List[str] = []

    for category, info in by_category.items():
        seen_list: List[str] = info.get("seen", [])
        for obj in seen_list:
            total_objects += 1
            src_dir = args.bimanual_dir / obj
            if not src_dir.exists() or not src_dir.is_dir():
                missing_objects.append(obj)
                print(f"[WARN] object directory not found, skip: {obj}")
                continue
            dst_dir = args.output_dir / category / obj
            dst_dir.mkdir(parents=True, exist_ok=True)
            npy_files = sorted(src_dir.glob(args.glob_pattern))
            if not npy_files:
                print(f"[WARN] no files matched for object: {obj}")
            for src in npy_files:
                dst = dst_dir / src.name
                if dst.exists() and not args.overwrite:
                    # skip to avoid duplicate work
                    continue
                shutil.copy2(src, dst)
                total_files_copied += 1

    print(f"Copied files: {total_files_copied} from {total_objects} seen objects into {args.output_dir}")
    if missing_objects:
        print(f"Missing object directories (skipped): {len(missing_objects)}")


if __name__ == "__main__":
    main()






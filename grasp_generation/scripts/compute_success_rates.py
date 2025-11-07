#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute success rates for each object and category for bimanual grasp data.

Success rate is defined as valid / count, aggregated across all chunks
inside each object's stats.json.

Inputs:
  - --bimanual-dir       Path to the bimanual objects root directory
  - --categories-file    Path to object_by_category.json
  - --output-json        Path to write consolidated results JSON

Outputs:
  - JSON with per-object and per-category success rates, and missing objects

Notes:
  - Objects may be numeric IDs or string names; both are supported.
  - Missing objects (no directory or no stats.json) are specially marked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime


def load_categories(categories_file: Path) -> Dict[str, List[str]]:
    """Load category-to-objects mapping, normalizing all object IDs to strings."""
    with categories_file.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    normalized: Dict[str, List[str]] = {}
    for category, objects in raw.items():
        # Normalize all object identifiers to strings for directory lookup
        normalized[category] = [str(obj) for obj in objects]
    return normalized


def compute_object_totals(stats_path: Path) -> Tuple[int, int]:
    """Sum count and valid across all chunk entries within stats.json.

    Returns:
        (total_count, total_valid)
    """
    with stats_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    total_count = 0
    total_valid = 0
    # Each key corresponds to a chunk file; the value is a dict with fields
    for chunk_meta in data.values():
        count = int(chunk_meta.get("count", 0))
        valid = int(chunk_meta.get("valid", 0))
        total_count += count
        total_valid += valid
    return total_count, total_valid


def safe_rate(valid: int, count: int) -> float:
    """Compute success rate valid/count with zero-division guard."""
    if count <= 0:
        return 0.0
    return float(valid) / float(count)


def compute_rates(
    bimanual_dir: Path,
    categories: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Compute per-object and per-category success metrics.

    Returns a dictionary suitable for JSON serialization.
    """
    by_object: Dict[str, Dict[str, Any]] = {}
    by_category: Dict[str, Dict[str, Any]] = {}
    missing_objects: List[Dict[str, str]] = []

    # First pass: compute per-object stats for all objects referenced by categories
    for category, objects in categories.items():
        for obj_name in objects:
            if obj_name in by_object:
                # Avoid duplicate work if objects are accidentally repeated across categories
                continue
            obj_dir = bimanual_dir / obj_name
            if not obj_dir.exists() or not obj_dir.is_dir():
                by_object[obj_name] = {"missing": True, "reason": "object directory not found"}
                missing_objects.append({"object": obj_name, "category": category, "reason": "object directory not found"})
                continue
            stats_path = obj_dir / "stats.json"
            if not stats_path.exists():
                by_object[obj_name] = {"missing": True, "reason": "stats.json not found"}
                missing_objects.append({"object": obj_name, "category": category, "reason": "stats.json not found"})
                continue
            try:
                total_count, total_valid = compute_object_totals(stats_path)
                by_object[obj_name] = {
                    "count": int(total_count),
                    "valid": int(total_valid),
                    "rate": safe_rate(total_valid, total_count),
                }
            except Exception as e:
                by_object[obj_name] = {"missing": True, "reason": f"failed to parse stats.json: {e}"}
                missing_objects.append({"object": obj_name, "category": category, "reason": "failed to parse stats.json"})

    # Second pass: per-category aggregation
    for category, objects in categories.items():
        cat_count = 0
        cat_valid = 0
        cat_missing = 0
        object_rates: Dict[str, Any] = {}
        for obj_name in objects:
            obj_info = by_object.get(obj_name)
            if not obj_info or obj_info.get("missing"):
                cat_missing += 1
                object_rates[obj_name] = None  # mark missing with None
                continue
            count = int(obj_info.get("count", 0))
            valid = int(obj_info.get("valid", 0))
            rate = float(obj_info.get("rate", safe_rate(valid, count)))
            cat_count += count
            cat_valid += valid
            object_rates[obj_name] = rate
        by_category[category] = {
            "count": int(cat_count),
            "valid": int(cat_valid),
            "rate": safe_rate(cat_valid, cat_count),
            "num_objects": len(objects),
            "num_missing": int(cat_missing),
            "object_rates": object_rates,
        }

    # Optional: include present-but-uncategorized objects (not required by prompt)
    # Skipped to keep output focused on requested objects.

    result = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "base_dir": str(bimanual_dir),
        "by_object": by_object,
        "by_category": by_category,
        "missing_objects": missing_objects,
    }
    return result


def write_json(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute success rates per object and category.")
    parser.add_argument(
        "--bimanual-dir",
        required=True,
        type=Path,
        help="Path to the bimanual directory containing object subfolders",
    )
    parser.add_argument(
        "--categories-file",
        required=True,
        type=Path,
        help="Path to object_by_category.json",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
        help="Path to write consolidated results JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = load_categories(args.categories_file)
    results = compute_rates(args.bimanual_dir, categories)
    write_json(args.output_json, results)

    # Print concise summary to stdout
    num_objects = len(results["by_object"]) if "by_object" in results else 0
    num_missing = len(results["missing_objects"]) if "missing_objects" in results else 0
    print(f"Computed success rates for {num_objects} objects; missing: {num_missing}.")
    for cat, info in results["by_category"].items():
        rate = info.get("rate", 0.0)
        print(f"Category {cat}: rate={rate:.4f}, objects={info.get('num_objects')}, missing={info.get('num_missing')}")


if __name__ == "__main__":
    main()



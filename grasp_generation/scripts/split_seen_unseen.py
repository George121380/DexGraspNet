#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create seen/unseen splits per category using a 2:1 ratio (seen:unseen).

Rules:
- Only consider objects present (exclude those marked missing in success rates).
- Target unseen fraction is 1/3. For each category with N present objects,
  unseen_count = round(N / 3). Edge cases:
  - If N < 3: unseen_count = 0 by default; optionally set to 1 for N==2 if desired.
  - You can override behavior with --min-unseen-if-two to force 1 unseen when N==2.
- To balance difficulty, we select unseen across the success rate spectrum by
  spacing picks on the sorted-by-rate list.

Inputs:
- --categories-file: object_by_category.json
- --rates-file: graspdata_bimanual_success_rates.json (from compute_success_rates.py)

Output:
- JSON mapping per category {seen: [...], unseen: [...], excluded_missing: [...]}.

Comments in English only, per user preference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime


def load_categories(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Normalize to strings
    return {k: [str(x) for x in v] for k, v in data.items()}


def load_rates(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def spaced_indices(n: int, k: int) -> List[int]:
    """Pick k indices spaced across [0, n-1] inclusive, deterministic.

    Example: n=10, k=3 -> near [2, 5, 7] depending on rounding.
    """
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))
    chosen = []
    used = set()
    for i in range(k):
        # Evenly spaced positions across the range
        pos = round((i + 1) * (n + 1) / (k + 1)) - 1
        pos = max(0, min(n - 1, pos))
        # Resolve collisions by shifting to nearest free spot
        if pos in used:
            left = pos - 1
            right = pos + 1
            picked = None
            while left >= 0 or right < n:
                if right < n and right not in used:
                    picked = right
                    break
                if left >= 0 and left not in used:
                    picked = left
                    break
                right += 1
                left -= 1
            pos = picked if picked is not None else pos
        used.add(pos)
        chosen.append(pos)
    chosen.sort()
    return chosen


def split_category(objects: List[str], by_object: Dict[str, Any], force_one_if_two: bool) -> Dict[str, Any]:
    present: List[Tuple[str, float]] = []
    excluded_missing: List[str] = []
    for obj in objects:
        info = by_object.get(obj)
        if not info or info.get("missing"):
            excluded_missing.append(obj)
            continue
        rate = float(info.get("rate", 0.0))
        present.append((obj, rate))

    present.sort(key=lambda x: x[1])  # sort by rate ascending
    n = len(present)
    if n == 0:
        return {
            "seen": [],
            "unseen": [],
            "excluded_missing": excluded_missing,
            "num_present": 0,
            "num_unseen": 0,
            "num_seen": 0,
        }

    # Unseen count using 1/3 fraction, rounded
    unseen = round(n / 3)
    if n < 3:
        unseen = 0
        if n == 2 and force_one_if_two:
            unseen = 1

    idxs = spaced_indices(n, unseen)
    unseen_objs = [present[i][0] for i in idxs]
    unseen_set = set(unseen_objs)
    seen_objs = [obj for obj, _ in present if obj not in unseen_set]

    return {
        "seen": seen_objs,
        "unseen": unseen_objs,
        "excluded_missing": excluded_missing,
        "num_present": n,
        "num_unseen": len(unseen_objs),
        "num_seen": len(seen_objs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 2:1 seen/unseen splits per category.")
    parser.add_argument("--categories-file", type=Path, required=True)
    parser.add_argument("--rates-file", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--min-unseen-if-two",
        action="store_true",
        help="If a category has exactly 2 present objects, force 1 unseen (default off)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    categories = load_categories(args.categories_file)
    rates = load_rates(args.rates_file)
    by_object = rates.get("by_object", {})

    result = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "ratio": {"seen": 2, "unseen": 1},
        "notes": "Missing objects (no data) are excluded from both seen and unseen.",
        "by_category": {},
    }

    for cat, objs in categories.items():
        result["by_category"][cat] = split_category(objs, by_object, args.min_unseen_if_two)

    out: Path = args.output_json
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Print summary
    for cat, info in result["by_category"].items():
        print(
            f"{cat}: present={info['num_present']}, unseen={info['num_unseen']}, seen={info['num_seen']}, excluded_missing={len(info['excluded_missing'])}"
        )


if __name__ == "__main__":
    main()






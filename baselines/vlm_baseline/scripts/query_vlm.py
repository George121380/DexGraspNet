import argparse
import os
import json
from typing import Dict, Any, List

import numpy as np
import imageio.v2 as imageio

from baselines.vlm_baseline.utils.io import ensure_dir, write_json


def select_representative_views(num_images: int, max_images: int) -> List[int]:
    if num_images <= 0:
        return []
    k = min(max_images, num_images)
    if k == 1:
        return [1]
    idxs = np.linspace(1, num_images, k, dtype=int).tolist()
    return sorted(set(idxs))


def find_foreground(mask: np.ndarray, u: int, v: int) -> (int, int):
    if mask[v, u] > 0:
        return u, v
    h, w = mask.shape
    for r in range(1, 15):
        for du in range(-r, r + 1):
            for dv in (-r, r):
                uu = np.clip(u + du, 0, w - 1)
                vv = np.clip(v + dv, 0, h - 1)
                if mask[vv, uu] > 0:
                    return uu, vv
        for dv in range(-r + 1, r):
            for du in (-r, r):
                uu = np.clip(u + du, 0, w - 1)
                vv = np.clip(v + dv, 0, h - 1)
                if mask[vv, uu] > 0:
                    return uu, vv
    return u, v


def mock_two_points(rgb: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    h, w = mask.shape
    u_left = int(w * 0.35)
    u_right = int(w * 0.65)
    v_mid = int(h * 0.5)
    u_left, v_left = find_foreground(mask, u_left, v_mid)
    u_right, v_right = find_foreground(mask, u_right, v_mid)
    return {
        "left": {"point_px": [int(u_left), int(v_left)], "confidence": 0.5},
        "right": {"point_px": [int(u_right), int(v_right)], "confidence": 0.5},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders", required=True)
    parser.add_argument("--model", default="gpt-5-vision")
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--two-points-only", action="store_true")
    parser.add_argument("--outputs-root", required=True)
    args = parser.parse_args()

    objects = [d for d in os.listdir(args.renders) if os.path.isdir(os.path.join(args.renders, d))]
    objects.sort()

    for obj in objects:
        obj_dir = os.path.join(args.renders, obj)
        cam_path = os.path.join(obj_dir, "cameras.json")
        rgb_dir = os.path.join(obj_dir, "rgb")
        mask_dir = os.path.join(obj_dir, "mask")
        if not os.path.isfile(cam_path) or not os.path.isdir(rgb_dir):
            continue
        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
        num_images = len(rgb_files)
        if num_images == 0:
            continue
        chosen_ids = select_representative_views(num_images, max_images=args.max_images)
        first_id = chosen_ids[0]
        rgb = imageio.imread(os.path.join(rgb_dir, f"{first_id:04d}.png"))
        mask = imageio.imread(os.path.join(mask_dir, f"{first_id:04d}.png")) if os.path.isdir(mask_dir) else np.ones(rgb.shape[:2], dtype=np.uint8) * 255

        two = mock_two_points(rgb, mask)

        if len(chosen_ids) >= 2:
            two["right"]["view_id"] = int(chosen_ids[1])
            two["left"]["view_id"] = int(first_id)
        else:
            two["right"]["view_id"] = int(first_id)
            two["left"]["view_id"] = int(first_id)

        out_dir = os.path.join(args.outputs_root, obj)
        ensure_dir(out_dir)
        payload = {"object_name": obj, "left": two["left"], "right": two["right"]}
        write_json(os.path.join(out_dir, "vlm_raw.json"), payload)


if __name__ == "__main__":
    main()


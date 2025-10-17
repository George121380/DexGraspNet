import argparse
import os
from typing import Dict, Any

import numpy as np
import imageio.v2 as imageio

from baselines.vlm_baseline.utils.io import read_json, write_json, ensure_dir


def overlay_points(rgb: np.ndarray, pts: Dict[str, Any]) -> np.ndarray:
    img = rgb.copy()
    colors = {"left": (255, 0, 0), "right": (0, 255, 0)}
    for k, c in colors.items():
        if k in pts and "point_px" in pts[k]:
            u, v = int(pts[k]["point_px"][0]), int(pts[k]["point_px"][1])
            for du in range(-3, 4):
                for dv in range(-3, 4):
                    uu = np.clip(u + du, 0, img.shape[1] - 1)
                    vv = np.clip(v + dv, 0, img.shape[0] - 1)
                    img[vv, uu, 0] = c[0]
                    img[vv, uu, 1] = c[1]
                    img[vv, uu, 2] = c[2]
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused-root", required=True)
    parser.add_argument("--renders", default=None)
    args = parser.parse_args()

    objects = [d for d in os.listdir(args.fused_root) if os.path.isdir(os.path.join(args.fused_root, d))]
    objects.sort()

    total = {"objects": 0, "left_world": 0, "right_world": 0}

    for obj in objects:
        fused_path = os.path.join(args.fused_root, obj, "fused_3d.json")
        if not os.path.isfile(fused_path):
            continue
        fused = read_json(fused_path)
        total["objects"] += 1
        if fused.get("left", {}).get("point_world") is not None:
            total["left_world"] += 1
        if fused.get("right", {}).get("point_world") is not None:
            total["right_world"] += 1

        if args.renders is not None:
            rgb_dir = os.path.join(args.renders, obj, "rgb")
            out_vis = os.path.join(args.fused_root, obj, "vis")
            ensure_dir(out_vis)
            for side in ("left", "right"):
                vid = fused.get(side, {}).get("view_id", 1)
                rgb_path = os.path.join(rgb_dir, f"{vid:04d}.png")
                if os.path.isfile(rgb_path):
                    rgb = imageio.imread(rgb_path)
                    over = overlay_points(rgb, {side: fused[side]})
                    imageio.imwrite(os.path.join(out_vis, f"overlay_{side}_{vid:04d}.png"), over)

    report = {
        "num_objects": total["objects"],
        "frac_left_world": (total["left_world"] / max(1, total["objects"])) ,
        "frac_right_world": (total["right_world"] / max(1, total["objects"])) ,
    }
    write_json(os.path.join(args.fused_root, "report.json"), report)


if __name__ == "__main__":
    main()


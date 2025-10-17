import argparse
import os
import json
from typing import Dict, Any

import numpy as np

from baselines.vlm_baseline.utils.io import read_json, write_json, ensure_dir


def backproject_point(u: int, v: int, depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    h, w = depth.shape
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))
    Z = float(depth[v, u])
    if Z <= 0.0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    return np.array([X, Y, Z], dtype=np.float32)


def transform_cam_to_world(p_c: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    p_h = np.ones(4, dtype=np.float32)
    p_h[:3] = p_c
    pw = T_wc @ p_h
    return pw[:3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders", required=True)
    parser.add_argument("--vlm-json-root", required=True)
    parser.add_argument("--fused-key", default="fused_3d.json")
    args = parser.parse_args()

    objects = [d for d in os.listdir(args.vlm_json_root) if os.path.isdir(os.path.join(args.vlm_json_root, d))]
    objects.sort()

    for obj in objects:
        vlm_path = os.path.join(args.vlm_json_root, obj, "vlm_raw.json")
        if not os.path.isfile(vlm_path):
            continue
        payload = read_json(vlm_path)

        cam_meta = read_json(os.path.join(args.renders, obj, "cameras.json"))
        fx, fy, cx, cy = cam_meta["fx"], cam_meta["fy"], cam_meta["cx"], cam_meta["cy"]
        views = {int(v["view_id"]): np.asarray(v["T_wc"], dtype=np.float32) for v in cam_meta["views"]}

        def process_one(side: str) -> Dict[str, Any]:
            entry = dict(payload[side])
            vid = int(entry.get("view_id", 1))
            u, v = int(entry["point_px"][0]), int(entry["point_px"][1])
            depth_path = os.path.join(args.renders, obj, "depth", f"{vid:04d}.npy")
            if os.path.isfile(depth_path):
                depth = np.load(depth_path)
                p_c = backproject_point(u, v, depth, fx, fy, cx, cy)
                if np.isfinite(p_c).all():
                    T_wc = views.get(vid)
                    if T_wc is not None:
                        p_w = transform_cam_to_world(p_c, T_wc)
                        entry["point_world"] = [float(p_w[0]), float(p_w[1]), float(p_w[2])]
                    else:
                        entry["point_world"] = None
                else:
                    entry["point_world"] = None
            else:
                entry["point_world"] = None
            return entry

        fused = {
            "object_name": payload.get("object_name", obj),
            "left": process_one("left"),
            "right": process_one("right"),
        }

        out_path = os.path.join(args.vlm_json_root, obj, args.fused_key)
        write_json(out_path, fused)


if __name__ == "__main__":
    main()


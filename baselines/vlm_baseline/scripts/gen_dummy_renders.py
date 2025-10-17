import argparse
import os
import math
import numpy as np
import imageio.v2 as imageio
from baselines.vlm_baseline.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--object", default="dummy_object")
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    args = parser.parse_args()

    obj_dir = os.path.join(args.out_dir, args.object)
    rgb_dir = os.path.join(obj_dir, "rgb")
    depth_dir = os.path.join(obj_dir, "depth")
    mask_dir = os.path.join(obj_dir, "mask")
    ensure_dir(rgb_dir)
    ensure_dir(depth_dir)
    ensure_dir(mask_dir)

    fx = args.width * 0.8
    fy = args.height * 0.8
    cx = args.width / 2.0
    cy = args.height / 2.0

    cams = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": args.width, "height": args.height, "views": []}

    for i in range(1, args.views + 1):
        # Simple RGB pattern per view
        img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        color = (int(50 + 40 * i), int(80 + 30 * i), int(100 + 20 * i))
        img[:, :] = color
        # Draw a rectangle to have some structure
        u0, v0 = args.width // 4, args.height // 4
        u1, v1 = 3 * args.width // 4, 3 * args.height // 4
        img[v0:v1, u0:u1, :] = (255, 255, 255)

        depth = np.ones((args.height, args.width), dtype=np.float32) * (1.0 + 0.05 * i)
        mask = np.ones((args.height, args.width), dtype=np.uint8) * 255

        imageio.imwrite(os.path.join(rgb_dir, f"{i:04d}.png"), img)
        np.save(os.path.join(depth_dir, f"{i:04d}.npy"), depth)
        imageio.imwrite(os.path.join(mask_dir, f"{i:04d}.png"), mask)

        # Simple T_wc: translate on a circle with fixed Z
        az = 2.0 * math.pi * (i - 1) / args.views
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = float(math.cos(az))
        T[1, 3] = float(math.sin(az))
        T[2, 3] = 1.0
        cams["views"].append({"view_id": i, "T_wc": T.tolist()})

    write_json(os.path.join(obj_dir, "cameras.json"), cams)


if __name__ == "__main__":
    main()

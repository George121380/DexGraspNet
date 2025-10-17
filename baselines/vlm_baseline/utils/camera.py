import math
import numpy as np
from typing import Tuple


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z = normalize(eye - target)
    x = normalize(np.cross(up, z))
    y = np.cross(z, x)
    T = np.eye(4, dtype=np.float32)
    T[:3, 0] = x
    T[:3, 1] = y
    T[:3, 2] = z
    T[:3, 3] = eye
    return T


def intrinsics_from_fov(width: int, height: int, fov_deg: float) -> Tuple[float, float, float, float]:
    fov_rad = math.radians(fov_deg)
    fx = 0.5 * width / math.tan(0.5 * fov_rad)
    fy = 0.5 * height / math.tan(0.5 * fov_rad)
    cx = width * 0.5
    cy = height * 0.5
    return fx, fy, cx, cy


from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import os
import numpy as np
import plotly.graph_objects as go


def write_pointcloud_with_values_html(
    points: np.ndarray,
    values: Optional[np.ndarray],
    keypoints: Optional[List[Tuple[int, np.ndarray]]] = None,
    out_path: str = "viz.html",
    title: str = "Point Cloud",
    point_size: float = 2.0,
    cmap: str = "Viridis",
) -> None:
    fig = go.Figure()
    if values is not None:
        c = values.reshape(-1)
    else:
        c = np.zeros((points.shape[0],), dtype=np.float32)

    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=point_size, color=c, colorscale=cmap, showscale=values is not None),
        name='points'
    ))

    if keypoints:
        for idx, kp in keypoints:
            fig.add_trace(go.Scatter3d(
                x=[kp[0]], y=[kp[1]], z=[kp[2]],
                mode='markers', marker=dict(size=6, color='red', symbol='circle'),
                name=f'kp_{idx}'
            ))

    fig.update_layout(scene=dict(aspectmode='data'), title=title, title_x=0.5)
    fig.write_html(out_path)


TRANSLATION_NAMES = ['WRJTx', 'WRJTy', 'WRJTz']
ROTATION_NAMES = ['WRJRx', 'WRJRy', 'WRJRz']
JOINT_NAMES = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
]
JOINT_ORDER = JOINT_NAMES + ROTATION_NAMES + TRANSLATION_NAMES


def vector_to_qpos_dict(vec: np.ndarray) -> Dict[str, float]:
    return {name: float(val) for name, val in zip(JOINT_ORDER, vec.tolist())}


@lru_cache(maxsize=None)
def load_default_scale(object_code: str, ref_scale_dir: str = '', ref_scale_file: str = '', ref_scale_value: float = None) -> float:
    # Priority 0: explicit numeric value from config (works even after file deletion)
    try:
        if ref_scale_value is not None:
            v = float(ref_scale_value)
            if v > 0:
                return v
    except Exception:
        pass
    # Priority 1: explicit file
    if ref_scale_file and os.path.exists(ref_scale_file):
        try:
            arr = np.load(ref_scale_file, allow_pickle=True)
            # Common cases: scalar dict, array of dicts, or np.void structured
            def _try_extract(x):
                try:
                    if isinstance(x, dict) and 'scale' in x:
                        return float(x['scale'])
                except Exception:
                    return None
                return None
            # scalar object
            if getattr(arr, 'shape', ()) == ():
                ent = arr.item() if hasattr(arr, 'item') else arr
                val = _try_extract(ent)
                if val is not None:
                    return val
            # flat iterate
            try:
                for ent in arr.flat:
                    ent = ent.item() if hasattr(ent, 'item') else ent
                    val = _try_extract(ent)
                    if val is not None:
                        return val
            except Exception:
                pass
        except Exception:
            pass
    # Priority 2: directory by object code
    if ref_scale_dir:
        path = os.path.join(ref_scale_dir, f"{object_code}_GT.npy")
        if os.path.exists(path):
            try:
                arr = np.load(path, allow_pickle=True)
                if len(arr) > 0:
                    ent = arr[0]
                    ent = ent.item() if hasattr(ent, 'item') else ent
                    if isinstance(ent, dict) and 'scale' in ent:
                        return float(ent['scale'])
            except Exception:
                pass
    return 1.0


def _ensure_qpos_dict(data) -> Dict[str, float]:
    if isinstance(data, dict):
        return {k: float(v) for k, v in data.items()}
    arr = np.asarray(data)
    return vector_to_qpos_dict(arr)


def build_bimanual_entry(
    left_vec,
    right_vec,
    scale: float,
    left_st: Optional[Dict[str, float]] = None,
    right_st: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    entry = {
        'qpos_left': _ensure_qpos_dict(left_vec),
        'qpos_right': _ensure_qpos_dict(right_vec),
        'scale': float(scale),
        'index_left': int(0),
        'index_right': int(0),
        'E_pen_left': float(0.0),
        'E_pen_right': float(0.0),
    }
    if left_st is not None:
        entry['qpos_left_st'] = _ensure_qpos_dict(left_st)
    if right_st is not None:
        entry['qpos_right_st'] = _ensure_qpos_dict(right_st)
    return entry


def save_bimanual_entry(path: str, entry: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.array([entry], dtype=object))

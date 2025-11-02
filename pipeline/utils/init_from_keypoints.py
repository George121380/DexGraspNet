import argparse
import os
import sys
import numpy as np
import transforms3d


JOINT_NAMES = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
]
ROTATION_NAMES = ['WRJRx', 'WRJRy', 'WRJRz']
TRANSLATION_NAMES = ['WRJTx', 'WRJTy', 'WRJTz']


def _axis_angle_from_to(v_from: np.ndarray, v_to: np.ndarray):
    v_from = v_from / (np.linalg.norm(v_from) + 1e-8)
    v_to = v_to / (np.linalg.norm(v_to) + 1e-8)
    dot = np.clip(np.dot(v_from, v_to), -1.0, 1.0)
    if dot > 1.0 - 1e-6:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32), 0.0
    if dot < -1.0 + 1e-6:
        # 180 deg: choose any orthogonal axis
        axis = np.cross(v_from, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(v_from, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        return axis, np.pi
    axis = np.cross(v_from, v_to)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    angle = float(np.arccos(dot))
    return axis, angle


def _build_wrist_euler_from_normal_with_local(n_local: np.ndarray, n_target_world: np.ndarray, twist_rad: float = 0.0) -> np.ndarray:
    # Find rotation that maps n_local (in wrist frame) to n_target_world (in world)
    n_local = n_local / (np.linalg.norm(n_local) + 1e-8)
    n_target_world = n_target_world / (np.linalg.norm(n_target_world) + 1e-8)
    axis = np.cross(n_local, n_target_world)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        # Vectors are parallel or anti-parallel
        if np.dot(n_local, n_target_world) > 0:
            R = np.eye(3, dtype=np.float32)
        else:
            # 180-degree flip around any perpendicular axis
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(np.dot(axis, n_local)) > 0.9:
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            R = transforms3d.axangles.axangle2mat(axis, np.pi).astype(np.float32)
    else:
        axis = axis / axis_norm
        angle = float(np.arccos(np.clip(np.dot(n_local, n_target_world), -1.0, 1.0)))
        R = transforms3d.axangles.axangle2mat(axis, angle).astype(np.float32)
    if abs(twist_rad) > 1e-8:
        R_twist = transforms3d.axangles.axangle2mat(n_target_world, float(twist_rad)).astype(np.float32)
        R = R_twist @ R
    rx, ry, rz = transforms3d.euler.mat2euler(R, axes='sxyz')
    return np.array([rx, ry, rz], dtype=np.float32)


def _qpos_from_tran_rot(tx, ty, tz, rx, ry, rz, fingers_zero=True, finger_values=None):
    q = {}
    if finger_values is None:
        for name in JOINT_NAMES:
            q[name] = 0.0 if fingers_zero else 0.0
    else:
        for name, val in zip(JOINT_NAMES, finger_values):
            q[name] = float(val)
    q['WRJRx'] = float(rx)
    q['WRJRy'] = float(ry)
    q['WRJRz'] = float(rz)
    q['WRJTx'] = float(tx)
    q['WRJTy'] = float(ty)
    q['WRJTz'] = float(tz)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--points', required=True, type=str)
    ap.add_argument('--kpleft', required=True, type=str)
    ap.add_argument('--kpright', required=True, type=str)
    ap.add_argument('--offset', type=float, default=0.04)
    ap.add_argument('--random_twist', action='store_true')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--biman_root', type=str, required=True)
    ap.add_argument('--out_entry', required=True, type=str)
    args = ap.parse_args()

    if args.seed is not None:
        np.random.seed(int(args.seed))

    pts = np.load(args.points).astype(np.float32)
    kpL = np.load(args.kpleft).astype(np.float32).reshape(3)
    kpR = np.load(args.kpright).astype(np.float32).reshape(3)
    centroid = np.mean(pts, axis=0)

    # Load HandModel to compute (1) palm normal in wrist frame and (2) wrist->palm centroid offset
    biman_root = os.path.abspath(args.biman_root)
    if biman_root not in sys.path:
        sys.path.insert(0, biman_root)
    parent_root = os.path.dirname(biman_root)
    thirdparty_root = os.path.join(parent_root, 'thirdparty')
    if thirdparty_root not in sys.path and os.path.isdir(thirdparty_root):
        sys.path.insert(0, thirdparty_root)
    from utils.hand_model import HandModel  # type: ignore

    def calibrate(handedness: str):
        cwd = os.getcwd()
        os.chdir(biman_root)
        try:
            xml = 'mjcf/left_shadow_hand.xml' if handedness == 'left' else 'mjcf/right_shadow_hand.xml'
            hand = HandModel(
                mjcf_path=xml,
                mesh_path='mjcf/meshes',
                contact_points_path='mjcf/left_hand_contact_points.json' if handedness == 'left' else 'mjcf/right_hand_contact_points.json',
                penetration_points_path='mjcf/penetration_points.json',
                device='cpu',
                handedness=f'{handedness}_hand'
            )
        finally:
            os.chdir(cwd)
        # zero pose
        trans = np.zeros(3, dtype=np.float32)
        R0 = np.eye(3, dtype=np.float32)
        rot6 = R0[:, :2].T.reshape(-1)
        joints = np.zeros(22, dtype=np.float32)
        hand_pose = np.concatenate([trans, rot6, joints])[None, ...]
        import torch
        hand.set_parameters(torch.tensor(hand_pose, dtype=torch.float))
        # Get palm vertices in world (== wrist frame since global R=I T=0)
        v = hand.current_status['robot0:palm'].transform_points(hand.mesh['robot0:palm']['vertices'])
        if len(v.shape) == 3:
            v = v[0]
        v = v.detach().cpu().numpy().astype(np.float32)
        p_centroid_local = v.mean(axis=0)
        vv = v - p_centroid_local
        C = vv.T @ vv
        eigvals, eigvecs = np.linalg.eigh(C)
        n_local = eigvecs[:, 0]
        n_local = n_local / (np.linalg.norm(n_local) + 1e-8)
        # Keep "outward" convention consistent: flip for right hand
        if handedness == 'right':
            n_local = -n_local
        return n_local.astype(np.float32), p_centroid_local.astype(np.float32)

    n_local_L, palm_off_L = calibrate('left')
    n_local_R, palm_off_R = calibrate('right')

    def compute_one(kp, handedness: str, n_local: np.ndarray, palm_local: np.ndarray):
        # Desired palm normal points from kp toward object centroid so the normal line passes through both
        dir_to_center = centroid - kp
        dir_norm = np.linalg.norm(dir_to_center) + 1e-8
        # Flip to let palm face the object (previous为手背朝向物体)
        n = -dir_to_center / dir_norm  # outward palm normal toward object
        # Desired palm center on the same line; move along outward normal n to远离物体
        p_center = kp + n * float(args.offset)
        twist = (np.random.rand() - 0.5) * 2 * np.pi if args.random_twist else 0.0
        euler = _build_wrist_euler_from_normal_with_local(n_local, n, twist_rad=twist)
        # Convert to rotation matrix
        R = transforms3d.euler.euler2mat(euler[0], euler[1], euler[2])
        # Compute wrist translation so that palm centroid hits p_center
        t = p_center - R @ palm_local
        return t, euler

    posL, eulL = compute_one(kpL, 'left', n_local_L, palm_off_L)
    posR, eulR = compute_one(kpR, 'right', n_local_R, palm_off_R)

    # Slightly curled canonical finger poses (same为初始化模块里的mu)
    mu_left = np.array([
        0.1, 0, -0.6, 0, 0, 0, -0.6, 0, -0.1, 0, -0.6, 0,
        0, -0.2, 0, -0.6, 0, 0, -1.2, 0, -0.2, 0
    ], dtype=np.float32)
    mu_right = np.array([
        0.1, 0, 0.6, 0, 0, 0, 0.6, 0, -0.1, 0, 0.6, 0,
        0, -0.2, 0, 0.6, 0, 0, 1.2, 0, -0.2, 0
    ], dtype=np.float32)

    qpos_left = _qpos_from_tran_rot(posL[0], posL[1], posL[2], eulL[0], eulL[1], eulL[2], fingers_zero=False, finger_values=mu_left)
    qpos_right = _qpos_from_tran_rot(posR[0], posR[1], posR[2], eulR[0], eulR[1], eulR[2], fingers_zero=False, finger_values=mu_right)

    entry = dict(
        qpos_left=qpos_left,
        qpos_right=qpos_right,
        scale=1.0,
        index_left=0,
        index_right=0,
        E_pen_left=0.0,
        E_pen_right=0.0,
    )
    os.makedirs(os.path.dirname(args.out_entry), exist_ok=True)
    np.save(args.out_entry, np.array([entry], dtype=object))


if __name__ == '__main__':
    main()



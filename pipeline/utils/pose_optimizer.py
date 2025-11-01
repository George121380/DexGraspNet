import os
import sys
import json
import argparse
from typing import Tuple, Optional

import numpy as np
import torch
import transforms3d
from tqdm import tqdm


def _euler_to_rot6d(rx: float, ry: float, rz: float) -> torch.Tensor:
    from thirdparty.pytorch_kinematics.pytorch_kinematics.transforms import matrix_to_rotation_6d  # type: ignore
    R = torch.tensor(transforms3d.euler.euler2mat(rx, ry, rz), dtype=torch.float)
    d6 = matrix_to_rotation_6d(R[None, ...])[0]
    return d6


def _dex_vec_to_hand_pose(vec: np.ndarray) -> torch.Tensor:
    """Map DexGrasp vector (J+6 with wrist at tail: Rx, Ry, Rz, Tx, Ty, Tz) to hand_pose (3+6+J).

    Assumptions:
      - vec order: finger joints (22) followed by [WRJRx, WRJRy, WRJRz, WRJTx, WRJTy, WRJTz]
      - units: radians and meters
    """
    assert vec.ndim == 1 and vec.shape[0] >= 28, f"Expected 1D vec with >=28, got {vec.shape}"
    fingers = vec[:-6].astype(np.float32)
    rx, ry, rz, tx, ty, tz = vec[-6:].astype(np.float32)
    rot6d = _euler_to_rot6d(float(rx), float(ry), float(rz)).numpy().astype(np.float32)
    hand_pose = np.concatenate([np.array([tx, ty, tz], dtype=np.float32), rot6d, fingers], axis=0)
    tensor = torch.from_numpy(hand_pose)[None, ...]
    return tensor.requires_grad_(True)


def optimize_from_dexgrasp(
    biman_root: str,
    data_root: str,
    experiments_base: str,
    object_code: str,
    left_vec: np.ndarray,
    right_vec: np.ndarray,
    steps: int = 100,
    gpu: str = "0",
    capture_video: bool = False,
    frames_dir: Optional[str] = None,
    video_path: Optional[str] = None,
    frame_stride: int = 10,
    width: int = 900,
    height: int = 900,
    show_contacts: bool = False,
    bg_color: str = "#E2F0D9",
    fps: int = 20,
    object_scale: Optional[float] = None,
    opt_step_size: Optional[float] = None,
) -> Tuple[dict, dict, Optional[str]]:
    # Prepare sys.path for BimanGrasp-Optimization
    if biman_root not in sys.path:
        sys.path.insert(0, biman_root)
    parent_root = os.path.dirname(biman_root)
    if parent_root not in sys.path:
        sys.path.insert(0, parent_root)
    thirdparty_root = os.path.join(parent_root, 'thirdparty')
    if thirdparty_root not in sys.path and os.path.isdir(thirdparty_root):
        sys.path.insert(0, thirdparty_root)

    from utils.hand_model import HandModel  # type: ignore
    from utils.object_model import ObjectModel  # type: ignore
    from utils.bimanual_handler import BimanualPair, hand_pose_to_dict  # type: ignore
    from utils.bimanual_energy import BimanualEnergyComputer  # type: ignore
    from utils.bimanual_optimizer import MALAOptimizer  # type: ignore
    from utils.config import ExperimentConfig  # type: ignore
    recorder = None

    device = torch.device(f"cuda:{gpu}" if (gpu != "cpu" and torch.cuda.is_available()) else "cpu")

    # Build models (ensure cwd points to biman_root so relative MJCF assets resolve)
    cwd = os.getcwd()
    os.chdir(biman_root)
    try:
        right_hand = HandModel(
            mjcf_path='mjcf/right_shadow_hand.xml',
            mesh_path='mjcf/meshes',
            contact_points_path='mjcf/right_hand_contact_points.json',
            penetration_points_path='mjcf/penetration_points.json',
            device=device,
            n_surface_points=2000,
            handedness='right_hand'
        )
        left_hand = HandModel(
            mjcf_path='mjcf/left_shadow_hand.xml',
            mesh_path='mjcf/meshes',
            contact_points_path='mjcf/left_hand_contact_points.json',
            penetration_points_path='mjcf/penetration_points.json',
            device=device,
            n_surface_points=2000,
            handedness='left_hand'
        )
    finally:
        os.chdir(cwd)

    # Set initial poses from DexGrasp
    left_pose = _dex_vec_to_hand_pose(left_vec).to(device)
    right_pose = _dex_vec_to_hand_pose(right_vec).to(device)
    contact_num = 4
    left_contacts = torch.randint(left_hand.n_contact_candidates, (left_pose.shape[0], contact_num), device=device)
    right_contacts = torch.randint(right_hand.n_contact_candidates, (right_pose.shape[0], contact_num), device=device)
    left_hand.set_parameters(left_pose, left_contacts)
    right_hand.set_parameters(right_pose, right_contacts)

    # Object model (loads mesh by code)
    obj = ObjectModel(data_root_path=data_root, batch_size_each=1, num_samples=2000, device=device)
    obj.initialize([object_code])
    if object_scale is not None:
        try:
            scale_tensor = torch.tensor([[float(object_scale)]], dtype=torch.float, device=device)
            obj.object_scale_tensor = scale_tensor
        except Exception:
            pass

    # Pair and optimization
    pair = BimanualPair(left_hand, right_hand, device)
    if capture_video:
        try:
            from utils.visualization_recorder import FrameRecorder  # type: ignore
            if frames_dir and video_path:
                os.makedirs(frames_dir, exist_ok=True)
                recorder = FrameRecorder(
                    object_model=obj,
                    bimanual_pair=pair,
                    global_idx=0,
                    frames_dir=frames_dir,
                    width=int(width),
                    height=int(height),
                    show_contacts=bool(show_contacts),
                    bg_color=bg_color,
                )
                recorder.capture(step=0)
            else:
                print("[Recorder] frames_dir or video_path not provided; disable video capture")
                recorder = None
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(f"[Recorder] Disabled due to error: {exc}")
            recorder = None

    cfg = ExperimentConfig()
    if opt_step_size is not None:
        try:
            cfg.optimizer.step_size = float(opt_step_size)
        except Exception:
            pass
    energy = BimanualEnergyComputer(cfg.energy, device)
    opt = MALAOptimizer(pair.left, pair.right, config=cfg.optimizer, device=device)
    # Override iterations
    opt.num_iterations = int(steps)

    # Single-loop optimization
    energy_terms = energy.compute_all_energies(pair, obj, verbose=False)
    energy_terms.total.sum().backward(retain_graph=True)
    accepted_total = 0
    total_proposals = 0
    for step in tqdm(range(1, int(steps) + 1), total=int(steps), desc="optimizing", dynamic_ncols=True):
        opt.langevin_proposal()
        opt.zero_grad()
        new_terms = energy.compute_all_energies(pair, obj, verbose=False)
        new_terms.total.sum().backward(retain_graph=True)
        with torch.no_grad():
            accept, _ = opt.metropolis_hastings_step(energy_terms.total, new_terms.total)
            accepted_total += int(accept.sum().item())
            total_proposals += int(accept.numel())
            energy_terms.total[accept] = new_terms.total[accept]
            energy_terms.distance[accept] = new_terms.distance[accept]
            energy_terms.force_closure[accept] = new_terms.force_closure[accept]
            energy_terms.penetration[accept] = new_terms.penetration[accept]
            energy_terms.self_penetration[accept] = new_terms.self_penetration[accept]
            energy_terms.joint_limits[accept] = new_terms.joint_limits[accept]

        if recorder and (step % max(frame_stride, 1) == 0 or step == int(steps)):
            try:
                recorder.capture(step)
            except Exception as exc:  # pragma: no cover
                print(f"[Recorder] capture failed at step {step}: {exc}")
                recorder = None

    # Export results
    left_qpos = hand_pose_to_dict(left_hand.hand_pose[0])
    right_qpos = hand_pose_to_dict(right_hand.hand_pose[0])
    if total_proposals > 0:
        ar = accepted_total / total_proposals
        print(f"[Opt] acceptance_ratio={ar:.6f} ({accepted_total}/{total_proposals})")
    if recorder and video_path:
        try:
            recorder.finalize(video_path, fps=int(fps))
            print(f"[Recorder] video saved to {video_path}")
        except Exception as exc:  # pragma: no cover
            print(f"[Recorder] finalize failed: {exc}")
    return left_qpos, right_qpos, video_path if recorder and video_path else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--biman_root', required=True, type=str)
    ap.add_argument('--data_root', required=True, type=str)
    ap.add_argument('--experiments_base', required=False, type=str, default=None)
    ap.add_argument('--object_code', required=True, type=str)
    ap.add_argument('--dex_npz', required=True, type=str)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--out_npz', type=str, required=True)
    ap.add_argument('--out_json', type=str, required=True)
    ap.add_argument('--capture_video', action='store_true')
    ap.add_argument('--frames_dir', type=str, default='')
    ap.add_argument('--video_path', type=str, default='')
    ap.add_argument('--frame_stride', type=int, default=10)
    ap.add_argument('--video_width', type=int, default=900)
    ap.add_argument('--video_height', type=int, default=900)
    ap.add_argument('--video_fps', type=int, default=20)
    ap.add_argument('--show_contacts', action='store_true')
    ap.add_argument('--bg_color', type=str, default='#E2F0D9')
    ap.add_argument('--object_scale', type=float, default=None)
    ap.add_argument('--opt_step_size', type=float, default=None)
    args = ap.parse_args()

    data = np.load(args.dex_npz)
    left_vec = data['left'][0]
    right_vec = data['right'][0]

    # Build config via optimize function (handles sys.path setup)
    lq, rq, video_path = optimize_from_dexgrasp(
        biman_root=args.biman_root,
        data_root=args.data_root,
        experiments_base=args.experiments_base or '',
        object_code=args.object_code,
        left_vec=left_vec,
        right_vec=right_vec,
        steps=int(args.steps),
        gpu=args.gpu,
        capture_video=args.capture_video,
        frames_dir=args.frames_dir or None,
        video_path=args.video_path or None,
        frame_stride=int(args.frame_stride),
        width=int(args.video_width),
        height=int(args.video_height),
        show_contacts=bool(args.show_contacts),
        bg_color=args.bg_color,
        fps=int(args.video_fps),
        object_scale=args.object_scale,
        opt_step_size=args.opt_step_size,
    )

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez(args.out_npz, left_qpos=lq, right_qpos=rq)
    with open(args.out_json, 'w') as f:
        json.dump({"left_qpos": lq, "right_qpos": rq}, f, indent=2)

    if video_path:
        meta_path = os.path.join(os.path.dirname(args.out_json), 'optimization_video.txt')
        with open(meta_path, 'w') as f:
            f.write(video_path)


if __name__ == '__main__':
    main()




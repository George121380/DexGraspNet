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
    left_pose_override: Optional[torch.Tensor] = None,
    right_pose_override: Optional[torch.Tensor] = None,
    accept_warmup_steps: int = 0,
    freeze_joints_steps: int = 0,
    freeze_translation_steps: int = 0,
    noise_factor: Optional[float] = None,
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

    # Set initial poses from DexGrasp or direct overrides
    if left_pose_override is not None:
        left_pose = left_pose_override.to(device)
    else:
        left_pose = _dex_vec_to_hand_pose(left_vec).to(device)
    if right_pose_override is not None:
        right_pose = right_pose_override.to(device)
    else:
        right_pose = _dex_vec_to_hand_pose(right_vec).to(device)
    contact_num = 4
    left_contacts = torch.randint(left_hand.n_contact_candidates, (left_pose.shape[0], contact_num), device=device)
    right_contacts = torch.randint(right_hand.n_contact_candidates, (right_pose.shape[0], contact_num), device=device)
    left_hand.set_parameters(left_pose, left_contacts)
    right_hand.set_parameters(right_pose, right_contacts)

    # Object model (loads mesh by code); match batch size of poses
    batch_size = int(left_hand.hand_pose.shape[0])
    obj = ObjectModel(data_root_path=data_root, batch_size_each=batch_size, num_samples=2000, device=device)
    obj.initialize([object_code])
    if object_scale is not None:
        try:
            bs = int(batch_size)
            scale_tensor = torch.full((bs, 1), float(object_scale), dtype=torch.float, device=device)
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
    # Apply energy weight overrides from environment (set by pipeline)
    try:
        w_dis_env = os.environ.get('PIPELINE_W_DIS', None)
        w_pen_env = os.environ.get('PIPELINE_W_PEN', None)
        w_spen_env = os.environ.get('PIPELINE_W_SPEN', None)
        w_joints_env = os.environ.get('PIPELINE_W_JOINTS', None)
        w_vew_env = os.environ.get('PIPELINE_W_VEW', None)
        if w_dis_env is not None:
            cfg.energy.w_dis = float(w_dis_env)
        if w_pen_env is not None:
            cfg.energy.w_pen = float(w_pen_env)
        if w_spen_env is not None:
            cfg.energy.w_spen = float(w_spen_env)
        if w_joints_env is not None:
            cfg.energy.w_joints = float(w_joints_env)
        if w_vew_env is not None:
            cfg.energy.w_vew = float(w_vew_env)
    except Exception:
        pass
    energy = BimanualEnergyComputer(cfg.energy, device)
    opt = MALAOptimizer(pair.left, pair.right, config=cfg.optimizer, device=device)
    # Optional: reduce stochasticity to避免早期偏移
    if noise_factor is not None:
        try:
            opt.langevin_noise_factor = torch.tensor(float(noise_factor), dtype=torch.float, device=device)
        except Exception:
            pass
    # Override iterations
    opt.num_iterations = int(steps)

    # Single-loop optimization
    energy_terms = energy.compute_all_energies(pair, obj, verbose=False)
    energy_terms.total.sum().backward(retain_graph=True)
    accepted_total = 0
    total_proposals = 0
    # Cache initial joint angles for optional early freeze
    joint_start_idx = 3 + 6
    left_joints_init = left_hand.hand_pose.clone()[:, joint_start_idx:]
    right_joints_init = right_hand.hand_pose.clone()[:, joint_start_idx:]
    left_trans_init = left_hand.hand_pose.clone()[:, :3]
    right_trans_init = right_hand.hand_pose.clone()[:, :3]

    progress_file = os.environ.get('PIPELINE_PROGRESS_FILE', '')
    for step in tqdm(range(1, int(steps) + 1), total=int(steps), desc="optimizing", dynamic_ncols=True):
        opt.langevin_proposal()
        if step <= int(freeze_joints_steps):
            # Keep finger joints fixed in early iterations to stabilize wrist/object positioning
            with torch.no_grad():
                left_contacts_cur = left_hand.contact_point_indices.clone()
                right_contacts_cur = right_hand.contact_point_indices.clone()
                left_pose_cur = left_hand.hand_pose.detach()
                right_pose_cur = right_hand.hand_pose.detach()
                left_new = torch.cat([left_pose_cur[:, :joint_start_idx], left_joints_init], dim=1).clone().requires_grad_(True)
                right_new = torch.cat([right_pose_cur[:, :joint_start_idx], right_joints_init], dim=1).clone().requires_grad_(True)
                left_hand.set_parameters(left_new, left_contacts_cur)
                right_hand.set_parameters(right_new, right_contacts_cur)
        if step <= int(freeze_translation_steps):
            # Keep wrist translation fixed to initial value to防止早期整体漂移
            with torch.no_grad():
                left_contacts_cur = left_hand.contact_point_indices.clone()
                right_contacts_cur = right_hand.contact_point_indices.clone()
                left_pose_cur = left_hand.hand_pose.detach()
                right_pose_cur = right_hand.hand_pose.detach()
                left_new = torch.cat([left_trans_init, left_pose_cur[:, 3:]], dim=1).clone().requires_grad_(True)
                right_new = torch.cat([right_trans_init, right_pose_cur[:, 3:]], dim=1).clone().requires_grad_(True)
                left_hand.set_parameters(left_new, left_contacts_cur)
                right_hand.set_parameters(right_new, right_contacts_cur)
        opt.zero_grad()
        new_terms = energy.compute_all_energies(pair, obj, verbose=False)
        new_terms.total.sum().backward(retain_graph=True)
        with torch.no_grad():
            if step <= int(accept_warmup_steps):
                accept = torch.ones_like(new_terms.total, dtype=torch.bool)
            else:
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
        # progress file update
        if progress_file:
            try:
                with open(progress_file, 'w') as pf:
                    pf.write(str(step))
            except Exception:
                pass

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
    ap.add_argument('--dex_npz', required=False, type=str, default=None)
    ap.add_argument('--entry', required=False, type=str, default=None)
    ap.add_argument('--entry_multi', action='append', default=None, help='Repeatable. Multiple entry .npy paths for batched optimization')
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--out_npz', type=str, required=False, default='')
    ap.add_argument('--out_json', type=str, required=False, default='')
    ap.add_argument('--out_dir', type=str, default='', help='Output directory for batched mode (writes optimized_<suffix>.json/npz)')
    ap.add_argument('--suffixes', type=str, default='', help='Comma-separated suffix list aligned to entry_multi')
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
    ap.add_argument('--accept_warmup', type=int, default=0)
    ap.add_argument('--freeze_joints_steps', type=int, default=0)
    ap.add_argument('--freeze_translation_steps', type=int, default=0)
    ap.add_argument('--progress_file', type=str, default='')
    args = ap.parse_args()

    def qpos_dict_to_vec(qd: dict) -> np.ndarray:
        # Order: 22 joints then [Rx, Ry, Rz, Tx, Ty, Tz]
        joint_names = [
            'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
            'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
            'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
            'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
            'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
        ]
        fingers = [float(qd.get(n, 0.0)) for n in joint_names]
        rx = float(qd.get('WRJRx', 0.0)); ry = float(qd.get('WRJRy', 0.0)); rz = float(qd.get('WRJRz', 0.0))
        tx = float(qd.get('WRJTx', 0.0)); ty = float(qd.get('WRJTy', 0.0)); tz = float(qd.get('WRJTz', 0.0))
        return np.asarray(fingers + [rx, ry, rz, tx, ty, tz], dtype=np.float32)

    def qpos_dict_to_pose(qpos: dict) -> torch.Tensor:
        trans = [float(qpos.get('WRJTx', 0.0)), float(qpos.get('WRJTy', 0.0)), float(qpos.get('WRJTz', 0.0))]
        R = transforms3d.euler.euler2mat(float(qpos.get('WRJRx', 0.0)), float(qpos.get('WRJRy', 0.0)), float(qpos.get('WRJRz', 0.0)))
        rot6d = torch.tensor(R[:, :2].T.reshape(-1), dtype=torch.float)
        joint_names = [
            'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
            'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
            'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
            'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
            'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
        ]
        fingers = torch.tensor([float(qpos.get(n, 0.0)) for n in joint_names], dtype=torch.float)
        data = torch.tensor(trans, dtype=torch.float)
        pose = torch.cat([data, rot6d, fingers], dim=0).unsqueeze(0)
        return pose.requires_grad_(True)

    # Batched mode: multiple entries optimized within a single process
    if args.entry_multi is not None and len(args.entry_multi) > 0:
        entry_paths = list(args.entry_multi)
        suffix_list = [s for s in args.suffixes.split(',') if s]
        if suffix_list and len(suffix_list) != len(entry_paths):
            raise RuntimeError("suffixes count must match entry_multi count")
        # Resolve entries
        entries = []
        for p in entry_paths:
            arr = np.load(p, allow_pickle=True)
            if isinstance(arr, np.ndarray) and len(arr) > 0:
                ent = arr[0]
                if hasattr(ent, 'item'):
                    ent = ent.item()
                entries.append(ent)
            else:
                raise RuntimeError(f"Invalid entry file: {p}")

        # Prepare batched poses
        left_list = []
        right_list = []
        for ent in entries:
            left_list.append(qpos_dict_to_pose(ent['qpos_left']))
            right_list.append(qpos_dict_to_pose(ent['qpos_right']))
        left_pose_override = torch.cat(left_list, dim=0)
        right_pose_override = torch.cat(right_list, dim=0)

        # Run optimize (inline to handle batch outputs)
        # Prepare sys.path
        if args.biman_root not in sys.path:
            sys.path.insert(0, args.biman_root)
        parent_root = os.path.dirname(args.biman_root)
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

        device = torch.device(f"cuda:{args.gpu}" if (args.gpu != "cpu" and torch.cuda.is_available()) else "cpu")
        # Build models (cwd hack for MJCF)
        cwd = os.getcwd()
        os.chdir(args.biman_root)
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

        left_pose = left_pose_override.to(device)
        right_pose = right_pose_override.to(device)
        contact_num = 4
        left_contacts = torch.randint(left_hand.n_contact_candidates, (left_pose.shape[0], contact_num), device=device)
        right_contacts = torch.randint(right_hand.n_contact_candidates, (right_pose.shape[0], contact_num), device=device)
        left_hand.set_parameters(left_pose, left_contacts)
        right_hand.set_parameters(right_pose, right_contacts)

        # Object model batch
        obj = ObjectModel(data_root_path=args.data_root, batch_size_each=left_pose.shape[0], num_samples=2000, device=device)
        obj.initialize([args.object_code])
        if args.object_scale is not None:
            try:
                bs = int(left_pose.shape[0])
                scale_tensor = torch.full((bs, 1), float(args.object_scale), dtype=torch.float, device=device)
                obj.object_scale_tensor = scale_tensor
            except Exception:
                pass

        pair = BimanualPair(left_hand, right_hand, device)
        cfg = ExperimentConfig()
        if args.opt_step_size is not None:
            try:
                cfg.optimizer.step_size = float(args.opt_step_size)
            except Exception:
                pass
        # Energy overrides via env
        try:
            w_dis_env = os.environ.get('PIPELINE_W_DIS', None)
            w_pen_env = os.environ.get('PIPELINE_W_PEN', None)
            w_spen_env = os.environ.get('PIPELINE_W_SPEN', None)
            w_joints_env = os.environ.get('PIPELINE_W_JOINTS', None)
            w_vew_env = os.environ.get('PIPELINE_W_VEW', None)
            if w_dis_env is not None:
                cfg.energy.w_dis = float(w_dis_env)
            if w_pen_env is not None:
                cfg.energy.w_pen = float(w_pen_env)
            if w_spen_env is not None:
                cfg.energy.w_spen = float(w_spen_env)
            if w_joints_env is not None:
                cfg.energy.w_joints = float(w_joints_env)
            if w_vew_env is not None:
                cfg.energy.w_vew = float(w_vew_env)
        except Exception:
            pass
        energy = BimanualEnergyComputer(cfg.energy, device)
        opt = MALAOptimizer(pair.left, pair.right, config=cfg.optimizer, device=device)
        if os.environ.get('PIPELINE_NOISE_FACTOR'):
            try:
                opt.langevin_noise_factor = torch.tensor(float(os.environ['PIPELINE_NOISE_FACTOR']), dtype=torch.float, device=device)
            except Exception:
                pass
        opt.num_iterations = int(args.steps)

        # Prepare progress
        progress_file = os.environ.get('PIPELINE_PROGRESS_FILE', '')
        # Pre-compute energy
        energy_terms = energy.compute_all_energies(pair, obj, verbose=False)
        energy_terms.total.sum().backward(retain_graph=True)
        # Cache initials for freezes
        joint_start_idx = 3 + 6
        left_joints_init = left_hand.hand_pose.clone()[:, joint_start_idx:]
        right_joints_init = right_hand.hand_pose.clone()[:, joint_start_idx:]
        left_trans_init = left_hand.hand_pose.clone()[:, :3]
        right_trans_init = right_hand.hand_pose.clone()[:, :3]
        for step in tqdm(range(1, int(args.steps) + 1), total=int(args.steps), desc="optimizing", dynamic_ncols=True):
            opt.langevin_proposal()
            if step <= int(args.freeze_joints_steps or 0):
                with torch.no_grad():
                    lc = left_hand.contact_point_indices.clone(); rc = right_hand.contact_point_indices.clone()
                    lp = left_hand.hand_pose.detach(); rp = right_hand.hand_pose.detach()
                    left_new = torch.cat([lp[:, :joint_start_idx], left_joints_init], dim=1).clone().requires_grad_(True)
                    right_new = torch.cat([rp[:, :joint_start_idx], right_joints_init], dim=1).clone().requires_grad_(True)
                    left_hand.set_parameters(left_new, lc); right_hand.set_parameters(right_new, rc)
            if step <= int(args.freeze_translation_steps or 0):
                with torch.no_grad():
                    lc = left_hand.contact_point_indices.clone(); rc = right_hand.contact_point_indices.clone()
                    lp = left_hand.hand_pose.detach(); rp = right_hand.hand_pose.detach()
                    left_new = torch.cat([left_trans_init, lp[:, 3:]], dim=1).clone().requires_grad_(True)
                    right_new = torch.cat([right_trans_init, rp[:, 3:]], dim=1).clone().requires_grad_(True)
                    left_hand.set_parameters(left_new, lc); right_hand.set_parameters(right_new, rc)
            opt.zero_grad()
            new_terms = energy.compute_all_energies(pair, obj, verbose=False)
            new_terms.total.sum().backward(retain_graph=True)
            with torch.no_grad():
                if step <= int(args.accept_warmup or 0):
                    accept = torch.ones_like(new_terms.total, dtype=torch.bool)
                else:
                    accept, _ = opt.metropolis_hastings_step(energy_terms.total, new_terms.total)
                energy_terms.total[accept] = new_terms.total[accept]
                energy_terms.distance[accept] = new_terms.distance[accept]
                energy_terms.force_closure[accept] = new_terms.force_closure[accept]
                energy_terms.penetration[accept] = new_terms.penetration[accept]
                energy_terms.self_penetration[accept] = new_terms.self_penetration[accept]
                energy_terms.joint_limits[accept] = new_terms.joint_limits[accept]
            if progress_file:
                try:
                    with open(progress_file, 'w') as pf:
                        pf.write(str(step))
                except Exception:
                    pass

        # Write outputs per sample
        out_dir = args.out_dir or os.path.dirname(args.out_json or args.out_npz or '.')
        os.makedirs(out_dir, exist_ok=True)
        for idx, ent in enumerate(entries):
            lq = hand_pose_to_dict(left_hand.hand_pose[idx])
            rq = hand_pose_to_dict(right_hand.hand_pose[idx])
            suffix = suffix_list[idx] if suffix_list else f"{idx:02d}"
            out_json_i = os.path.join(out_dir, f"optimized_{suffix}.json")
            out_npz_i = os.path.join(out_dir, f"optimized_{suffix}.npz")
            with open(out_json_i, 'w') as f:
                json.dump({"left_qpos": lq, "right_qpos": rq}, f, indent=2)
            np.savez(out_npz_i, left_qpos=lq, right_qpos=rq)
        return

    if args.entry is not None and len(args.entry) > 0:
        arr = np.load(args.entry, allow_pickle=True)
        if isinstance(arr, np.ndarray) and len(arr) > 0:
            ent = arr[0]
            if hasattr(ent, 'item'):
                ent = ent.item()
        else:
            raise RuntimeError(f"Invalid entry file: {args.entry}")
        left_vec = qpos_dict_to_vec(ent['qpos_left'])
        right_vec = qpos_dict_to_vec(ent['qpos_right'])
        # Build direct hand_pose overrides to ensure exact match with Step4 visualization
        left_pose_override = qpos_dict_to_pose(ent['qpos_left'])
        right_pose_override = qpos_dict_to_pose(ent['qpos_right'])
    else:
        if args.dex_npz is None or not os.path.exists(args.dex_npz):
            raise RuntimeError("Either --dex_npz or --entry must be provided")
        data = np.load(args.dex_npz)
        left_vec = data['left'][0]
        right_vec = data['right'][0]
        left_pose_override = None
        right_pose_override = None

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
        left_pose_override=left_pose_override,
        right_pose_override=right_pose_override,
        accept_warmup_steps=int(args.accept_warmup or 0),
        freeze_joints_steps=int(args.freeze_joints_steps or 0),
        freeze_translation_steps=int(args.freeze_translation_steps or 0),
        noise_factor=float(os.environ.get('PIPELINE_NOISE_FACTOR', 'nan')) if os.environ.get('PIPELINE_NOISE_FACTOR') else None,
    )

    # Single-sample outputs
    if args.out_npz:
        os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
        np.savez(args.out_npz, left_qpos=lq, right_qpos=rq)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump({"left_qpos": lq, "right_qpos": rq}, f, indent=2)

    if video_path:
        meta_path = os.path.join(os.path.dirname(args.out_json), 'optimization_video.txt')
        with open(meta_path, 'w') as f:
            f.write(video_path)


if __name__ == '__main__':
    main()




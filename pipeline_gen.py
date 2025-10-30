import os
import sys
import argparse
import json
import shutil
import subprocess
import uuid
from typing import Tuple, Optional, List

import numpy as np
import time
try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


def ensure_dir(path: str, clean: bool = False) -> None:
    """Create a directory; optionally remove it first."""
    if clean and os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def save_numpy(path: str, arr: np.ndarray) -> None:
    """Save numpy array to path with parent creation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def load_points_as_is(points_path: str) -> np.ndarray:
    """Load a preprocessed point cloud as (N, 3) float array.

    Assumes the point cloud was preprocessed offline. Only supports .npy format for now.
    """
    if not points_path.endswith('.npy'):
        raise ValueError("Only .npy point clouds are supported by this orchestrator. Provide an (N,3) float numpy array.")
    pts = np.load(points_path)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points of shape (N,3), got {pts.shape}")
    return pts.astype(np.float32)


def softmax_topk_sample(points: np.ndarray,
                        scores: np.ndarray,
                        *,
                        top_k: int = 512,
                        temperature: float = 0.3,
                        avoid_indices: Optional[List[int]] = None,
                        avoid_radius: float = 0.0) -> int:
    """Sample one index using softmax over top-K scores with optional distance avoidance.

    - Normalizes scores to [0,1] range to stabilize the softmax temperature effect.
    - Optionally removes candidates too close to 'avoid_indices' by a radius.
    """
    assert points.ndim == 2 and points.shape[1] == 3, "points must be (N,3)"
    assert scores.ndim == 1 and scores.shape[0] == points.shape[0], "scores must be (N,)"

    N = points.shape[0]
    idx_all = np.arange(N)

    # Apply distance avoidance mask if needed
    if avoid_indices is not None and len(avoid_indices) > 0 and avoid_radius > 0.0:
        keep_mask = np.ones(N, dtype=bool)
        for a in avoid_indices:
            d = np.linalg.norm(points - points[a], axis=1)
            keep_mask &= (d >= float(avoid_radius))
        idx_all = idx_all[keep_mask]
        if idx_all.size == 0:
            # fallback: ignore avoidance if it removes all
            idx_all = np.arange(N)

    # Select top-K from remaining candidates
    k = int(max(1, min(top_k, idx_all.size)))
    cand_scores = scores[idx_all]
    top_idx_rel = np.argpartition(-cand_scores, kth=k - 1)[:k]
    top_indices = idx_all[top_idx_rel]

    # Min-max normalize to stabilize softmax
    s = scores[top_indices]
    s_min, s_max = float(s.min()), float(s.max())
    if s_max - s_min < 1e-8:
        # nearly flat distribution -> uniform sample among top-k
        return int(np.random.choice(top_indices))
    s_hat = (s - s_min) / (s_max - s_min + 1e-12)

    # Softmax with temperature
    tau = max(1e-6, float(temperature))
    logits = s_hat / tau
    logits = logits - float(logits.max())  # numerical stability
    probs = np.exp(logits)
    probs = probs / (probs.sum() + 1e-12)

    pick_rel = int(np.random.choice(np.arange(k), p=probs))
    return int(top_indices[pick_rel])


def apply_radial_penalty(points: np.ndarray,
                         scores: np.ndarray,
                         *,
                         center: np.ndarray,
                         sigma: float = 0.03,
                         min_dist: float = 0.05) -> np.ndarray:
    """Apply radial penalty around 'center' to encourage complementary picks for the second hand.

    - Zero out scores inside 'min_dist'.
    - Multiply outside by (1 - exp(-d^2/(2*sigma^2))).
    """
    assert center.shape == (3,), "center must be (3,)"
    d = np.linalg.norm(points - center[None, :], axis=1)
    scores2 = scores.copy()
    scores2[d < float(min_dist)] = 0.0
    if float(sigma) > 1e-9:
        penalty = 1.0 - np.exp(- (d ** 2) / (2.0 * float(sigma) ** 2))
        scores2 *= penalty
    return scores2


def run_subprocess(cmd: List[str], *, cwd: Optional[str] = None, timeout: Optional[int] = None, log_path: Optional[str] = None) -> None:
    """Run a subprocess with robust logging and error handling."""
    out = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, text=True)
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            f.write("COMMAND:\n" + " ".join(cmd) + "\n\n")
            f.write("STDOUT:\n" + (out.stdout or "") + "\n\n")
            f.write("STDERR:\n" + (out.stderr or "") + "\n")
    if out.returncode != 0:
        raise RuntimeError(f"Subprocess failed with code {out.returncode}. See log: {log_path or '<captured>'}")


def write_affordance_html(points: np.ndarray, scores: np.ndarray, sample_point: Optional[np.ndarray], out_path: str, title: str) -> None:
    """Write a simple Plotly HTML visualizing per-point scores and the sampled point."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not _HAS_PLOTLY:
        # Fallback: write a tiny HTML note and save arrays alongside
        with open(out_path, 'w') as f:
            f.write("<html><body><h3>Plotly not installed</h3><p>Cannot render 3D visualization.</p></body></html>")
        np.save(out_path.replace('.html', '_points.npy'), points)
        np.save(out_path.replace('.html', '_scores.npy'), scores)
        if sample_point is not None:
            np.save(out_path.replace('.html', '_sample.npy'), sample_point)
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=2, color=scores, colorscale='Viridis', showscale=True),
        name='scores'))
    if sample_point is not None:
        fig.add_trace(go.Scatter3d(
            x=[float(sample_point[0])], y=[float(sample_point[1])], z=[float(sample_point[2])],
            mode='markers', marker=dict(size=7, color='red', symbol='circle'), name='sample'))
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data')
    fig.update_layout(title=title, title_x=0.5)
    fig.write_html(out_path)


class AffordanceServerClient:
    """Manage a persistent affordance inference server in the PN env.

    The server loads both models once and accepts JSONL requests on stdin/stdout.
    """
    def __init__(self, *, conda_exe: str, pn_env: str, first_ckpt: str, second_ckpt: Optional[str], device: str,
                 condition_mode: str, use_condition: bool, logs_dir: str, quiet: bool = False) -> None:
        import subprocess as _sp
        self._sp = _sp
        self._proc = None
        self._logs_dir = logs_dir
        self._quiet = bool(quiet)
        os.makedirs(self._logs_dir, exist_ok=True)
        server_script = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'tools', 'affordance_server.py')
        cmd = [
            conda_exe, 'run', '-n', pn_env, '--no-capture-output',
            'python', server_script,
            '--first_ckpt', os.path.abspath(first_ckpt),
            '--device', device,
            '--condition_mode', condition_mode,
        ]
        if use_condition:
            cmd.append('--use_condition')
        if second_ckpt is not None and len(str(second_ckpt)) > 0 and os.path.isfile(os.path.abspath(second_ckpt)):
            cmd.extend(['--second_ckpt', os.path.abspath(second_ckpt)])
        # Stderr to file for debugging; keep stdout for protocol
        self._stderr_fp = open(os.path.join(self._logs_dir, 'pn_server_stderr.txt'), 'w')
        if not self._quiet:
            print('[pipeline] Starting PN affordance server (loading models...)', flush=True)
        self._proc = _sp.Popen(cmd, cwd=os.path.abspath(os.path.dirname(__file__)), stdin=_sp.PIPE, stdout=_sp.PIPE, stderr=self._stderr_fp, text=True, bufsize=1)
        # Wait for READY (skip any non-READY noise lines)
        while True:
            ready = (self._proc.stdout.readline() or '').strip()
            if ready == 'READY':
                break
            if ready == '':
                raise RuntimeError("Affordance server failed to start; no READY signal")
        if not self._quiet:
            print('[pipeline] PN affordance server READY (models loaded)', flush=True)

    def request(self, payload: dict) -> dict:
        assert self._proc and self._proc.stdin and self._proc.stdout
        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            rc = self._proc.poll()
            raise RuntimeError(f"Affordance server is not available (rc={rc}). See pn_server_stderr.txt for details.")
        # Read lines until we get a JSON object
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError('Affordance server closed unexpectedly while waiting for response')
            line = line.strip()
            try:
                rsp = json.loads(line)
            except Exception:
                continue
            if not rsp.get('ok', False):
                raise RuntimeError(f"Affordance server error: {rsp.get('error')}")
            return rsp

    def infer_first(self, points_path: str, output_dir: str) -> str:
        rsp = self.request({'type': 'first', 'points_path': os.path.abspath(points_path), 'output_dir': os.path.abspath(output_dir)})
        return str(rsp['scores_path'])

    def infer_second(self, points_path: str, center_path: str, output_dir: str, *, condition_mode: str, use_condition: bool) -> str:
        rsp = self.request({
            'type': 'second',
            'points_path': os.path.abspath(points_path),
            'center_path': os.path.abspath(center_path),
            'output_dir': os.path.abspath(output_dir),
            'condition_mode': condition_mode,
            'use_condition': bool(use_condition),
        })
        return str(rsp['scores_path'])

    def shutdown(self) -> None:
        try:
            if self._proc and self._proc.stdin and self._proc.stdout:
                self._proc.stdin.write(json.dumps({'type': 'shutdown'}) + "\n")
                self._proc.stdin.flush()
                _ = self._proc.stdout.readline()
        except Exception:
            pass
        finally:
            try:
                if self._proc:
                    self._proc.terminate()
            except Exception:
                pass
            try:
                if hasattr(self, '_stderr_fp') and self._stderr_fp:
                    self._stderr_fp.close()
            except Exception:
                pass


def orchestrate(args: argparse.Namespace) -> None:
    raise RuntimeError('Single-run mode has been removed. Please use --interactive to run the pipeline.')


def orchestrate_interactive(args: argparse.Namespace) -> None:
    """Interactive mode: preload PN models once, then process multiple point clouds sequentially."""
    def _log(msg: str) -> None:
        if not getattr(args, 'quiet', False):
            print(f"[pipeline][interactive] {msg}", flush=True)

    base_out_dir = os.path.abspath(args.out_dir)
    ensure_dir(base_out_dir, clean=False)
    session_tag = time.strftime('%Y%m%d_%H%M%S')
    session_dir = os.path.join(base_out_dir, f'session_{session_tag}')
    ensure_dir(session_dir, clean=False)
    logs_dir = os.path.join(session_dir, 'logs')
    ensure_dir(logs_dir, clean=False)

    _log('Starting interactive session')
    _log(f"first_ckpt: {os.path.abspath(args.first_ckpt)}")
    _log(f"second_ckpt: {os.path.abspath(args.second_ckpt) if args.second_ckpt else '<fallback to first>'}")
    _log(f"session_dir: {session_dir}")

    client = AffordanceServerClient(
        conda_exe=args.conda_exe,
        pn_env=args.pn_env,
        first_ckpt=args.first_ckpt,
        second_ckpt=args.second_ckpt,
        device=args.device,
        condition_mode=args.condition_mode,
        use_condition=bool(args.use_condition),
        logs_dir=logs_dir,
        quiet=getattr(args, 'quiet', False),
    )

    try:
        run_idx = 0
        while True:
            try:
                prompt = "Enter object code (e.g., 100015) or absolute path to points .npy (q to quit): "
                inp = input(prompt).strip()
            except EOFError:
                break
            if inp.lower() in {'q', 'quit', 'exit'}:
                break
            if not inp:
                continue

            # Resolve input to a points.npy path
            selected_code: Optional[str] = None
            if inp.lower().endswith('.npy') and os.path.isfile(inp):
                resolved_points = os.path.abspath(inp)
            else:
                selected_code = str(inp)
                def _resolve_from_code(code: str) -> Optional[str]:
                    candidates = []
                    # Highest priority: explicit CLI root
                    if getattr(args, 'input_root', None):
                        candidates.append(os.path.join(os.path.abspath(args.input_root), code, args.points_filename))
                    # Env override
                    env_root = os.getenv('PIPELINE_INPUT_ROOT', '')
                    if len(env_root) > 0:
                        candidates.append(os.path.join(os.path.abspath(env_root), code, args.points_filename))
                    # Repository-local common defaults
                    repo_root = os.path.abspath(os.path.dirname(__file__))
                    candidates.append(os.path.join(repo_root, 'data', 'pipeline_input', code, args.points_filename))
                    candidates.append(os.path.join(repo_root, 'data', 'preprocess_data', 'pot_data', 'train', code, args.points_filename))
                    for cand in candidates:
                        if os.path.isfile(cand):
                            return os.path.abspath(cand)
                    return None
                resolved_points = _resolve_from_code(selected_code)
                if resolved_points is None:
                    print("Could not resolve points file for code='{}'".format(selected_code))
                    if getattr(args, 'input_root', None):
                        print("Tried: {}".format(os.path.join(os.path.abspath(args.input_root), selected_code, args.points_filename)))
                    print("You may set PIPELINE_INPUT_ROOT or provide an absolute .npy path.")
                    continue

            # Per-run directories
            out_dir = os.path.join(session_dir, f'run_{run_idx:03d}')
            ensure_dir(out_dir, clean=False)
            tmp_dir = os.path.join(out_dir, f"tmp_{uuid.uuid4().hex[:8]}")
            ensure_dir(tmp_dir, clean=True)
            vis_dir = os.path.join(out_dir, 'vis')
            ensure_dir(vis_dir, clean=False)

            try:
                _log(f"[run {run_idx}] Loading points: {resolved_points}")
                points = load_points_as_is(resolved_points)
                # Optional pre-downsampling to ensure consistent mapping with server outputs
                if args.pn_max_points is not None and int(args.pn_max_points) > 0 and points.shape[0] > int(args.pn_max_points):
                    sel = np.random.choice(points.shape[0], size=int(args.pn_max_points), replace=False)
                    points = points[sel]
                points_path = os.path.join(tmp_dir, 'points.npy')
                save_numpy(points_path, points)

                bbox_min = points.min(axis=0)
                bbox_max = points.max(axis=0)
                centroid = points.mean(axis=0)
                extents = bbox_max - bbox_min
                _log(f"[run {run_idx}] points stats: N={points.shape[0]}, centroid={centroid.tolist()} extents={extents.tolist()}")

                # 1) First-stage inference via persistent server
                first_out_dir = os.path.join(tmp_dir, 'first')
                ensure_dir(first_out_dir, clean=True)
                _log(f"[run {run_idx}] [1/6] First-stage inference ...")
                t0 = time.time()
                _ = client.infer_first(points_path, first_out_dir)
                _log(f"[run {run_idx}] First-stage finished in {time.time() - t0:.2f}s")
                scores_first = np.load(os.path.join(first_out_dir, 'scores_first.npy'))
                _log(f"[run {run_idx}] scores_first range: [{float(scores_first.min()):.6f}, {float(scores_first.max()):.6f}]")

                # 2) Sample left grasp point
                idx_left = softmax_topk_sample(points, scores_first, top_k=args.top_k, temperature=args.temperature)
                p_left = points[idx_left]
                p_left_path = os.path.join(tmp_dir, 'p_left.npy')
                save_numpy(p_left_path, p_left)
                _log(f"[run {run_idx}] [2/6] Sampled left index={idx_left} point={p_left.tolist()} (top_k={args.top_k}, temp={args.temperature})")
                write_affordance_html(points, scores_first, p_left, os.path.join(vis_dir, 'first_stage.html'), 'First-stage scores and sampled point')

                # 3) Second-stage inference (optional), then apply radial penalty
                if args.second_ckpt is not None and len(str(args.second_ckpt)) > 0 and os.path.isfile(os.path.abspath(args.second_ckpt)):
                    second_out_dir = os.path.join(tmp_dir, 'second')
                    ensure_dir(second_out_dir, clean=True)
                    _log(f"[run {run_idx}] [3/6] Second-stage inference ...")
                    t1 = time.time()
                    _ = client.infer_second(points_path, p_left_path, second_out_dir, condition_mode=args.condition_mode, use_condition=bool(args.use_condition))
                    _log(f"[run {run_idx}] Second-stage finished in {time.time() - t1:.2f}s")
                    scores_second = np.load(os.path.join(second_out_dir, 'scores_second.npy'))
                else:
                    _log(f"[run {run_idx}] [3/6] No valid second_ckpt provided. Falling back to first-stage scores for right hand.")
                    scores_second = scores_first.copy()
                scores_second = apply_radial_penalty(points, scores_second, center=p_left, sigma=args.mask_sigma, min_dist=args.min_pair_dist)
                _log(f"[run {run_idx}] Applied radial penalty (min_pair_dist={args.min_pair_dist}, sigma={args.mask_sigma}). New range: [{float(scores_second.min()):.6f}, {float(scores_second.max()):.6f}]")

                # 4) Sample right grasp point with avoidance
                idx_right = softmax_topk_sample(points, scores_second, top_k=max(args.top_k, 2 * args.top_k), temperature=args.temperature, avoid_indices=[idx_left], avoid_radius=args.min_pair_dist)
                p_right = points[idx_right]
                p_right_path = os.path.join(tmp_dir, 'p_right.npy')
                save_numpy(p_right_path, p_right)
                _log(f"[run {run_idx}] [4/6] Sampled right index={idx_right} point={p_right.tolist()} (avoid left, r={args.min_pair_dist})")
                write_affordance_html(points, scores_second, p_right, os.path.join(vis_dir, 'second_stage.html'), 'Second-stage scores and sampled point')

                # 4.5) Pose generation (DexGraspNet2)
                pose_npz_path = None
                if args.pose_yaml is not None and len(str(args.pose_yaml)) > 0 and \
                   args.pose_ckpt is not None and len(str(args.pose_ckpt)) > 0 and \
                   os.path.isfile(os.path.abspath(args.pose_yaml)) and os.path.isfile(os.path.abspath(args.pose_ckpt)):
                    pose_out_dir = os.path.join(tmp_dir, 'pose_gen')
                    ensure_dir(pose_out_dir, clean=True)
                    pose_npz_path = os.path.join(pose_out_dir, 'pred_pose.npz')
                    pose_script = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'third_party', 'DexGraspNet2', 'src', 'infer_bimanual.py')
                    cmd_pose = [
                        args.conda_exe, 'run', '-n', args.pose_env, '--no-capture-output',
                        'python', pose_script,
                        '--yaml', os.path.abspath(args.pose_yaml),
                        '--ckpt', os.path.abspath(args.pose_ckpt),
                        '--pc', os.path.abspath(points_path),
                        '--kpleft', os.path.abspath(p_left_path),
                        '--kpright', os.path.abspath(p_right_path),
                        '--device', args.device,
                        '--save', os.path.abspath(pose_npz_path),
                    ]
                    if args.pose_backbone is not None and len(str(args.pose_backbone)) > 0:
                        cmd_pose.extend(['--backbone', args.pose_backbone])
                    _log(f"[run {run_idx}] [4.5/6] Launching pose generator (DexGraspNet2) ...")
                    t_pose = time.time()
                    run_subprocess(cmd_pose, cwd=os.path.abspath(os.path.dirname(__file__)), timeout=args.timeout_sec, log_path=os.path.join(logs_dir, f'pose_gen_{run_idx:03d}.txt'))
                    _log(f"[run {run_idx}] Pose generator finished in {time.time() - t_pose:.2f}s. Outputs: {pose_npz_path}")
                else:
                    _log(f"[run {run_idx}] [4.5/6] Pose generator skipped (pose_yaml/pose_ckpt not provided or invalid).")

                # 5) Approximate initial pose info for logging
                n_l = p_left - centroid
                n_r = p_right - centroid
                def _safe_dir(v: np.ndarray) -> List[float]:
                    n = np.linalg.norm(v) + 1e-12
                    return (v / n).astype(float).tolist()
                T_left0 = {'position': (p_left + args.expand_dist_left * (n_l / (np.linalg.norm(n_l) + 1e-12))).astype(float).tolist(), 'approach_dir': _safe_dir(-n_l)}
                T_right0 = {'position': (p_right + args.expand_dist_right * (n_r / (np.linalg.norm(n_r) + 1e-12))).astype(float).tolist(), 'approach_dir': _safe_dir(-n_r)}
                with open(os.path.join(out_dir, 'T_left0.json'), 'w') as f:
                    json.dump(T_left0, f, indent=2)
                with open(os.path.join(out_dir, 'T_right0.json'), 'w') as f:
                    json.dump(T_right0, f, indent=2)

                # 6) Run BimanGrasp optimization
                lx, ly, lz = [float(x) for x in p_left.tolist()]
                rx, ry, rz = [float(x) for x in p_right.tolist()]
                bim_main = os.path.abspath(os.path.join(os.path.dirname(__file__), 'third_party', 'BimanGrasp-Generation', 'BimanGrasp-Optimization', 'main.py'))
                bim_wrapper = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tools', 'run_biman_with_init.py'))
                use_wrapper = (args.pose_yaml is not None and args.pose_ckpt is not None and os.path.isfile(os.path.abspath(args.pose_yaml)) and os.path.isfile(os.path.abspath(args.pose_ckpt))) and os.path.isfile(os.path.join(tmp_dir, 'pose_gen', 'pred_pose.npz'))
                if use_wrapper:
                    pose_npz_path = os.path.join(tmp_dir, 'pose_gen', 'pred_pose.npz')
                    cmd_bim = [
                        args.conda_exe, 'run', '-n', args.bim_env, '--no-capture-output',
                        'python', bim_wrapper,
                        '--name', args.exp_name,
                        '--gpu', str(args.gpu),
                        '--num_iterations', str(args.num_iterations),
                        '--init_pose_npz', os.path.abspath(pose_npz_path),
                        '--left_translation', f"{lx:.6f} {ly:.6f} {lz:.6f}",
                        '--right_translation', f"{rx:.6f} {ry:.6f} {rz:.6f}",
                    ]
                    if args.metrics:
                        cmd_bim.append('--metrics')
                    if args.vis:
                        cmd_bim.extend(['--vis', '--vis_frame_stride', str(args.vis_frame_stride)])
                    obj_code_for_bim = args.object_code if args.object_code is not None else selected_code
                    if obj_code_for_bim is not None and len(obj_code_for_bim) > 0:
                        cmd_bim.extend(['--object_code', obj_code_for_bim])
                else:
                    cmd_bim = [
                        args.conda_exe, 'run', '-n', args.bim_env, '--no-capture-output',
                        'python', bim_main,
                        '--name', args.exp_name,
                        '--gpu', str(args.gpu),
                        '--num_iterations', str(args.num_iterations),
                        '--init_at_targets',
                        '--left_target', f"{lx:.6f} {ly:.6f} {lz:.6f}",
                        '--right_target', f"{rx:.6f} {ry:.6f} {rz:.6f}",
                        '--snap_to_surface',
                    ]
                    if args.metrics:
                        cmd_bim.append('--metrics')
                    if args.vis:
                        cmd_bim.extend(['--vis', '--vis_frame_stride', str(args.vis_frame_stride)])
                    obj_code_for_bim = args.object_code if args.object_code is not None else selected_code
                    if obj_code_for_bim is not None and len(obj_code_for_bim) > 0:
                        cmd_bim.extend(['--object_code', obj_code_for_bim])

                bim_cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), 'third_party', 'BimanGrasp-Generation', 'BimanGrasp-Optimization'))
                _log(f"[run {run_idx}] [5/6] Launching BimanGrasp ({'wrapper' if use_wrapper else 'vanilla'}) ... cwd={bim_cwd}")
                t2 = time.time()
                run_subprocess(cmd_bim, cwd=bim_cwd, timeout=args.timeout_sec, log_path=os.path.join(logs_dir, f'bimangrasp_{run_idx:03d}.txt'))
                _log(f"[run {run_idx}] BimanGrasp finished in {time.time() - t2:.2f}s")

                # Copy optimization videos
                try:
                    bim_results_dir = os.path.abspath(os.path.join(os.path.dirname(bim_main), '..', 'data', 'experiments', args.exp_name, 'results'))
                    if os.path.isdir(bim_results_dir):
                        for fn in os.listdir(bim_results_dir):
                            if fn.lower().endswith('.mp4'):
                                src = os.path.join(bim_results_dir, fn)
                                dst = os.path.join(vis_dir, f'bimangrasp_{fn}')
                                shutil.copy2(src, dst)
                        _log(f"[run {run_idx}] Copied optimization video(s) from {bim_results_dir} to {vis_dir}")
                except Exception:
                    pass

                # Save manifest
                manifest = {
                    'exp_name': args.exp_name,
                    'points_path': resolved_points,
                    'object_code': (selected_code if selected_code is not None else args.object_code),
                    'first_ckpt': os.path.abspath(args.first_ckpt),
                    'second_ckpt': (os.path.abspath(args.second_ckpt) if args.second_ckpt else None),
                    'p_left': [float(x) for x in p_left.tolist()],
                    'p_right': [float(x) for x in p_right.tolist()],
                    'pose_gen': {
                        'yaml': (os.path.abspath(args.pose_yaml) if getattr(args, 'pose_yaml', None) else None),
                        'ckpt': (os.path.abspath(args.pose_ckpt) if getattr(args, 'pose_ckpt', None) else None),
                        'env': getattr(args, 'pose_env', None),
                        'backbone': getattr(args, 'pose_backbone', None),
                    },
                    'sampler': {
                        'top_k': int(args.top_k),
                        'temperature': float(args.temperature),
                        'min_pair_dist': float(args.min_pair_dist),
                        'mask_sigma': float(args.mask_sigma),
                    },
                    'envs': {'pn': args.pn_env, 'bimangrasp': args.bim_env, 'conda_exe': args.conda_exe},
                    'pn_max_points': (int(args.pn_max_points) if getattr(args, 'pn_max_points', None) is not None else None),
                }
                with open(os.path.join(out_dir, 'pipeline_manifest.json'), 'w') as f:
                    json.dump(manifest, f, indent=2)
                _log(f"[run {run_idx}] [6/6] Run complete. Manifest: {os.path.join(out_dir, 'pipeline_manifest.json')}")
                _log(f"[run {run_idx}] Visualizations: {os.path.join(vis_dir, 'first_stage.html')} , {os.path.join(vis_dir, 'second_stage.html')}")

            except Exception as e:
                with open(os.path.join(out_dir, 'FAILED.txt'), 'w') as f:
                    f.write(str(e))
                print(f"[pipeline][interactive][ERROR] {e}. See FAILED.txt and logs in {logs_dir}", flush=True)
            finally:
                # Clean tmp per-run to save space
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass
            run_idx += 1

    finally:
        client.shutdown()

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='End-to-end bimanual grasp pipeline orchestrator')
    # Inputs (interactive mode does not require --points_path)
    p.add_argument('--first_ckpt', type=str, required=True, help='Path to first-stage checkpoint (.pth)')
    p.add_argument('--second_ckpt', type=str, required=True, help='Path to second-stage checkpoint (.pth)')
    p.add_argument('--out_dir', type=str, required=True, help='Output directory for pipeline artifacts')
    p.add_argument('--input_root', type=str, default=None, help='Root directory containing <object_code>/obj_points.npy folders for interactive mode')
    p.add_argument('--points_filename', type=str, default='obj_points.npy', help='Points filename to load under each object_code directory')

    # Sampling
    p.add_argument('--top_k', type=int, default=512)
    p.add_argument('--temperature', type=float, default=0.3)
    p.add_argument('--min_pair_dist', type=float, default=0.05)
    p.add_argument('--mask_sigma', type=float, default=0.03)

    # Second-stage conditioning
    p.add_argument('--condition_mode', type=str, default='rd', choices=['none', 'r', 'rd', 'kp'])
    p.add_argument('--use_condition', action='store_true', default=True)

    # Init pose logging (not directly used by BimanGrasp call)
    p.add_argument('--expand_dist_left', type=float, default=0.08)
    p.add_argument('--expand_dist_right', type=float, default=0.08)

    # Env orchestration
    p.add_argument('--pn_env', type=str, default='pn')
    p.add_argument('--bim_env', type=str, default='bimangrasp')
    p.add_argument('--conda_exe', type=str, default='conda')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--timeout_sec', type=int, default=7200)
    p.add_argument('--pn_max_points', type=int, default=8192, help='Downsample point cloud to at most this number for PN inference')

    # BimanGrasp
    p.add_argument('--exp_name', type=str, default='pipeline_exp')
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--num_iterations', type=int, default=10000)
    p.add_argument('--metrics', action='store_true', default=True)
    p.add_argument('--vis', action='store_true', default=False)
    p.add_argument('--vis_frame_stride', type=int, default=50)
    p.add_argument('--object_code', type=str, default=None, help='Optional object code for BimanGrasp')
    p.add_argument('--quiet', action='store_true', help='Suppress step-by-step console logs')

    # Pose generator (DexGraspNet2)
    p.add_argument('--pose_yaml', type=str, default=None, help='Path to DexGraspNet2 config yaml')
    p.add_argument('--pose_ckpt', type=str, default=None, help='Path to DexGraspNet2 checkpoint (.pt/.pth)')
    p.add_argument('--pose_backbone', type=str, default=None, help='Override backbone: pointnet2 | sparseconv | sparse_glob_conv')
    p.add_argument('--pose_env', type=str, default='dex', help='Conda env name for pose generator')

    # Interactive mode (preload PN models and process multiple point clouds)
    p.add_argument('--interactive', action='store_true', help='Load affordance models once and enter an interactive loop to process multiple point clouds')

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if getattr(args, 'interactive', False):
        orchestrate_interactive(args)
    else:
        orchestrate(args)


if __name__ == '__main__':
    main()



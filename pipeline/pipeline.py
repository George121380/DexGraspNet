import os
import sys
import argparse
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CUR_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.utils.run_utils import load_yaml, resolve_config_paths, make_session_dirs, setup_logger, time_block, run_subprocess
from pipeline.utils.io import load_point_cloud, save_json, save_npy, snapshot_config
from pipeline.utils.viz import (
    write_pointcloud_with_values_html,
    write_values_histogram_html,
    build_bimanual_entry,
    save_bimanual_entry,
    load_default_scale,
)
from pipeline.utils.pose_generator import run_dexgrasp

import torch

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def _serialize_keypoint(kp: Dict[str, object]) -> Dict[str, object]:
    xyz = np.asarray(kp["xyz"], dtype=np.float32)
    return {"index": int(kp["index"]), "xyz": xyz.tolist()}


def _render_bimanual_viz(
    entry_path: str,
    object_code: str,
    out_html: str,
    vis_cfg: Dict[str, object],
    env_name: str,
    pk_root: str,
    logs_dir: str,
    logger,
    baseline_entry: Optional[str] = None,
    kpleft_path: Optional[str] = None,
    kpright_path: Optional[str] = None,
    points_path: Optional[str] = None,
) -> None:
    if not vis_cfg.get('enable', True):
        return

    script = os.path.join(REPO_ROOT, 'pipeline', 'utils', 'render_bimanual.py')
    cmd = [
        'conda', 'run', '-n', env_name, 'python', script,
        '--entry', entry_path,
        '--object_code', object_code,
        '--html', out_html,
        '--vis_root', vis_cfg['root'],
        '--mesh_root', vis_cfg['object_mesh_root'],
        '--background', vis_cfg.get('background_color', '#E2F0D9'),
        '--pk_root', pk_root,
    ]
    if 'ref_scale_dir' in vis_cfg:
        cmd += ['--ref_scale_dir', vis_cfg['ref_scale_dir']]
    if vis_cfg.get('debug', False):
        cmd += ['--debug']
        if vis_cfg.get('palm_center_offset') is not None:
            cmd += ['--palm_offset', str(float(vis_cfg['palm_center_offset']))]
        if vis_cfg.get('palm_normal_len') is not None:
            cmd += ['--palm_normal_len', str(float(vis_cfg['palm_normal_len']))]
        if kpleft_path and os.path.exists(kpleft_path):
            cmd += ['--kpleft', kpleft_path]
        if kpright_path and os.path.exists(kpright_path):
            cmd += ['--kpright', kpright_path]
        if points_path and os.path.exists(points_path):
            cmd += ['--points', points_path]
    if baseline_entry:
        cmd += ['--baseline', baseline_entry]

    base = os.path.splitext(os.path.basename(out_html))[0]
    stdout_path = os.path.join(logs_dir, f"{base}_stdout.txt")
    stderr_path = os.path.join(logs_dir, f"{base}_stderr.txt")
    rc, _, _ = run_subprocess(cmd, stdout_path=stdout_path, stderr_path=stderr_path, logger=logger)
    if rc != 0:
        raise RuntimeError(f"Bimanual visualization failed for {out_html}: rc={rc}")


def _load_all_configs(pipeline_cfg: str, aff1_cfg: str, aff2_cfg: str, dex_cfg: str, opt_cfg: str) -> Dict:
    base = load_yaml(pipeline_cfg)
    base = resolve_config_paths(base)
    # Merge others under keys
    def _with_base_paths(cfg_child: Dict) -> Dict:
        cfg_child = dict(cfg_child or {})
        base_paths = dict(base.get("paths", {}))
        child_paths = dict(cfg_child.get("paths", {}))
        merged_paths = {**base_paths, **child_paths}
        cfg_child['paths'] = merged_paths
        return resolve_config_paths(cfg_child)

    base['aff1'] = _with_base_paths(load_yaml(aff1_cfg))
    base['aff2'] = _with_base_paths(load_yaml(aff2_cfg))
    base['dex'] = _with_base_paths(load_yaml(dex_cfg))
    base['opt'] = _with_base_paths(load_yaml(opt_cfg))
    return base


def process_object(obj_name: str, cfg: Dict, session_dirs: Dict[str, str], aff1, aff2, logger) -> None:
    repo_root = cfg['paths']['repo_root']
    input_root = cfg['paths']['input_root']
    outputs_root = session_dirs['object']
    logs_dir = session_dirs['logs']

    obj_dir = os.path.join(outputs_root, obj_name)
    os.makedirs(obj_dir, exist_ok=True)

    logger.info(f"[OBJ] Processing object: {obj_name}")
    points_path = os.path.join(input_root, obj_name, 'obj_points.npy')
    logger.info(f"Loading point cloud: {points_path}")
    points = load_point_cloud(points_path)
    logger.info(f"Point cloud loaded: shape={points.shape}, dtype={points.dtype}")
    save_npy(os.path.join(obj_dir, 'points.npy'), points)

    # Centralized multipose controls (fallback to sub-configs if absent)
    mp_cfg = dict(cfg.get('multipose', {}))
    sample_by_value = bool(mp_cfg.get('sample_by_value', False))

    # Step 2: Affordance First → multiple kp1 candidates
    with time_block("Affordance First", logger):
        logger.info("Running affordance-first inference ...")
        aff1_scores = aff1.predict(points)
        logger.info(f"Aff1 scores: min={float(np.min(aff1_scores)):.4f}, max={float(np.max(aff1_scores)):.4f}")
        n_kp1 = int(mp_cfg.get('kp1_samples', cfg['aff1']['sampling'].get('num_samples', 1)))
        kp1_list = []
        if sample_by_value:
            # Softmax sampling without replacement
            N = points.shape[0]
            temp1 = float(mp_cfg.get('temperature_kp1', cfg['aff1']['sampling'].get('temperature', 0.1)))
            s = aff1_scores.reshape(N)
            # bottom-quantile filtering
            q1 = float(mp_cfg.get('min_quantile_kp1', 0.0))
            if q1 > 0.0:
                th = float(np.quantile(s, min(max(q1, 0.0), 0.99)))
                keep = s >= th
            else:
                keep = np.ones_like(s, dtype=bool)
            idxs = np.arange(N)[keep]
            s_keep = s[keep]
            if idxs.size == 0:
                idxs = np.arange(N)
                s_keep = s
            # normalize to [0,1]
            s_min, s_max = float(s_keep.min()), float(s_keep.max())
            denom = (s_max - s_min) + 1e-8
            s_norm = (s_keep - s_min) / denom
            if not np.isfinite(s_norm).all() or s_norm.sum() <= 1e-12:
                s_norm = np.ones_like(s_keep) / max(1, s_keep.size)
            logits = s_norm / max(1e-6, temp1)
            logits = logits - float(np.max(logits))
            probs = np.exp(logits)
            Z = float(np.sum(probs)) + 1e-12
            probs = probs / Z
            k = min(n_kp1, idxs.size)
            sel_local = np.random.choice(np.arange(idxs.size), size=k, replace=False, p=probs)
            sel = idxs[sel_local]
            for t, idx in enumerate(sel):
                kp1_serial = {"index": int(idx), "xyz": points[int(idx)].astype(np.float32).tolist()}
                kp1_list.append(kp1_serial)
                save_json(os.path.join(obj_dir, f"kp1_{t:02d}.json"), kp1_serial)
        else:
            used_idx_1 = set()
            max_trials = max(10, n_kp1 * 10)
            trials = 0
            while len(kp1_list) < n_kp1 and trials < max_trials:
                trials += 1
                kp1 = aff1.sample_keypoint(
                    points, aff1_scores,
                    strategy=cfg['aff1']['sampling']['strategy'],
                    top_k=cfg['aff1']['sampling']['top_k'],
                    nms_radius=cfg['aff1']['sampling']['nms_radius'],
                    temperature=cfg['aff1']['sampling']['temperature']
                )
                if int(kp1['index']) in used_idx_1:
                    continue
                kp1_serial = _serialize_keypoint(kp1)
                kp1_list.append(kp1_serial)
                used_idx_1.add(int(kp1['index']))
                save_json(os.path.join(obj_dir, f"kp1_{len(kp1_list)-1:02d}.json"), kp1_serial)
        np.save(os.path.join(obj_dir, 'aff1.npy'), aff1_scores)
        # Distribution plot for aff1
        try:
            write_values_histogram_html(aff1_scores, os.path.join(obj_dir, 'aff1_hist.html'), title='Affordance First - Scores Distribution')
            logger.info("Saved histogram: aff1_hist.html")
        except Exception:
            logger.warning("Failed to write aff1 histogram")
        if len(kp1_list) == 0:
            raise RuntimeError("Affordance First sampling produced no keypoints")
        logger.info(f"Sampled {len(kp1_list)} kp1 candidates")
        # Keep the original step2 visualization for the first candidate to limit outputs
        if cfg['visualization']['enable']:
            first = kp1_list[0]
            write_pointcloud_with_values_html(points, aff1_scores, [(first['index'], np.array(first['xyz']))], os.path.join(obj_dir, 'viz_step2.html'), title='Affordance First')
            logger.info("Saved visualization: viz_step2.html")

    # Step 3: Affordance Second → multiple kp2 per kp1
    with time_block("Affordance Second", logger):
        logger.info("Running affordance-second inference ...")
        n_kp2 = int(mp_cfg.get('kp2_samples_per_kp1', cfg['aff2']['sampling'].get('num_samples', 1)))
        # For visualization, we will save only the first pair (kp1_00, kp2_00)
        viz3_done = False
        for i, kp1_serial in enumerate(kp1_list):
            kp1_xyz = np.array(kp1_serial['xyz'], dtype=np.float32)
            aff2_scores = aff2.predict(points, kp1_xyz)
            logger.info(f"[kp1 {i}] Aff2 scores: min={float(np.min(aff2_scores)):.4f}, max={float(np.max(aff2_scores)):.4f}")
            # save aff2 scores optionally per kp1
            np.save(os.path.join(obj_dir, f'aff2_{i:02d}.npy'), aff2_scores)
            # Distribution plot for aff2
            try:
                write_values_histogram_html(aff2_scores, os.path.join(obj_dir, f'aff2_hist_{i:02d}.html'), title=f'Affordance Second (kp1 {i}) - Scores Distribution')
                logger.info(f"Saved histogram: aff2_hist_{i:02d}.html")
            except Exception:
                logger.warning(f"Failed to write aff2 histogram for kp1 {i}")
            N = points.shape[0]
            kp2_list_i = []
            if sample_by_value:
                temp2 = float(mp_cfg.get('temperature_kp2', cfg['aff2']['sampling'].get('temperature', 0.1)))
                s = aff2_scores.reshape(N)
                # bottom-quantile filtering
                q2 = float(mp_cfg.get('min_quantile_kp2', 0.0))
                if q2 > 0.0:
                    th = float(np.quantile(s, min(max(q2, 0.0), 0.99)))
                    keep = s >= th
                else:
                    keep = np.ones_like(s, dtype=bool)
                idxs = np.arange(N)[keep]
                s_keep = s[keep]
                if idxs.size == 0:
                    idxs = np.arange(N)
                    s_keep = s
                s_min, s_max = float(s_keep.min()), float(s_keep.max())
                denom = (s_max - s_min) + 1e-8
                s_norm = (s_keep - s_min) / denom
                if not np.isfinite(s_norm).all() or s_norm.sum() <= 1e-12:
                    s_norm = np.ones_like(s_keep) / max(1, s_keep.size)
                logits = s_norm / max(1e-6, temp2)
                logits = logits - float(np.max(logits))
                probs = np.exp(logits)
                Z = float(np.sum(probs)) + 1e-12
                probs = probs / Z
                k = min(n_kp2, idxs.size)
                sel_local = np.random.choice(np.arange(idxs.size), size=k, replace=False, p=probs)
                sel = idxs[sel_local]
                for j_idx, idx in enumerate(sel):
                    kp2_ser = {"index": int(idx), "xyz": points[int(idx)].astype(np.float32).tolist()}
                    kp2_list_i.append(kp2_ser)
                    save_json(os.path.join(obj_dir, f"kp2_{i:02d}_{j_idx:02d}.json"), kp2_ser)
                    if cfg['visualization']['enable'] and not viz3_done and i == 0:
                        write_pointcloud_with_values_html(
                            points, aff2_scores,
                            [(kp1_serial['index'], np.array(kp1_serial['xyz'])),
                             (kp2_ser['index'], np.array(kp2_ser['xyz']))],
                            os.path.join(obj_dir, 'viz_step3.html'), title='Affordance Second')
                        viz3_done = True
            else:
                # Deterministic top-n selection for kp2 to avoid duplicates
                top_k = int(cfg['aff2']['sampling'].get('top_k', N))
                top_k = max(1, min(top_k, N))
                order = np.argsort(aff2_scores.reshape(N))[::-1]
                top_candidates = order[:top_k]
                selected = top_candidates[:min(n_kp2, len(top_candidates))]
                for j_idx, idx in enumerate(selected):
                    kp2_ser = {"index": int(idx), "xyz": points[int(idx)].astype(np.float32).tolist()}
                    kp2_list_i.append(kp2_ser)
                    save_json(os.path.join(obj_dir, f"kp2_{i:02d}_{j_idx:02d}.json"), kp2_ser)
                    if cfg['visualization']['enable'] and not viz3_done and i == 0:
                        write_pointcloud_with_values_html(
                            points, aff2_scores,
                            [(kp1_serial['index'], np.array(kp1_serial['xyz'])),
                             (kp2_ser['index'], np.array(kp2_ser['xyz']))],
                            os.path.join(obj_dir, 'viz_step3.html'), title='Affordance Second')
                        viz3_done = True
            # attach kp2 list back to kp1 item for downstream loops
            kp1_serial['kp2_list'] = kp2_list_i
        logger.info(f"Prepared kp2 candidates for {len(kp1_list)} kp1s (each up to {n_kp2})")

    vis_cfg = cfg['dex'].get('visualization', {})
    pk_root = os.path.join(cfg['paths']['third_party']['bimangrasp'], 'thirdparty')

    # Step 4: Pose Initialization (DexGrasp or Keypoint init) via conda run
    with time_block("DexGrasp Pose Generation", logger):
        init_cfg = cfg['dex']['dexgrasp'].get('init', {})
        random_twist_count = int(mp_cfg.get('twist_count_per_pair', init_cfg.get('random_twist_count', 1)))
        # Iterate over all kp1/kp2 combinations and twists
        for i, kp1_serial in enumerate(kp1_list):
            left_kp_path = os.path.join(obj_dir, f'kpleft_{i:02d}.npy')
            np.save(left_kp_path, np.array(kp1_serial['xyz'], dtype=np.float32))
            kp2_list = kp1_serial.get('kp2_list', [])
            for j, kp2_serial in enumerate(kp2_list):
                right_kp_path = os.path.join(obj_dir, f'kpright_{i:02d}_{j:02d}.npy')
                np.save(right_kp_path, np.array(kp2_serial['xyz'], dtype=np.float32))
                logger.info(f"Saved keypoints: {left_kp_path}, {right_kp_path}")

                for k in range(max(1, random_twist_count)):
                    suffix = f"{i:02d}_{j:02d}_{k:02d}"
                    dex_entry_path = os.path.join(obj_dir, f'dexgrasp_entry_{suffix}.npy')
                    if init_cfg.get('mode', 'dexgrasp') == 'keypoint':
                        logger.info(f"[init] keypoint-based init for pair {suffix} ...")
                        script = os.path.join(repo_root, 'pipeline', 'utils', 'init_from_keypoints.py')
                        cmd = [
                            'conda', 'run', '-n', cfg['envs']['optimizer_env'], 'python', script,
                            '--points', os.path.join(obj_dir, 'points.npy'),
                            '--kpleft', left_kp_path,
                            '--kpright', right_kp_path,
                            '--biman_root', cfg['paths']['third_party']['bimangrasp'],
                            '--offset', str(float(init_cfg.get('kp_offset', 0.04))),
                            '--out_entry', dex_entry_path,
                        ]
                        if bool(init_cfg.get('random_twist', True)):
                            cmd.append('--random_twist')
                        # deterministic but unique seed per combo
                        seed_val = (abs(hash((obj_name, i, j, k))) % (2**31))
                        cmd += ['--seed', str(int(seed_val))]
                        stdout_path = os.path.join(logs_dir, f'init_kp_{suffix}_stdout.txt')
                        stderr_path = os.path.join(logs_dir, f'init_kp_{suffix}_stderr.txt')
                        rc, _, _ = run_subprocess(cmd, stdout_path=stdout_path, stderr_path=stderr_path, logger=logger)
                        if rc != 0:
                            raise RuntimeError(f"Keypoint init failed for {suffix}: rc={rc}")
                        scale = load_default_scale(
                            obj_name,
                            vis_cfg.get('ref_scale_dir', ''),
                            vis_cfg.get('ref_scale_file', ''),
                            vis_cfg.get('ref_scale_value', None),
                        )
                        dex_npz_path = None
                        # Load entry back for later baseline/step5 and stamp scale
                        try:
                            dex_arr = np.load(dex_entry_path, allow_pickle=True)
                            dex_entry = dex_arr[0].item() if hasattr(dex_arr[0], 'item') else dex_arr[0]
                            dex_entry['scale'] = float(scale)
                            save_bimanual_entry(dex_entry_path, dex_entry)
                        except Exception:
                            dex_entry = {'qpos_left': {}, 'qpos_right': {}, 'scale': float(scale)}
                    else:
                        # Single-run DexGrasp path (not expanded for multiple pairs to avoid heavy runtime)
                        if i > 0 or j > 0 or k > 0:
                            continue
                        dex_npz_path = os.path.join(obj_dir, cfg['dex']['io']['save_npz_name'])
                        logger.info("Launching DexGrasp subprocess ...")
                        run_dexgrasp(
                            conda_env=cfg['envs']['pose_gen_env'],
                            repo_root=repo_root,
                            dex_yaml=cfg['dex']['dexgrasp']['yaml_config'],
                            dex_ckpt=cfg['dex']['dexgrasp']['checkpoint'],
                            points_npy=os.path.join(obj_dir, 'points.npy'),
                            kpleft_npy=left_kp_path,
                            kpright_npy=right_kp_path,
                            save_npz_path=dex_npz_path,
                            device=cfg['dex']['dexgrasp']['device'],
                            backbone=cfg['dex']['dexgrasp']['backbone'],
                            logs_dir=logs_dir,
                        )
                        logger.info(f"DexGrasp npz saved: {dex_npz_path}")
                        dex = np.load(dex_npz_path)
                        left_vec = dex['left'][0]
                        right_vec = dex['right'][0]
                        logger.info(f"DexGrasp output: left.shape={left_vec.shape}, right.shape={right_vec.shape}")
                        scale = load_default_scale(
                            obj_name,
                            vis_cfg.get('ref_scale_dir', ''),
                            vis_cfg.get('ref_scale_file', ''),
                            vis_cfg.get('ref_scale_value', None),
                        )
                        dex_entry = build_bimanual_entry(left_vec, right_vec, scale)
                        save_bimanual_entry(dex_entry_path, dex_entry)

                    # Viz step4 per combo (guarded by global visualization.enable)
                    if cfg['visualization']['enable']:
                        _render_bimanual_viz(
                            dex_entry_path,
                            obj_name,
                            os.path.join(obj_dir, f'viz_step4_{suffix}.html'),
                            vis_cfg,
                            cfg['envs']['optimizer_env'],
                            pk_root,
                            logs_dir,
                            logger,
                            kpleft_path=left_kp_path,
                            kpright_path=right_kp_path,
                            points_path=os.path.join(obj_dir, 'points.npy'),
                        )
                        logger.info(f"Saved visualization: viz_step4_{suffix}.html")

    # Step 5: Optimization via BimanGrasp (iterate all initialized entries)
    with time_block("BimanGrasp Optimization", logger):
        obj_map: Dict[str, str] = cfg['opt']['objects']['object_code_map'] or {}
        object_code = obj_map.get(obj_name, obj_name)
        logger.info(f"Optimizer object_code: {object_code}")

        script = os.path.join(repo_root, 'pipeline', 'utils', 'pose_optimizer.py')
        total_steps = int(cfg['optimizer']['steps']) if 'optimizer' in cfg and 'steps' in cfg['optimizer'] else int(cfg['opt']['optimizer']['steps']) if 'optimizer' in cfg['opt'] else 100

        # Collect all optimized entries to merge at the end
        opt_entries_all = []

        # Prepare all (i,j) pairs
        all_pairs = []
        for i, kp1_serial in enumerate(kp1_list):
            kp2_list = kp1_serial.get('kp2_list', [])
            for j, _ in enumerate(kp2_list):
                all_pairs.append((i, j))

        random_twist_count = int(mp_cfg.get('twist_count_per_pair', cfg['dex']['dexgrasp'].get('init', {}).get('random_twist_count', 1)))
        pairs_per_batch = max(1, int(mp_cfg.get('pairs_per_batch', 1)))
        max_parallel = max(1, int(mp_cfg.get('max_parallel', 1)))

        # Base env overrides
        base_env = os.environ.copy()
        energy_cfg = cfg['opt'].get('energy', {}) if 'opt' in cfg else {}
        if energy_cfg:
            if energy_cfg.get('w_dis') is not None:
                base_env['PIPELINE_W_DIS'] = str(float(energy_cfg['w_dis']))
            if energy_cfg.get('w_pen') is not None:
                base_env['PIPELINE_W_PEN'] = str(float(energy_cfg['w_pen']))
            if energy_cfg.get('w_spen') is not None:
                base_env['PIPELINE_W_SPEN'] = str(float(energy_cfg['w_spen']))
            if energy_cfg.get('w_joints') is not None:
                base_env['PIPELINE_W_JOINTS'] = str(float(energy_cfg['w_joints']))
            if energy_cfg.get('w_vew') is not None:
                base_env['PIPELINE_W_VEW'] = str(float(energy_cfg['w_vew']))
        opt_cfg_env = cfg['opt'].get('optimizer', {}) if 'opt' in cfg else {}
        if opt_cfg_env.get('noise_factor') is not None:
            base_env['PIPELINE_NOISE_FACTOR'] = str(float(opt_cfg_env['noise_factor']))

        # Consistent object scale
        try:
            obj_scale_val = load_default_scale(
                obj_name,
                vis_cfg.get('ref_scale_dir', ''),
                vis_cfg.get('ref_scale_file', ''),
                vis_cfg.get('ref_scale_value', None),
            )
        except Exception:
            obj_scale_val = None

        import subprocess, time

        for start in range(0, len(all_pairs), pairs_per_batch):
            batch = all_pairs[start:start + pairs_per_batch]
            logger.info(f"[Batch] Processing pairs {batch}")

            if mp_cfg.get('process_mode', 'multiprocess') == 'batched':
                # One process handling all entries in this batch
                entry_list = []
                suffixes = []
                for (i, j) in batch:
                    for k in range(max(1, random_twist_count)):
                        suffix = f"{i:02d}_{j:02d}_{k:02d}"
                        dex_entry_path = os.path.join(obj_dir, f'dexgrasp_entry_{suffix}.npy')
                        if not os.path.exists(dex_entry_path):
                            continue
                        entry_list.append(dex_entry_path)
                        suffixes.append(suffix)
                if not entry_list:
                    continue
                cmd = [
                    'conda', 'run', '-n', cfg['envs']['optimizer_env'], 'python', script,
                    '--biman_root', cfg['paths']['third_party']['bimangrasp'],
                    '--data_root', cfg['opt']['paths']['data_root'],
                    '--object_code', object_code,
                    '--steps', str(total_steps),
                    '--gpu', cfg['opt']['optimizer']['gpu'] if 'optimizer' in cfg['opt'] else '0',
                    '--out_dir', obj_dir,
                    '--suffixes', ','.join(suffixes),
                ]
                for pth in entry_list:
                    cmd.extend(['--entry_multi', pth])
                if obj_scale_val is not None:
                    cmd.extend(['--object_scale', str(float(obj_scale_val))])
                stdout_path = os.path.join(logs_dir, f'biman_opt_batch_{start:04d}_stdout.txt')
                stderr_path = os.path.join(logs_dir, f'biman_opt_batch_{start:04d}_stderr.txt')
                progress_file = os.path.join(logs_dir, f'biman_opt_progress_batch_{start:04d}.txt')
                cmd.extend(['--progress_file', progress_file])
                env = {**base_env, 'PIPELINE_PROGRESS_FILE': progress_file}
                logger.info(f"[Batch] Launching single-process batched optimizer for {len(entry_list)} entries ...")
                import subprocess
                with open(stdout_path, 'w') as f_out, open(stderr_path, 'w') as f_err:
                    p = subprocess.Popen(cmd, stdout=f_out, stderr=f_err, env=env, text=True)
                    last_step = 0
                    bar = tqdm(total=int(total_steps), desc=f"optimizing(batch {start//max(1,pairs_per_batch)})", dynamic_ncols=True)
                    try:
                        while True:
                            rc = p.poll()
                            try:
                                if os.path.exists(progress_file):
                                    with open(progress_file, 'r') as pf:
                                        s = pf.read().strip()
                                        if s.isdigit():
                                            cur = int(s)
                                            if cur > last_step:
                                                bar.update(min(cur, int(total_steps)) - last_step)
                                                last_step = cur
                            except Exception:
                                pass
                            if rc is not None:
                                break
                            time.sleep(0.2)
                    finally:
                        # Do not force-fill to total on early exit; keep real progress
                        bar.close()
                    rc = p.returncode
                if rc != 0:
                    raise RuntimeError(f"Batched optimization failed for batch start={start}: rc={rc}")
                # Post-process results per suffix
                for suffix in suffixes:
                    out_json = os.path.join(obj_dir, f"optimized_{suffix}.json")
                    dex_entry_path = os.path.join(obj_dir, f'dexgrasp_entry_{suffix}.npy')
                    left_kp_path = os.path.join(obj_dir, f'kpleft_{suffix[:2]}.npy')
                    right_kp_path = os.path.join(obj_dir, f'kpright_{suffix[:2]}_{suffix[3:5]}.npy')
                    if os.path.exists(out_json):
                        data = json_load(out_json)
                        try:
                            dex_arr = np.load(dex_entry_path, allow_pickle=True)
                            dex_entry = dex_arr[0].item() if hasattr(dex_arr[0], 'item') else dex_arr[0]
                            scale_val = float(dex_entry.get('scale', 1.0))
                            left_st = dex_entry.get('qpos_left', {})
                            right_st = dex_entry.get('qpos_right', {})
                        except Exception:
                            scale_val = float(load_default_scale(
                                obj_name,
                                vis_cfg.get('ref_scale_dir', ''),
                                vis_cfg.get('ref_scale_file', ''),
                                vis_cfg.get('ref_scale_value', None),
                            ))
                            left_st, right_st = {}, {}
                        opt_entry = build_bimanual_entry(
                            data['left_qpos'],
                            data['right_qpos'],
                            scale_val,
                            left_st=left_st,
                            right_st=right_st,
                        )
                        opt_entry_path = os.path.join(obj_dir, f'optimized_pose_{suffix}.npy')
                        save_bimanual_entry(opt_entry_path, opt_entry)
                        opt_entries_all.append(opt_entry)
                        if cfg['visualization']['enable']:
                            try:
                                _render_bimanual_viz(
                                    opt_entry_path,
                                    obj_name,
                                    os.path.join(obj_dir, f'viz_step5_{suffix}.html'),
                                    vis_cfg,
                                    cfg['envs']['optimizer_env'],
                                    pk_root,
                                    logs_dir,
                                    logger,
                                    baseline_entry=dex_entry_path,
                                    kpleft_path=left_kp_path,
                                    kpright_path=right_kp_path,
                                    points_path=os.path.join(obj_dir, 'points.npy'),
                                )
                                logger.info(f"Saved visualization: viz_step5_{suffix}.html")
                            except Exception:
                                logger.warning(f"Failed to render viz_step5 for {suffix}")
                continue

            # Multiprocess path (existing)
            tasks = []
            for (i, j) in batch:
                left_kp_path = os.path.join(obj_dir, f'kpleft_{i:02d}.npy')
                right_kp_path = os.path.join(obj_dir, f'kpright_{i:02d}_{j:02d}.npy')
                for k in range(max(1, random_twist_count)):
                    suffix = f"{i:02d}_{j:02d}_{k:02d}"
                    dex_entry_path = os.path.join(obj_dir, f'dexgrasp_entry_{suffix}.npy')
                    if not os.path.exists(dex_entry_path):
                        continue
                    out_npz = os.path.join(obj_dir, f"optimized_{suffix}.npz")
                    out_json = os.path.join(obj_dir, f"optimized_{suffix}.json")
                    cmd = [
                        'conda', 'run', '-n', cfg['envs']['optimizer_env'], 'python', script,
                        '--biman_root', cfg['paths']['third_party']['bimangrasp'],
                        '--data_root', cfg['opt']['paths']['data_root'],
                        '--object_code', object_code,
                        '--steps', str(total_steps),
                        '--gpu', cfg['opt']['optimizer']['gpu'] if 'optimizer' in cfg['opt'] else '0',
                        '--out_npz', out_npz,
                        '--out_json', out_json,
                    ]
                    if cfg['dex']['dexgrasp'].get('init', {}).get('mode', 'dexgrasp') == 'keypoint':
                        cmd.extend(['--entry', dex_entry_path])
                    else:
                        dex_npz_path = os.path.join(obj_dir, cfg['dex']['io']['save_npz_name'])
                        cmd.extend(['--dex_npz', dex_npz_path])
                    if obj_scale_val is not None:
                        cmd.extend(['--object_scale', str(float(obj_scale_val))])
                    opt_vis_cfg = cfg['opt'].get('visualization', {})
                    frames_dir = None
                    video_path = None
                    if opt_vis_cfg.get('enable', False):
                        frames_dir = os.path.join(obj_dir, f'optimization_frames_{suffix}')
                        video_path = os.path.join(obj_dir, f"optimization_{suffix}.mp4")
                        cmd.extend([
                            '--capture_video',
                            '--frames_dir', frames_dir,
                            '--video_path', video_path,
                            '--frame_stride', str(int(opt_vis_cfg.get('frame_stride', 10))),
                            '--video_width', str(int(opt_vis_cfg.get('width', 900))),
                            '--video_height', str(int(opt_vis_cfg.get('height', 900))),
                            '--video_fps', str(int(opt_vis_cfg.get('fps', 20))),
                            '--bg_color', opt_vis_cfg.get('background', '#E2F0D9'),
                        ])
                        if opt_vis_cfg.get('show_contacts', False):
                            cmd.append('--show_contacts')
                    stdout_path = os.path.join(logs_dir, f'biman_opt_{suffix}_stdout.txt')
                    stderr_path = os.path.join(logs_dir, f'biman_opt_{suffix}_stderr.txt')
                    progress_file = os.path.join(logs_dir, f'biman_opt_progress_{suffix}.txt')
                    cmd.extend(['--progress_file', progress_file])
                    env = {**base_env, 'PIPELINE_PROGRESS_FILE': progress_file}
                    tasks.append({
                        'suffix': suffix,
                        'cmd': cmd,
                        'env': env,
                        'stdout_path': stdout_path,
                        'stderr_path': stderr_path,
                        'dex_entry_path': dex_entry_path,
                        'left_kp_path': left_kp_path,
                        'right_kp_path': right_kp_path,
                        'out_json': out_json,
                        'video_path': video_path,
                    })

            running = []
            tasks_progress = tqdm(total=len(tasks), desc="optimizing(tasks)", dynamic_ncols=True)
            it = iter(tasks)

            def _start():
                try:
                    t = next(it)
                except StopIteration:
                    return False
                f_out = open(t['stdout_path'], 'w')
                f_err = open(t['stderr_path'], 'w')
                p = subprocess.Popen(t['cmd'], stdout=f_out, stderr=f_err, text=True, env=t['env'])
                t['proc'] = p
                t['f_out'] = f_out
                t['f_err'] = f_err
                running.append(t)
                logger.info(f"[Batch] Launched optimizer for {t['suffix']}")
                return True

            for _ in range(min(max_parallel, len(tasks))):
                _start()

            while running:
                for t in list(running):
                    rc = t['proc'].poll()
                    if rc is None:
                        continue
                    try:
                        t['f_out'].close(); t['f_err'].close()
                    except Exception:
                        pass
                    running.remove(t)
                    if rc != 0:
                        logger.error(f"Optimizer failed for {t['suffix']}: rc={rc}")
                    else:
                        logger.info(f"Optimizer done for {t['suffix']}")
                        if os.path.exists(t['out_json']):
                            data = json_load(t['out_json'])
                            try:
                                dex_arr = np.load(t['dex_entry_path'], allow_pickle=True)
                                dex_entry = dex_arr[0].item() if hasattr(dex_arr[0], 'item') else dex_arr[0]
                                scale_val = float(dex_entry.get('scale', 1.0))
                                left_st = dex_entry.get('qpos_left', {})
                                right_st = dex_entry.get('qpos_right', {})
                            except Exception:
                                scale_val = float(load_default_scale(
                                    obj_name,
                                    vis_cfg.get('ref_scale_dir', ''),
                                    vis_cfg.get('ref_scale_file', ''),
                                    vis_cfg.get('ref_scale_value', None),
                                ))
                                left_st, right_st = {}, {}
                            opt_entry = build_bimanual_entry(
                                data['left_qpos'],
                                data['right_qpos'],
                                scale_val,
                                left_st=left_st,
                                right_st=right_st,
                            )
                            opt_entry_path = os.path.join(obj_dir, f"optimized_pose_{t['suffix']}.npy")
                            save_bimanual_entry(opt_entry_path, opt_entry)
                            opt_entries_all.append(opt_entry)
                            if cfg['visualization']['enable']:
                                try:
                                    _render_bimanual_viz(
                                        opt_entry_path,
                                        obj_name,
                                        os.path.join(obj_dir, f"viz_step5_{t['suffix']}.html"),
                                        vis_cfg,
                                        cfg['envs']['optimizer_env'],
                                        pk_root,
                                        logs_dir,
                                        logger,
                                        baseline_entry=t['dex_entry_path'],
                                        kpleft_path=t['left_kp_path'],
                                        kpright_path=t['right_kp_path'],
                                        points_path=os.path.join(obj_dir, 'points.npy'),
                                    )
                                    logger.info(f"Saved visualization: viz_step5_{t['suffix']}.html")
                                except Exception:
                                    logger.warning(f"Failed to render viz_step5 for {t['suffix']}")
                    tasks_progress.update(1)
                    _start()
                time.sleep(0.5)
            tasks_progress.close()

        # Save merged npy for this object if any entries were generated
        if len(opt_entries_all) > 0:
            merged_path = os.path.join(obj_dir, f"{obj_name}.npy")
            try:
                np.save(merged_path, np.array(opt_entries_all, dtype=object))
                logger.info(f"Merged optimized poses saved: {merged_path} (count={len(opt_entries_all)})")
            except Exception as e:
                logger.exception(f"Failed to save merged poses to {merged_path}: {e}")


def json_load(path: str) -> Dict:
    import json
    with open(path, 'r') as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pipeline_cfg', type=str, default=os.path.join('pipeline', 'configs', 'pipeline.yaml'))
    ap.add_argument('--aff1_cfg', type=str, default=os.path.join('pipeline', 'configs', 'affordance_first.yaml'))
    ap.add_argument('--aff2_cfg', type=str, default=os.path.join('pipeline', 'configs', 'affordance_second.yaml'))
    ap.add_argument('--dex_cfg', type=str, default=os.path.join('pipeline', 'configs', 'dexgrasp.yaml'))
    ap.add_argument('--opt_cfg', type=str, default=os.path.join('pipeline', 'configs', 'optimizer.yaml'))
    ap.add_argument('--object', type=str, default=None, help='Run once for given object name')
    args = ap.parse_args()

    cfg = _load_all_configs(args.pipeline_cfg, args.aff1_cfg, args.aff2_cfg, args.dex_cfg, args.opt_cfg)
    repo_root = cfg['paths']['repo_root']
    outputs_root = cfg['paths']['outputs_root']

    # Prepare session and logger
    session_dirs = make_session_dirs(outputs_root)
    log_path = os.path.join(session_dirs['base'], 'pipeline.log')
    logger = setup_logger(log_path)
    logger.info(f"Session: {session_dirs['session_id']}")

    # Snapshot configs used
    snapshot_config(
        {
            'pipeline': os.path.abspath(args.pipeline_cfg),
            'aff1': os.path.abspath(args.aff1_cfg),
            'aff2': os.path.abspath(args.aff2_cfg),
            'dex': os.path.abspath(args.dex_cfg),
            'opt': os.path.abspath(args.opt_cfg),
        },
        os.path.join(session_dirs['base'], 'config_snapshot')
    )

    # Preload affordance models (pn env)
    with time_block("Preload Affordance Models", logger):
        logger.info("Importing affordance modules (this may take a while on first use)...")
        from pipeline.utils.infer_affordance_first import AffordanceFirst
        from pipeline.utils.infer_affordance_second import AffordanceSecond
        aff1 = AffordanceFirst(repo_root, cfg['aff1']['model']['model_path'], cfg['aff1']['model']['device'], cfg['aff1']['model']['max_points'])
        aff2 = AffordanceSecond(repo_root, cfg['aff2']['model']['model_path'], cfg['aff2']['model']['device'], cfg['aff2']['model']['max_points'])

    if args.object:
        process_object(args.object, cfg, session_dirs, aff1, aff2, logger)
    else:
        # Simple CLI loop
        logger.info("Enter object name (or 'exit' to quit):")
        while True:
            try:
                obj_name = input('object> ').strip()
            except EOFError:
                break
            if obj_name.lower() in {"exit", "quit"}:
                break
            if not obj_name:
                continue
            try:
                # Create per-object directory under session
                per_obj_dirs = make_session_dirs(outputs_root, session_dirs['session_id'], obj_name)
                process_object(obj_name, cfg, per_obj_dirs, aff1, aff2, logger)
                logger.info(f"Completed object: {obj_name}")
            except Exception as e:
                logger.exception(f"Failed object {obj_name}: {e}")


if __name__ == '__main__':
    main()


# python /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/pipeline/pipeline.py --object 100015
import os
import sys
import argparse
from typing import Dict, Optional

import numpy as np


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CUR_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.utils.run_utils import load_yaml, resolve_config_paths, make_session_dirs, setup_logger, time_block, run_subprocess
from pipeline.utils.io import load_point_cloud, save_json, save_npy, snapshot_config
from pipeline.utils.viz import (
    write_pointcloud_with_values_html,
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

    # Step 2: Affordance First → kp1
    with time_block("Affordance First", logger):
        logger.info("Running affordance-first inference ...")
        aff1_scores = aff1.predict(points)
        logger.info(f"Aff1 scores: min={float(np.min(aff1_scores)):.4f}, max={float(np.max(aff1_scores)):.4f}")
        kp1 = aff1.sample_keypoint(points, aff1_scores,
                                   strategy=cfg['aff1']['sampling']['strategy'],
                                   top_k=cfg['aff1']['sampling']['top_k'],
                                   nms_radius=cfg['aff1']['sampling']['nms_radius'],
                                   temperature=cfg['aff1']['sampling']['temperature'])
        logger.info(f"Sampled kp1: idx={kp1['index']}, xyz={np.array(kp1['xyz']).round(4).tolist()}")
        kp1_serial = _serialize_keypoint(kp1)
        np.save(os.path.join(obj_dir, 'aff1.npy'), aff1_scores)
        save_json(os.path.join(obj_dir, 'kp1.json'), kp1_serial)
        if cfg['visualization']['enable']:
            write_pointcloud_with_values_html(points, aff1_scores, [(kp1_serial['index'], np.array(kp1_serial['xyz']))], os.path.join(obj_dir, 'viz_step2.html'), title='Affordance First')
            logger.info("Saved visualization: viz_step2.html")

    # Step 3: Affordance Second → kp2
    with time_block("Affordance Second", logger):
        logger.info("Running affordance-second inference ...")
        kp1_xyz = np.array(kp1_serial['xyz'], dtype=np.float32)
        aff2_scores = aff2.predict(points, kp1_xyz)
        logger.info(f"Aff2 scores: min={float(np.min(aff2_scores)):.4f}, max={float(np.max(aff2_scores)):.4f}")
        kp2 = aff2.sample_keypoint(points, aff2_scores,
                                   top_k=cfg['aff2']['sampling']['top_k'],
                                   temperature=cfg['aff2']['sampling']['temperature'])
        logger.info(f"Sampled kp2: idx={kp2['index']}, xyz={np.array(kp2['xyz']).round(4).tolist()}")
        kp2_serial = _serialize_keypoint(kp2)
        np.save(os.path.join(obj_dir, 'aff2.npy'), aff2_scores)
        save_json(os.path.join(obj_dir, 'kp2.json'), kp2_serial)
        if cfg['visualization']['enable']:
            write_pointcloud_with_values_html(points, aff2_scores,
                                              [(kp1_serial['index'], np.array(kp1_serial['xyz'])),
                                               (kp2_serial['index'], np.array(kp2_serial['xyz']))],
                                              os.path.join(obj_dir, 'viz_step3.html'), title='Affordance Second')
            logger.info("Saved visualization: viz_step3.html")

    vis_cfg = cfg['dex'].get('visualization', {})
    pk_root = os.path.join(cfg['paths']['third_party']['bimangrasp'], 'thirdparty')

    # Step 4: Pose Initialization (DexGrasp or Keypoint init) via conda run
    with time_block("DexGrasp Pose Generation", logger):
        left_kp_path = os.path.join(obj_dir, 'kpleft.npy')
        right_kp_path = os.path.join(obj_dir, 'kpright.npy')
        np.save(left_kp_path, np.array(kp1_serial['xyz'], dtype=np.float32))
        np.save(right_kp_path, np.array(kp2_serial['xyz'], dtype=np.float32))
        logger.info(f"Saved keypoints: {left_kp_path}, {right_kp_path}")

        dex_entry_path = os.path.join(obj_dir, 'dexgrasp_entry.npy')
        init_cfg = cfg['dex']['dexgrasp'].get('init', {})
        if init_cfg.get('mode', 'dexgrasp') == 'keypoint':
            logger.info("Using keypoint-based hand initialization ...")
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
            stdout_path = os.path.join(logs_dir, 'init_kp_stdout.txt')
            stderr_path = os.path.join(logs_dir, 'init_kp_stderr.txt')
            rc, _, _ = run_subprocess(cmd, stdout_path=stdout_path, stderr_path=stderr_path, logger=logger)
            if rc != 0:
                raise RuntimeError(f"Keypoint init failed: rc={rc}")
            scale = load_default_scale(
                obj_name,
                vis_cfg.get('ref_scale_dir', ''),
                vis_cfg.get('ref_scale_file', ''),
                vis_cfg.get('ref_scale_value', None),
            )
            dex_npz_path = None
            # Load entry back for later baseline/step5
            try:
                dex_arr = np.load(dex_entry_path, allow_pickle=True)
                dex_entry = dex_arr[0].item() if hasattr(dex_arr[0], 'item') else dex_arr[0]
                # ensure scale written for downstream viz/opt consistency
                dex_entry['scale'] = float(scale)
                save_bimanual_entry(dex_entry_path, dex_entry)
            except Exception:
                dex_entry = {'qpos_left': {}, 'qpos_right': {}, 'scale': float(scale)}
        else:
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
        _render_bimanual_viz(
            dex_entry_path,
            obj_name,
            os.path.join(obj_dir, 'viz_step4.html'),
            vis_cfg,
            cfg['envs']['optimizer_env'],
            pk_root,
            logs_dir,
            logger,
            kpleft_path=left_kp_path,
            kpright_path=right_kp_path,
            points_path=os.path.join(obj_dir, 'points.npy'),
        )
        logger.info("Saved visualization: viz_step4.html")

    # Step 5: Optimization via BimanGrasp
    with time_block("BimanGrasp Optimization", logger):
        obj_map: Dict[str, str] = cfg['opt']['objects']['object_code_map'] or {}
        object_code = obj_map.get(obj_name, obj_name)
        logger.info(f"Optimizer object_code: {object_code}")

        out_npz = os.path.join(obj_dir, cfg['opt']['io']['optimized_npz_name'])
        out_json = os.path.join(obj_dir, cfg['opt']['io']['optimized_json_name'])

        script = os.path.join(repo_root, 'pipeline', 'utils', 'pose_optimizer.py')
        total_steps = int(cfg['optimizer']['steps']) if 'optimizer' in cfg and 'steps' in cfg['optimizer'] else int(cfg['opt']['optimizer']['steps']) if 'optimizer' in cfg['opt'] else 100
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
        if init_cfg.get('mode', 'dexgrasp') == 'keypoint':
            cmd.extend(['--entry', dex_entry_path])
        else:
            cmd.extend(['--dex_npz', dex_npz_path])
        # Pass consistent object scale to optimizer (match viz/Dex scale)
        try:
            obj_scale = load_default_scale(
                obj_name,
                vis_cfg.get('ref_scale_dir', ''),
                vis_cfg.get('ref_scale_file', ''),
                vis_cfg.get('ref_scale_value', None),
            )
            if obj_scale is not None:
                cmd.extend(['--object_scale', str(float(obj_scale))])
        except Exception:
            pass
        opt_vis_cfg = cfg['opt'].get('visualization', {})
        frames_dir = None
        video_path = None
        if opt_vis_cfg.get('enable', False):
            frames_dir = os.path.join(obj_dir, 'optimization_frames')
            video_path = os.path.join(obj_dir, opt_vis_cfg.get('video_name', 'optimization.mp4'))
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
        # Optional optimizer overrides for stability
        try:
            if 'optimizer' in cfg['opt'] and cfg['opt']['optimizer'].get('step_size') is not None:
                cmd.extend(['--opt_step_size', str(float(cfg['opt']['optimizer']['step_size']))])
            if 'optimizer' in cfg['opt'] and cfg['opt']['optimizer'].get('accept_warmup') is not None:
                cmd.extend(['--accept_warmup', str(int(cfg['opt']['optimizer']['accept_warmup']))])
            if 'optimizer' in cfg['opt'] and cfg['opt']['optimizer'].get('freeze_joints_steps') is not None:
                cmd.extend(['--freeze_joints_steps', str(int(cfg['opt']['optimizer']['freeze_joints_steps']))])
            if 'optimizer' in cfg['opt'] and cfg['opt']['optimizer'].get('freeze_translation_steps') is not None:
                cmd.extend(['--freeze_translation_steps', str(int(cfg['opt']['optimizer']['freeze_translation_steps']))])
        except Exception:
            pass
        stdout_path = os.path.join(logs_dir, 'biman_opt_stdout.txt')
        stderr_path = os.path.join(logs_dir, 'biman_opt_stderr.txt')
        progress_file = os.path.join(logs_dir, 'biman_opt_progress.txt')
        cmd.extend(['--progress_file', progress_file])
        logger.info("Launching BimanGrasp subprocess ...")
        # Run asynchronously to show progress
        import subprocess, shlex, time
        env = os.environ.copy()
        env['PIPELINE_PROGRESS_FILE'] = progress_file
        # Energy weights overrides via env
        energy_cfg = cfg['opt'].get('energy', {}) if 'opt' in cfg else {}
        if energy_cfg:
            if energy_cfg.get('w_dis') is not None:
                env['PIPELINE_W_DIS'] = str(float(energy_cfg['w_dis']))
            if energy_cfg.get('w_pen') is not None:
                env['PIPELINE_W_PEN'] = str(float(energy_cfg['w_pen']))
            if energy_cfg.get('w_spen') is not None:
                env['PIPELINE_W_SPEN'] = str(float(energy_cfg['w_spen']))
            if energy_cfg.get('w_joints') is not None:
                env['PIPELINE_W_JOINTS'] = str(float(energy_cfg['w_joints']))
            if energy_cfg.get('w_vew') is not None:
                env['PIPELINE_W_VEW'] = str(float(energy_cfg['w_vew']))
        # noise factor
        opt_cfg = cfg['opt'].get('optimizer', {}) if 'opt' in cfg else {}
        if opt_cfg.get('noise_factor') is not None:
            env['PIPELINE_NOISE_FACTOR'] = str(float(opt_cfg['noise_factor']))
        with open(stdout_path, 'w') as f_out, open(stderr_path, 'w') as f_err:
            proc = subprocess.Popen(cmd, stdout=f_out, stderr=f_err, text=True, env=env)
            last = -1
            while True:
                rc = proc.poll()
                try:
                    if os.path.exists(progress_file):
                        with open(progress_file, 'r') as pf:
                            cur = int(pf.read().strip() or '0')
                        if cur != last:
                            logger.info(f"[Opt] progress {cur}/{total_steps} ({(cur/ max(total_steps,1))*100:.1f}%)")
                            last = cur
                except Exception:
                    pass
                if rc is not None:
                    break
                time.sleep(1.0)
        if rc != 0:
            raise RuntimeError(f"BimanGrasp optimization failed: rc={rc}")
        logger.info(f"BimanGrasp outputs: {out_npz}, {out_json}")
        if video_path and os.path.exists(video_path):
            logger.info(f"Optimization video saved: {video_path}")

        if os.path.exists(out_json):
            data = json_load(out_json)
            opt_entry = build_bimanual_entry(
                data['left_qpos'],
                data['right_qpos'],
                scale,
                left_st=dex_entry['qpos_left'],
                right_st=dex_entry['qpos_right'],
            )
            opt_entry_path = os.path.join(obj_dir, 'optimized_pose.npy')
            save_bimanual_entry(opt_entry_path, opt_entry)
            _render_bimanual_viz(
                opt_entry_path,
                obj_name,
                os.path.join(obj_dir, 'viz_step5.html'),
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
            logger.info("Saved visualization: viz_step5.html")


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



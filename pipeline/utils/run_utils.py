import os
import sys
import time
import json
import uuid
import shlex
import logging
import subprocess
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def generate_session_id() -> str:
    return datetime.now().strftime("session_%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def make_session_dirs(outputs_root: str, session_id: Optional[str] = None, object_name: Optional[str] = None) -> Dict[str, str]:
    sid = session_id or generate_session_id()
    base = ensure_dir(os.path.join(outputs_root, sid))
    obj_dir = ensure_dir(os.path.join(base, object_name)) if object_name else base
    logs_dir = ensure_dir(os.path.join(base, "logs"))
    return {"session_id": sid, "base": base, "object": obj_dir, "logs": logs_dir}


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger(f"pipeline.{uuid.uuid4().hex[:8]}")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


@contextmanager
def time_block(desc: str, logger: Optional[logging.Logger] = None):
    t0 = time.perf_counter()
    if logger:
        logger.info(f"[START] {desc} ...")
    else:
        print(f"[START] {desc} ...", flush=True)
    try:
        yield
    finally:
        t1 = time.perf_counter()
        msg = f"{desc} took {(t1 - t0):.3f}s"
        if logger:
            logger.info(msg)
        else:
            print(msg, flush=True)


def run_subprocess(
    cmd: List[str],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    stdout_path: Optional[str] = None,
    stderr_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[int, str, str]:
    if logger:
        logger.info(f"Running: {' '.join(shlex.quote(c) for c in cmd)}")
    else:
        print("Running:", " ".join(shlex.quote(c) for c in cmd), flush=True)

    stdout_f = open(stdout_path, "w") if stdout_path else subprocess.PIPE
    stderr_f = open(stderr_path, "w") if stderr_path else subprocess.PIPE
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=stdout_f, stderr=stderr_f, text=True)
        out, err = proc.communicate()
        rc = proc.returncode
    finally:
        if stdout_path and hasattr(stdout_f, "close"):
            stdout_f.close()
        if stderr_path and hasattr(stderr_f, "close"):
            stderr_f.close()

    if logger:
        logger.info(f"Return code: {rc}")
        if out:
            logger.info(f"stdout: {out[-500:]}" if len(out) > 500 else f"stdout: {out}")
        if err:
            logger.warning(f"stderr: {err[-500:]}" if len(err) > 500 else f"stderr: {err}")
    return rc, out or "", err or ""


def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(value: str, context: Dict[str, str]) -> str:
    if not isinstance(value, str):
        return value
    out = value
    for k, v in context.items():
        out = out.replace("${" + k + "}", str(v))
    return out


def resolve_config_paths(cfg: dict) -> dict:
    # Flatten top-level path placeholders
    placeholders = {}
    if "paths" in cfg:
        # Expand nested paths into placeholders map
        def _collect(prefix: str, d: dict):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _collect(key, v)
                else:
                    placeholders[key] = v
        _collect("paths", cfg.get("paths", {}))
        placeholders.update(cfg.get("paths", {}))

    def _resolve(obj):
        if isinstance(obj, dict):
            return {k: _resolve(resolve_path(v, placeholders)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve(x) for x in obj]
        if isinstance(obj, str):
            return resolve_path(obj, placeholders)
        return obj

    return _resolve(cfg)



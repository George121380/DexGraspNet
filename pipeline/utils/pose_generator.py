import os
from typing import Optional

from .run_utils import run_subprocess


def run_dexgrasp(
    conda_env: str,
    repo_root: str,
    dex_yaml: str,
    dex_ckpt: str,
    points_npy: str,
    kpleft_npy: str,
    kpright_npy: str,
    save_npz_path: str,
    device: str = "cuda:0",
    backbone: Optional[str] = None,
    logs_dir: Optional[str] = None,
) -> str:
    script = os.path.join(repo_root, "third_party", "DexGraspNet2", "src", "infer_bimanual.py")
    cmd = [
        "conda", "run", "-n", conda_env, "python", script,
        "--yaml", dex_yaml,
        "--ckpt", dex_ckpt,
        "--pc", points_npy,
        "--kpleft", kpleft_npy,
        "--kpright", kpright_npy,
        "--device", device,
        "--save", save_npz_path,
    ]
    if backbone:
        cmd += ["--backbone", backbone]

    stdout_path = os.path.join(logs_dir, "dexgrasp_stdout.txt") if logs_dir else None
    stderr_path = os.path.join(logs_dir, "dexgrasp_stderr.txt") if logs_dir else None
    rc, _, _ = run_subprocess(cmd, stdout_path=stdout_path, stderr_path=stderr_path)
    if rc != 0:
        raise RuntimeError(f"DexGrasp inference failed with code {rc}")
    return save_npz_path






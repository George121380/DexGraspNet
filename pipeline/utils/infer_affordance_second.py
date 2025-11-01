import os
import sys
from typing import Dict, Optional

import numpy as np
import torch


class AffordanceSecond:
    def __init__(self, repo_root: str, model_path: str, device: str = "auto", max_points: Optional[int] = None):
        self.repo_root = repo_root
        self.model_path = model_path
        self.device = self._set_device(device)
        self.max_points = max_points
        self.model = self._load_model()

    def _set_device(self, device: str) -> torch.device:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)

    def _load_model(self) -> torch.nn.Module:
        # Add train_eval_second to path and import model class
        tes = os.path.join(self.repo_root, "train_eval_second")
        if tes not in sys.path:
            sys.path.insert(0, tes)
        from models.affordance_second import AffordancePointNet2SSG  # type: ignore

        ckpt = torch.load(self.model_path, map_location="cpu")
        cfg = ckpt.get("cfg", {})
        model = AffordancePointNet2SSG(
            use_condition=cfg.get("use_condition", True),
            condition_mode=cfg.get("condition_mode", "rd"),
            use_xyz=True,
        )
        model.load_state_dict(ckpt["model"])  # type: ignore
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def predict(self, points: np.ndarray, keypoint_xyz: np.ndarray) -> np.ndarray:
        assert points.ndim == 2 and points.shape[1] == 3, f"Expected (N,3), got {points.shape}"
        assert keypoint_xyz.shape == (3,), f"Expected (3,), got {keypoint_xyz.shape}"
        pts = torch.from_numpy(points.astype(np.float32))[None, ...].to(self.device)
        centers = torch.from_numpy(keypoint_xyz.astype(np.float32))[None, ...].to(self.device)

        if isinstance(self.max_points, int) and self.max_points > 0 and pts.shape[1] > self.max_points:
            sel = torch.randperm(pts.shape[1], device=pts.device)[: self.max_points]
            pts = pts[:, sel, :]

        logits = self.model(pts, centers)  # (B,1,N)
        pred = torch.sigmoid(logits.squeeze(1))  # (B,N)
        return pred[0].detach().cpu().numpy()

    @staticmethod
    def sample_keypoint(points: np.ndarray, scores: np.ndarray, **kwargs) -> Dict[str, object]:
        return AffordanceSecond._sample_top(points, scores, kwargs.get("top_k", 128), kwargs.get("temperature", 0.1))

    @staticmethod
    def _sample_top(points: np.ndarray, scores: np.ndarray, top_k: int, temperature: float):
        N = points.shape[0]
        s = scores.reshape(N)
        if top_k is not None and top_k > 0 and top_k < N:
            sel = np.argpartition(s, -top_k)[-top_k:]
            idx = int(sel[np.argmax(s[sel])])
        else:
            idx = int(np.argmax(s))
        return {"index": idx, "xyz": points[idx].astype(np.float32)}






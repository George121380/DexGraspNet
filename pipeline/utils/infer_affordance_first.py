import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch


class AffordanceFirst:
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
        # Add train_eval_first to path and import model class
        tef = os.path.join(self.repo_root, "train_eval_first")
        if tef not in sys.path:
            sys.path.insert(0, tef)
        from models.affordance_first import AffordanceFirstPointNet2SSG  # type: ignore

        ckpt = torch.load(self.model_path, map_location="cpu")
        model = AffordanceFirstPointNet2SSG(use_xyz=True)
        model.load_state_dict(ckpt["model"])  # type: ignore
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def predict(self, points: np.ndarray) -> np.ndarray:
        assert points.ndim == 2 and points.shape[1] == 3, f"Expected (N,3), got {points.shape}"
        pts = torch.from_numpy(points.astype(np.float32))[None, ...].to(self.device)

        if isinstance(self.max_points, int) and self.max_points > 0 and pts.shape[1] > self.max_points:
            sel = torch.randperm(pts.shape[1], device=pts.device)[: self.max_points]
            pts = pts[:, sel, :]

        logits = self.model(pts)  # (B,1,N)
        pred = torch.sigmoid(logits.squeeze(1))  # (B,N)
        return pred[0].detach().cpu().numpy()

    @staticmethod
    def sample_keypoint(
        points: np.ndarray,
        scores: np.ndarray,
        strategy: str = "topk",
        top_k: int = 128,
        nms_radius: float = 0.01,
        temperature: float = 0.1,
    ) -> Dict[str, object]:
        N = points.shape[0]
        scores = scores.reshape(N)
        if strategy == "softmax":
            logits = scores / max(1e-6, float(temperature))
            probs = np.exp(logits - logits.max())
            probs = probs / (probs.sum() + 1e-8)
            idx = int(np.random.choice(np.arange(N), p=probs))
        else:
            # top-k then pick argmax within top-k; fallback to argmax
            if top_k is not None and top_k > 0 and top_k < N:
                sel = np.argpartition(scores, -top_k)[-top_k:]
                idx = int(sel[np.argmax(scores[sel])])
            else:
                idx = int(np.argmax(scores))
        kp = points[idx]
        return {"index": idx, "xyz": kp.astype(np.float32)}






import os
import argparse
import json
import sys
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import time

# Disable cuDNN for compatibility with older PyTorch builds on newer GPUs
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

CUR_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(CUR_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.affordance_first import AffordanceFirstPointNet2SSG  # type: ignore  # noqa: E402
from data.affordance_first_dataset import (  # type: ignore  # noqa: E402
    AffordanceFirstPairsDataset,
)


def set_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    profile: bool = False,
    max_points: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds_list = []
    targs_list = []
    todev_total = 0.0
    fwd_total = 0.0
    post_total = 0.0
    num_batches = 0
    t_eval0 = time.perf_counter()
    last_shape = None
    for pts, _, target in loader:
        num_batches += 1
        t0 = time.perf_counter()
        pts = pts.to(device)
        target = target.to(device)
        t1 = time.perf_counter()
        # Optional subsampling for speed
        if isinstance(max_points, int) and max_points > 0 and pts.shape[1] > max_points:
            if pts.shape[0] == 1:
                sel = torch.randperm(pts.shape[1], device=pts.device)[:max_points]
                pts = pts[:, sel, :]
                if target.dim() == 2:
                    target = target[:, sel]
                else:
                    target = target[sel]
            else:
                # Per-sample subsample for generality
                new_pts = []
                new_t = []
                for bi in range(pts.shape[0]):
                    sel = torch.randperm(pts.shape[1], device=pts.device)[:max_points]
                    new_pts.append(pts[bi:bi+1, sel, :])
                    if target.dim() == 2:
                        new_t.append(target[bi:bi+1, sel])
                    else:
                        new_t.append(target[sel])
                pts = torch.cat(new_pts, dim=0)
                target = torch.cat(new_t, dim=0 if target.dim() == 2 else 0)
        logits = model(pts)  # (B,1,N)
        pred = torch.sigmoid(logits.squeeze(1))  # (B,N)
        t2 = time.perf_counter()
        preds_list.append(pred.detach().cpu().numpy())
        targs_list.append(target.detach().cpu().numpy())
        t3 = time.perf_counter()
        todev_total += (t1 - t0)
        fwd_total += (t2 - t1)
        post_total += (t3 - t2)
        last_shape = pts.shape
    preds = np.concatenate(preds_list, axis=0)
    targs = np.concatenate(targs_list, axis=0)
    if profile:
        t_eval1 = time.perf_counter()
        total = t_eval1 - t_eval0
        avg_ms = (total / max(1, num_batches)) * 1000.0
        print(f"[Profile] eval: batches={num_batches}, total={total:.3f}s, avg_per_batch={avg_ms:.2f}ms, to_device={todev_total:.3f}s, forward={fwd_total:.3f}s, postproc={post_total:.3f}s, last_batch_shape={tuple(last_shape) if last_shape is not None else None}", flush=True)
    return preds, targs


def compute_metrics(preds: np.ndarray, targs: np.ndarray) -> dict:
    diff = preds - targs
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse + 1e-12))
    p = preds.reshape(-1)
    t = targs.reshape(-1)
    if p.std() > 1e-12 and t.std() > 1e-12:
        corr = float(np.corrcoef(p, t)[0, 1])
    else:
        corr = 0.0
    pr = [float(np.min(preds)), float(np.max(preds))]
    return dict(mse=mse, mae=mae, rmse=rmse, correlation=corr, pred_range=pr)


def _make_sample_figure(points_np: np.ndarray, pred_np: np.ndarray, targ_np: np.ndarray, center_np: np.ndarray):
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=("Prediction", "Target"))

    # Prediction panel
    fig.add_trace(
        go.Scatter3d(
            x=points_np[:, 0], y=points_np[:, 1], z=points_np[:, 2],
            mode='markers',
            marker=dict(size=2, color=pred_np, colorscale='Viridis', showscale=True),
            name='predicted_aff'
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter3d(
            x=[center_np[0]], y=[center_np[1]], z=[center_np[2]],
            mode='markers', marker=dict(size=6, color='red', symbol='circle'),
            name='center'
        ), row=1, col=1
    )

    # Target panel
    fig.add_trace(
        go.Scatter3d(
            x=points_np[:, 0], y=points_np[:, 1], z=points_np[:, 2],
            mode='markers',
            marker=dict(size=2, color=targ_np, colorscale='Viridis', showscale=True),
            name='target_aff'
        ), row=1, col=2
    )
    fig.add_trace(
        go.Scatter3d(
            x=[center_np[0]], y=[center_np[1]], z=[center_np[2]],
            mode='markers', marker=dict(size=6, color='red', symbol='circle'),
            name='center'
        ), row=1, col=2
    )

    for c in [1, 2]:
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                          aspectmode='data', row=1, col=c)
    fig.update_layout(title='Affordance (Prediction vs Target)', title_x=0.5)
    return fig


def _save_visualizations(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                         out_dir: str, max_vis: int = 5, profile: bool = False, max_points: int = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    index_lines = [
        "<html><head><meta charset='utf-8'><title>Affordance Visualizations</title></head><body>",
        "<h2>Affordance Visualizations</h2>",
        "<ul>"
    ]
    saved = 0
    model.eval()
    t_vis0 = time.perf_counter()
    per_sample_times = []
    with torch.no_grad():
        for pts, centers, target in loader:
            pts = pts.to(device)
            centers = centers.to(device)
            target = target.to(device)
            # Optional subsampling for speed during visualization as well
            if isinstance(max_points, int) and max_points > 0 and pts.shape[1] > max_points:
                if pts.shape[0] == 1:
                    sel = torch.randperm(pts.shape[1], device=pts.device)[:max_points]
                    pts = pts[:, sel, :]
                    centers = centers  # center is a single point; keep unchanged
                    target = target if target.dim() == 2 else target
                    if target.dim() == 2:
                        target = target[:, sel]
                else:
                    new_pts = []
                    new_t = []
                    for bi in range(pts.shape[0]):
                        sel = torch.randperm(pts.shape[1], device=pts.device)[:max_points]
                        new_pts.append(pts[bi:bi+1, sel, :])
                        if target.dim() == 2:
                            new_t.append(target[bi:bi+1, sel])
                    pts = torch.cat(new_pts, dim=0)
                    if target.dim() == 2 and len(new_t) > 0:
                        target = torch.cat(new_t, dim=0)
            logits = model(pts)
            pred = torch.sigmoid(logits.squeeze(1))

            for i in range(pts.shape[0]):
                t_s0 = time.perf_counter()
                points_np = pts[i].detach().cpu().numpy()
                pred_np = pred[i].detach().cpu().numpy()
                targ_np = target[i].detach().cpu().numpy()
                center_np = centers[i].detach().cpu().numpy()

                fig = _make_sample_figure(points_np, pred_np, targ_np, center_np)
                sample_path = os.path.join(out_dir, f'sample_{saved:04d}.html')
                fig.write_html(sample_path)
                index_lines.append(f"<li><a href='sample_{saved:04d}.html' target='_blank'>sample {saved:04d}</a></li>")
                t_s1 = time.perf_counter()
                per_sample_times.append(t_s1 - t_s0)
                saved += 1
                if saved >= max_vis:
                    break
            if saved >= max_vis:
                break

    index_lines.append("</ul></body></html>")
    with open(os.path.join(out_dir, 'index.html'), 'w') as f:
        f.write("\n".join(index_lines))
    if profile:
        t_vis1 = time.perf_counter()
        total = t_vis1 - t_vis0
        avg_ms = (total / max(1, len(per_sample_times))) * 1000.0
        file_ms = (sum(per_sample_times) / max(1, len(per_sample_times))) * 1000.0
        print(f"[Profile] visualize: saved={len(per_sample_times)}, total={total:.3f}s, avg_per_sample_total={avg_ms:.2f}ms, avg_per_sample_html={file_ms:.2f}ms", flush=True)


def _save_overview(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                   out_dir: str, max_vis: int = 36, page_size: int = 8, profile: bool = False, max_points: int = None) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def _new_page_header() -> list:
        return [
            "<html><head><meta charset='utf-8'><title>Affordance Overview</title>",
            "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
            "<style>body{font-family:sans-serif;} .row{margin-bottom:12px;} .card{display:inline-block;margin:6px;border:1px solid #ddd;padding:4px;vertical-align:top;} .title{font-size:12px;color:#555;margin:0 0 4px 2px;}</style>",
            "</head><body>",
            "<h2>Affordance Overview (Prediction vs Target)</h2>"
        ]

    pages: list = []
    page_parts = _new_page_header()
    saved_total = 0
    saved_on_page = 0

    model.eval()
    with torch.no_grad():
        for pts, centers, target in loader:
            pts = pts.to(device)
            centers = centers.to(device)
            target = target.to(device)
            if isinstance(max_points, int) and max_points > 0 and pts.shape[1] > max_points:
                if pts.shape[0] == 1:
                    sel = torch.randperm(pts.shape[1], device=pts.device)[:max_points]
                    pts = pts[:, sel, :]
                    if target.dim() == 2:
                        target = target[:, sel]
                else:
                    new_pts = []
                    new_t = []
                    for bi in range(pts.shape[0]):
                        sel = torch.randperm(pts.shape[1], device=pts.device)[:max_points]
                        new_pts.append(pts[bi:bi+1, sel, :])
                        if target.dim() == 2:
                            new_t.append(target[bi:bi+1, sel])
                    pts = torch.cat(new_pts, dim=0)
                    if target.dim() == 2 and len(new_t) > 0:
                        target = torch.cat(new_t, dim=0)
            logits = model(pts)
            pred = torch.sigmoid(logits.squeeze(1))

            for i in range(pts.shape[0]):
                points_np = pts[i].detach().cpu().numpy()
                pred_np = pred[i].detach().cpu().numpy()
                targ_np = target[i].detach().cpu().numpy()
                center_np = centers[i].detach().cpu().numpy()

                fig = _make_sample_figure(points_np, pred_np, targ_np, center_np)
                fig.update_layout(height=320, width=900, margin=dict(l=0, r=0, t=30, b=0))
                snippet = pio.to_html(fig, include_plotlyjs=False, full_html=False)
                page_parts.append("<div class='row'><div class='card'><div class='title'>sample %04d</div>%s</div></div>" % (saved_total, snippet))

                saved_total += 1
                saved_on_page += 1

                if saved_on_page >= page_size:
                    page_parts.append("</body></html>")
                    pages.append(page_parts)
                    page_parts = _new_page_header()
                    saved_on_page = 0

                if saved_total >= max_vis:
                    break
            if saved_total >= max_vis:
                break

    if saved_on_page > 0:
        page_parts.append("</body></html>")
        pages.append(page_parts)

    page_files = []
    for idx, parts in enumerate(pages, start=1):
        fname = f"overview_page_{idx:04d}.html"
        with open(os.path.join(out_dir, fname), 'w') as f:
            f.write("\n".join(parts))
        page_files.append(fname)

    index_lines = [
        "<html><head><meta charset='utf-8'><title>Affordance Overview Index</title></head><body>",
        "<h2>Affordance Overview Pages</h2>",
        "<ul>"
    ]
    for fname in page_files:
        index_lines.append(f"<li><a href='{fname}' target='_blank'>{fname}</a></li>")
    index_lines.append("</ul></body></html>")
    with open(os.path.join(out_dir, 'overview_index.html'), 'w') as f:
        f.write("\n".join(index_lines))
    if profile:
        # We did not record fine-grained times inside; report page count only.
        print(f"[Profile] overview: pages={len(page_files)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to aff_first_pairs_<hand>.npy file')
    parser.add_argument('--output_dir', type=str, default='./evaluation_first_results')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--max_vis_samples', type=int, default=5)
    parser.add_argument('--pair_idx', type=int, default=None, help='evaluate only a specific pair index')
    parser.add_argument('--target_key', type=str, default='aff_scores_left',
                        help='Key for affordance scores inside each pair record')
    parser.add_argument('--center_key', type=str, default=None,
                        help='Override center key if stored under a different name')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader worker processes (0 for main process)')
    parser.add_argument('--overview', action='store_true',
                        help='Also generate overview page(s) summarizing many samples')
    parser.add_argument('--overview_max_vis', type=int, default=None,
                        help='Max samples to include in overview; defaults to --max_vis_samples')
    parser.add_argument('--profile', action='store_true',
                        help='Print detailed timing of each major step')
    parser.add_argument('--max_points', type=int, default=None,
                        help='Randomly subsample each point cloud to at most this many points during eval/vis')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.perf_counter()
    device = set_device(args.device)
    if args.profile:
        print(f"[Profile] device: {device}", flush=True)

    t_ds0 = time.perf_counter()
    dset = AffordanceFirstPairsDataset(
        npy_path=args.data_path,
        split='all',
        augment_rotate_z=False,
        augment_jitter_std=0.0,
        shuffle_points=False,
        only_pair_idx=args.pair_idx,
        center_key_override=args.center_key,
        target_key_override=args.target_key,
    )
    t_ds1 = time.perf_counter()
    loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    t_dl1 = time.perf_counter()

    t_m0 = time.perf_counter()
    ckpt = torch.load(args.model_path, map_location='cpu')
    cfg = ckpt.get('cfg', {})
    model = AffordanceFirstPointNet2SSG(use_xyz=True)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    t_m1 = time.perf_counter()

    t_ev0 = time.perf_counter()
    preds, targs = evaluate_model(model, loader, device, profile=args.profile, max_points=args.max_points)
    t_ev1 = time.perf_counter()
    t_sv0 = time.perf_counter()
    np.save(os.path.join(args.output_dir, 'predictions.npy'), preds)
    np.save(os.path.join(args.output_dir, 'targets.npy'), targs)
    t_sv1 = time.perf_counter()

    t_mt0 = time.perf_counter()
    metrics = compute_metrics(preds, targs)
    t_mt1 = time.perf_counter()
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    t_mt2 = time.perf_counter()

    print(json.dumps(metrics))

    if args.visualize:
        vis_dir = os.path.join(args.output_dir, 'visualizations')
        _save_visualizations(model, loader, device, vis_dir, max_vis=args.max_vis_samples, profile=args.profile, max_points=args.max_points)
        if args.overview:
            ov_max = args.overview_max_vis if args.overview_max_vis is not None else args.max_vis_samples
            _save_overview(model, loader, device, vis_dir, max_vis=min(int(ov_max), len(dset)), profile=args.profile, max_points=args.max_points)

    if args.profile:
        t1 = time.perf_counter()
        print(
            f"[Profile] summary: total={t1 - t0:.3f}s | dataset={t_ds1 - t_ds0:.3f}s | dataloader={t_dl1 - t_ds1:.3f}s | model_load={t_m1 - t_m0:.3f}s | eval={t_ev1 - t_ev0:.3f}s | save_arrays={t_sv1 - t_sv0:.3f}s | metrics_compute={t_mt1 - t_mt0:.3f}s | metrics_write={t_mt2 - t_mt1:.3f}s",
            flush=True,
        )


if __name__ == '__main__':
    main()







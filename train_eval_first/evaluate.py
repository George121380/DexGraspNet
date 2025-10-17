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
def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds_list = []
    targs_list = []
    for pts, _, target in loader:
        pts = pts.to(device)
        target = target.to(device)
        logits = model(pts)  # (B,1,N)
        pred = torch.sigmoid(logits.squeeze(1))  # (B,N)
        preds_list.append(pred.detach().cpu().numpy())
        targs_list.append(target.detach().cpu().numpy())
    preds = np.concatenate(preds_list, axis=0)
    targs = np.concatenate(targs_list, axis=0)
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
                         out_dir: str, max_vis: int = 5) -> None:
    os.makedirs(out_dir, exist_ok=True)
    index_lines = [
        "<html><head><meta charset='utf-8'><title>Affordance Visualizations</title></head><body>",
        "<h2>Affordance Visualizations</h2>",
        "<ul>"
    ]
    saved = 0
    model.eval()
    with torch.no_grad():
        for pts, centers, target in loader:
            pts = pts.to(device)
            centers = centers.to(device)
            target = target.to(device)
            logits = model(pts)
            pred = torch.sigmoid(logits.squeeze(1))

            for i in range(pts.shape[0]):
                points_np = pts[i].detach().cpu().numpy()
                pred_np = pred[i].detach().cpu().numpy()
                targ_np = target[i].detach().cpu().numpy()
                center_np = centers[i].detach().cpu().numpy()

                fig = _make_sample_figure(points_np, pred_np, targ_np, center_np)
                sample_path = os.path.join(out_dir, f'sample_{saved:04d}.html')
                fig.write_html(sample_path)
                index_lines.append(f"<li><a href='sample_{saved:04d}.html' target='_blank'>sample {saved:04d}</a></li>")
                saved += 1
                if saved >= max_vis:
                    break
            if saved >= max_vis:
                break

    index_lines.append("</ul></body></html>")
    with open(os.path.join(out_dir, 'index.html'), 'w') as f:
        f.write("\n".join(index_lines))


def _save_overview(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                   out_dir: str, max_vis: int = 36, page_size: int = 8) -> None:
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = set_device(args.device)

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
    loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    ckpt = torch.load(args.model_path, map_location='cpu')
    cfg = ckpt.get('cfg', {})
    model = AffordanceFirstPointNet2SSG(use_xyz=True)
    model.load_state_dict(ckpt['model'])
    model.to(device)

    preds, targs = evaluate_model(model, loader, device)
    np.save(os.path.join(args.output_dir, 'predictions.npy'), preds)
    np.save(os.path.join(args.output_dir, 'targets.npy'), targs)

    metrics = compute_metrics(preds, targs)
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics))

    if args.visualize:
        vis_dir = os.path.join(args.output_dir, 'visualizations')
        _save_visualizations(model, loader, device, vis_dir, max_vis=args.max_vis_samples)
        _save_overview(model, loader, device, vis_dir, max_vis=min(9999, len(dset)))


if __name__ == '__main__':
    main()







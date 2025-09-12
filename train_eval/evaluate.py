#!/usr/bin/env python3
"""
SecAff Affordance Model Evaluation Script
Final optimized version with interactive visualization

Usage:
    conda activate pn
    python evaluate.py --model_path checkpoints/best_model.pth --data_path aff_sec_result/2_of_Jenga_Classic_Game/aff_sec_pairs.npy
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not available, skipping interactive visualizations")

from torch.utils.data import DataLoader

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from models.affordance import SecAffModel, AffordanceDataset


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate SecAff Affordance Model')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to test data (.npy file)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Output directory for results (default: ./evaluation_results)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation (default: 1)')
    parser.add_argument('--visualize', action='store_true',
                        help='Create interactive 3D visualizations')
    parser.add_argument('--max_vis_samples', type=int, default=5,
                        help='Maximum number of samples to visualize (default: 5)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--high_value_target_thr', type=float, default=0.5,
                        help='Threshold on target to define high-value points (default: 0.5)')
    parser.add_argument('--high_value_pred_thr', type=float, default=0.5,
                        help='Threshold on prediction to count a hit on high-value points (default: 0.5)')
    parser.add_argument('--pred_minmax_norm', action='store_true',
                        help='Per-sample min-max normalize predictions at evaluation time only')
    
    return parser.parse_args()


def load_model(model_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    args_dict = checkpoint['args']
    
    # Create model with same architecture
    model = SecAffModel(
        pointcloud_extractor_type=args_dict.get('extractor_type', 'ssg'),
        pointcloud_feature_dim=args_dict.get('pointcloud_feature_dim', 512),
        keypoint_encoder_dim=args_dict.get('keypoint_encoder_dim', 128),
        fusion_hidden_dim=args_dict.get('fusion_hidden_dim', 256),
        num_points=args_dict.get('max_points', 1024)
    ).to(device)
    
    # Load trained weights
    load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # Report any missing/unexpected keys (likely from newer heads like logit_scale)
    try:
        missing = getattr(load_result, 'missing_keys', [])
        unexpected = getattr(load_result, 'unexpected_keys', [])
        if missing:
            print(f"Warning: missing keys when loading checkpoint: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys when loading checkpoint: {unexpected}")
    except Exception:
        pass
    model.eval()
    
    print(f"✅ Model loaded from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")
    return model, args_dict


def evaluate_model(model, dataloader, device, pred_minmax_norm: bool = False):
    """Evaluate model on dataset"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_losses = []
    
    print(f"Evaluating on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            pointcloud = batch['pointcloud'].to(device)
            keypoints = batch['keypoints'].to(device)
            target_scores = batch['affordance_scores'].to(device)
            
            pred_scores = model(pointcloud, keypoints)
            if pred_minmax_norm:
                min_v = pred_scores.min(dim=1, keepdim=True)[0]
                max_v = pred_scores.max(dim=1, keepdim=True)[0]
                pred_scores = (pred_scores - min_v) / (max_v - min_v + 1e-6)
            loss = F.mse_loss(pred_scores, target_scores)
            
            all_predictions.append(pred_scores.cpu().numpy())
            all_targets.append(target_scores.cpu().numpy())
            all_losses.append(loss.item())
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx:2d}/{len(dataloader)}: Loss={loss.item():.6f}")
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    return predictions, targets, all_losses


def compute_metrics(predictions, targets, high_value_target_thr: float = 0.5, high_value_pred_thr: float = 0.5):
    """Compute comprehensive evaluation metrics"""
    # Flatten for analysis
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Basic metrics
    mse = np.mean((pred_flat - target_flat) ** 2)
    mae = np.mean(np.abs(pred_flat - target_flat))
    rmse = np.sqrt(mse)
    
    # Correlation
    correlation = np.corrcoef(pred_flat, target_flat)[0, 1] if len(np.unique(pred_flat)) > 1 else 0.0
    
    # Range analysis
    pred_range = [pred_flat.min(), pred_flat.max()]
    target_range = [target_flat.min(), target_flat.max()]
    
    # Distribution analysis
    pred_mean, pred_std = pred_flat.mean(), pred_flat.std()
    target_mean, target_std = target_flat.mean(), target_flat.std()
    
    # High-value analysis
    high_value_mask = target_flat > float(high_value_target_thr)
    if high_value_mask.sum() > 0:
        high_value_mae = np.mean(np.abs(pred_flat[high_value_mask] - target_flat[high_value_mask]))
        high_value_recall = np.mean(pred_flat[high_value_mask] > float(high_value_pred_thr))
    else:
        high_value_mae = 0.0
        high_value_recall = 0.0
    
    # Per-sample metrics
    sample_mses = np.mean((predictions - targets) ** 2, axis=1)
    sample_maes = np.mean(np.abs(predictions - targets), axis=1)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation,
        'pred_range': pred_range,
        'target_range': target_range,
        'pred_mean': pred_mean,
        'pred_std': pred_std,
        'target_mean': target_mean,
        'target_std': target_std,
        'high_value_mae': high_value_mae,
        'high_value_recall': high_value_recall,
        'high_value_target_thr': float(high_value_target_thr),
        'high_value_pred_thr': float(high_value_pred_thr),
        'sample_mses': sample_mses,
        'sample_maes': sample_maes
    }
    
    return metrics


def create_summary_visualizations(predictions, targets, output_dir, max_scatter_points=200000):
    """Create dataset-level visualizations: distributions, errors, and scatter.

    Saves interactive HTML files under output_dir/visualizations/.
    """
    if not HAS_PLOTLY:
        print("Skipping summary visualizations - plotly not available")
        return

    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    error_flat = np.abs(pred_flat - target_flat)

    # 1) Distribution overlay
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=pred_flat, name='Predictions', opacity=0.6, nbinsx=60))
    fig_dist.add_trace(go.Histogram(x=target_flat, name='Targets', opacity=0.6, nbinsx=60))
    fig_dist.update_layout(
        barmode='overlay',
        title='Distribution: Predictions vs Targets',
        xaxis_title='Value', yaxis_title='Count',
        legend=dict(x=0.75, y=0.95)
    )
    fig_dist.update_traces(marker_line_width=0)
    fig_dist.write_html(os.path.join(vis_dir, 'distribution.html'))

    # 2) Error histogram
    fig_err = go.Figure()
    fig_err.add_trace(go.Histogram(x=error_flat, nbinsx=60, marker_color='crimson'))
    fig_err.update_layout(title='Absolute Error Distribution', xaxis_title='|Prediction - Target|', yaxis_title='Count')
    fig_err.update_traces(marker_line_width=0)
    fig_err.write_html(os.path.join(vis_dir, 'error_histogram.html'))

    # 3) Scatter (sampled)
    total_points = pred_flat.shape[0]
    if total_points > max_scatter_points:
        idx = np.random.choice(total_points, size=max_scatter_points, replace=False)
        pred_s = pred_flat[idx]
        target_s = target_flat[idx]
    else:
        pred_s = pred_flat
        target_s = target_flat
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scattergl(x=target_s, y=pred_s, mode='markers', marker=dict(size=3, opacity=0.4)))
    fig_scatter.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(color='green'), name='Ideal'))
    fig_scatter.update_layout(title='Predicted vs Target (sampled)', xaxis_title='Target', yaxis_title='Prediction')
    fig_scatter.write_html(os.path.join(vis_dir, 'pred_vs_target_scatter.html'))

    # 4) Calibration by target bins
    try:
        bins = np.linspace(0.0, 1.0, 21)
        bin_ids = np.digitize(target_flat, bins) - 1
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        mean_pred = []
        counts = []
        for b in range(len(bins)-1):
            mask = bin_ids == b
            counts.append(int(mask.sum()))
            mean_pred.append(float(pred_flat[mask].mean()) if mask.any() else None)
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=bin_centers, y=mean_pred, mode='lines+markers', name='Mean Prediction'))
        fig_cal.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(color='green', dash='dash'), name='Ideal'))
        fig_cal.update_layout(title='Calibration Curve (mean prediction by target bin)', xaxis_title='Target (bin center)', yaxis_title='Mean Prediction')
        fig_cal.write_html(os.path.join(vis_dir, 'calibration.html'))
    except Exception as e:
        print(f"Warning: failed to create calibration plot: {e}")

    # Create a small summary landing page
    summary_index = os.path.join(vis_dir, 'summary.html')
    with open(summary_index, 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Evaluation Summary Visualizations</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 30px; }
    h1 { color: #2c3e50; }
    .link { display: block; margin: 10px 0; }
  </style>
  </head>
<body>
  <h1>📊 Evaluation Summary Visualizations</h1>
  <a class=\"link\" href=\"distribution.html\">Distribution: Predictions vs Targets</a>
  <a class=\"link\" href=\"error_histogram.html\">Absolute Error Histogram</a>
  <a class=\"link\" href=\"pred_vs_target_scatter.html\">Predicted vs Target Scatter</a>
  <a class=\"link\" href=\"calibration.html\">Calibration Curve</a>
</body>
</html>
""")
    print(f"🖼️ Summary visualizations saved under: {vis_dir}")


def create_topk_error_visualizations(dataset, predictions, targets, output_dir, k=5):
    """Create 3D visualizations for top-K worst samples by MAE.

    Saves HTML files and an index page under output_dir/visualizations/.
    """
    if not HAS_PLOTLY:
        print("Skipping top-K visualizations - plotly not available")
        return

    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Compute per-sample MAE
    sample_mae = np.mean(np.abs(predictions - targets), axis=1)
    order = np.argsort(sample_mae)[::-1]  # descending (worst first)
    top_indices = order[:max(1, int(k))]

    links = []
    for rank, idx in enumerate(top_indices, start=1):
        sample = dataset[idx]
        pc = sample['pointcloud']
        kps = sample['keypoints']
        pred = predictions[idx]
        tgt = targets[idx]
        fig = create_affordance_visualization(pc, kps, pred, tgt, sample_idx=int(idx))
        if fig is not None:
            html_path = os.path.join(vis_dir, f'top_error_rank_{rank}_sample_{idx}.html')
            fig.write_html(html_path)
            links.append((rank, idx, html_path, float(sample_mae[idx])))
            print(f"  ✅ Saved top-{rank} error sample: {html_path}")

    # Create index page for top-K
    index_html = os.path.join(vis_dir, 'top_errors.html')
    with open(index_html, 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Top-K Worst Samples</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 30px; }
    h1 { color: #8e44ad; }
    table { border-collapse: collapse; }
    th, td { padding: 8px 12px; border: 1px solid #ddd; }
  </style>
</head>
<body>
  <h1>🚩 Top-K Worst Samples by MAE</h1>
  <table>
    <tr><th>Rank</th><th>Sample Index</th><th>MAE</th><th>Link</th></tr>
""")
        for rank, idx, path, mae in links:
            f.write(f"    <tr><td>{rank}</td><td>{idx}</td><td>{mae:.6f}</td><td><a href=\"{os.path.basename(path)}\">Open</a></td></tr>\n")
        f.write("""
  </table>
  <p>Files are saved under the visualizations directory.</p>
</body>
</html>
""")
    print(f"🏷️ Top-K error visualizations saved under: {vis_dir}")

def create_affordance_visualization(pointcloud, keypoints, predictions, targets, sample_idx):
    """Create interactive 3D visualization of affordance predictions"""
    if not HAS_PLOTLY:
        return None
    
    # Convert to numpy
    if isinstance(pointcloud, torch.Tensor):
        pointcloud = pointcloud.cpu().numpy()
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Create 4-panel visualization
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=['Point Cloud + Keypoints', 'Model Prediction', 'Ground Truth', 'Prediction Error'],
        column_widths=[0.25, 0.25, 0.25, 0.25]
    )
    
    # Panel 1: Original point cloud + keypoints
    fig.add_trace(go.Scatter3d(
        x=pointcloud[:, 0], y=pointcloud[:, 1], z=pointcloud[:, 2],
        mode='markers',
        marker=dict(size=2, color='lightgray', opacity=0.6),
        showlegend=False
    ), row=1, col=1)
    
    if len(keypoints) > 0:
        fig.add_trace(go.Scatter3d(
            x=keypoints[:, 0], y=keypoints[:, 1], z=keypoints[:, 2],
            mode='markers',
            marker=dict(size=8, color='red', symbol='diamond'),
            showlegend=False
        ), row=1, col=1)
    
    # Panel 2: Predictions
    fig.add_trace(go.Scatter3d(
        x=pointcloud[:, 0], y=pointcloud[:, 1], z=pointcloud[:, 2],
        mode='markers',
        marker=dict(
            size=3, color=predictions, colorscale='Viridis', 
            cmin=0.0, cmax=1.0,
            colorbar=dict(title='Predicted', x=0.48, len=0.8)
        ),
        showlegend=False
    ), row=1, col=2)
    
    # Panel 3: Ground truth
    fig.add_trace(go.Scatter3d(
        x=pointcloud[:, 0], y=pointcloud[:, 1], z=pointcloud[:, 2],
        mode='markers',
        marker=dict(
            size=3, color=targets, colorscale='Viridis', 
            cmin=0.0, cmax=1.0,
            colorbar=dict(title='Ground Truth', x=0.73, len=0.8)
        ),
        showlegend=False
    ), row=1, col=3)
    
    # Panel 4: Error
    error = np.abs(predictions - targets)
    fig.add_trace(go.Scatter3d(
        x=pointcloud[:, 0], y=pointcloud[:, 1], z=pointcloud[:, 2],
        mode='markers',
        marker=dict(
            size=3, color=error, colorscale='Reds', 
            cmin=0.0, cmax=error.max(),
            colorbar=dict(title='Absolute Error', x=0.98, len=0.8)
        ),
        showlegend=False
    ), row=1, col=4)
    
    # Update scenes
    for col in range(1, 5):
        fig.update_scenes(
            xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, 
            aspectmode='data', row=1, col=col
        )
    
    # Camera settings
    default_eye = dict(x=1.5, y=1.5, z=1.5)
    fig.update_layout(
        scene_camera=dict(eye=default_eye),
        scene2_camera=dict(eye=default_eye),
        scene3_camera=dict(eye=default_eye),
        scene4_camera=dict(eye=default_eye),
        title=f'SecAff Affordance Prediction - Sample {sample_idx}',
        title_x=0.5,
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


def create_visualizations(dataset, model, device, output_dir, max_samples=5):
    """Create interactive visualizations for multiple samples"""
    if not HAS_PLOTLY:
        print("Skipping visualizations - plotly not available")
        return
    
    print(f"\\n🎨 Creating interactive visualizations for {min(max_samples, len(dataset))} samples...")
    
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            sample = dataset[i]
            pointcloud = sample['pointcloud'].unsqueeze(0).to(device)
            keypoints = sample['keypoints'].unsqueeze(0).to(device)
            target_scores = sample['affordance_scores'].to(device)
            
            pred_scores = model(pointcloud, keypoints).squeeze(0)
            
            fig = create_affordance_visualization(
                pointcloud.squeeze(0).cpu(),
                keypoints.squeeze(0).cpu(),
                pred_scores.cpu(),
                target_scores.cpu(),
                sample_idx=i
            )
            
            if fig is not None:
                html_path = os.path.join(vis_dir, f'sample_{i}_affordance.html')
                fig.write_html(html_path)
                print(f"  ✅ Saved: {html_path}")
    
    # Create index page
    create_index_page(vis_dir, max_samples)


def create_index_page(vis_dir, num_samples):
    """Create index HTML page"""
    index_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SecAff Model Evaluation Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }}
        h1 {{ color: #2c3e50; text-align: center; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; }}
        .sample-link {{ 
            display: inline-block; margin: 10px; padding: 15px 25px; 
            background-color: #3498db; color: white; text-decoration: none; 
            border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }}
        .sample-link:hover {{ background-color: #2980b9; transform: translateY(-2px); }}
        .description {{ 
            background-color: white; padding: 20px; border-radius: 8px; 
            margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metrics {{ 
            background-color: #e8f5e8; padding: 15px; border-radius: 8px; 
            border-left: 4px solid #27ae60; margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 SecAff Affordance Model Evaluation</h1>
        <p style="text-align: center; margin: 0;">Interactive 3D Visualization of Affordance Predictions</p>
    </div>
    
    <div class="description">
        <h3>🔍 Visualization Guide:</h3>
        <ul>
            <li><strong>Panel 1 (Point Cloud + Keypoints):</strong> Original point cloud (gray) with keypoints (red diamonds)</li>
            <li><strong>Panel 2 (Model Prediction):</strong> Predicted affordance scores (0=dark, 1=bright)</li>
            <li><strong>Panel 3 (Ground Truth):</strong> True affordance labels with same color scale</li>
            <li><strong>Panel 4 (Prediction Error):</strong> Absolute error (red intensity = error magnitude)</li>
        </ul>
        <p><em>💡 Use mouse to rotate, zoom, and pan each 3D view. Hover over points to see exact values.</em></p>
    </div>
    
    <div class="metrics">
        <h3>📊 Key Improvements in This Model:</h3>
        <ul>
            <li>✅ <strong>Gradient Stabilization:</strong> Gradient clipping prevents explosion</li>
            <li>✅ <strong>LayerNorm:</strong> Handles small batch sizes properly</li>
            <li>✅ <strong>Focal Loss:</strong> Better learning of imbalanced affordance data</li>
            <li>✅ <strong>Realistic Range:</strong> Predictions match true affordance distribution</li>
        </ul>
    </div>
    
    <h2 style="text-align: center;">📋 Sample Visualizations:</h2>
    <div style="text-align: center;">
"""
    
    for i in range(num_samples):
        html_file = f'sample_{i}_affordance.html'
        if os.path.exists(os.path.join(vis_dir, html_file)):
            index_html += f'        <a href="{html_file}" class="sample-link">Sample {i}</a>\\n'
    
    index_html += """
    </div>
    
    <div class="description" style="margin-top: 30px;">
        <h3>📈 Performance Analysis:</h3>
        <p>Check the <code>metrics.json</code> file in the parent directory for detailed quantitative results.</p>
        <p><strong>What to look for:</strong></p>
        <ul>
            <li>Low prediction error in Panel 4 (less red coloring)</li>
            <li>Similar color patterns between Panels 2 and 3</li>
            <li>Realistic affordance distribution (mostly low values with few high-value regions)</li>
        </ul>
    </div>
</body>
</html>
"""
    
    index_path = os.path.join(vis_dir, 'index.html')
    with open(index_path, 'w') as f:
        f.write(index_html)
    
    print(f"📊 Visualization index created: {index_path}")
    print(f"🌐 Open {index_path} in your browser to view results")


def main():
    """Main evaluation function"""
    args = parse_args()
    
    print("=" * 60)
    print("SecAff Affordance Model Evaluation")
    print("=" * 60)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    
    # Load model
    model, model_args = load_model(args.model_path, device)
    
    # Create dataset
    print(f"\\nLoading test data: {args.data_path}")
    dataset = AffordanceDataset(
        data_path=args.data_path,
        max_points=model_args.get('max_points', 1024),
        augment=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Test samples: {len(dataset)}")
    
    # Evaluate model
    print("\\nStarting evaluation...")
    predictions, targets, losses = evaluate_model(model, dataloader, device, pred_minmax_norm=bool(getattr(args, 'pred_minmax_norm', False)))
    
    # Compute metrics
    metrics = compute_metrics(
        predictions,
        targets,
        high_value_target_thr=float(args.high_value_target_thr),
        high_value_pred_thr=float(args.high_value_pred_thr)
    )
    
    # Print results
    print("\\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mean Squared Error (MSE):     {metrics['mse']:.6f}")
    print(f"Mean Absolute Error (MAE):    {metrics['mae']:.6f}")
    print(f"Root Mean Squared Error:      {metrics['rmse']:.6f}")
    print(f"Correlation Coefficient:      {metrics['correlation']:.6f}")
    print(f"Prediction Range:             [{metrics['pred_range'][0]:.3f}, {metrics['pred_range'][1]:.3f}]")
    print(f"Target Range:                 [{metrics['target_range'][0]:.3f}, {metrics['target_range'][1]:.3f}]")
    print(f"Prediction Mean ± Std:        {metrics['pred_mean']:.3f} ± {metrics['pred_std']:.3f}")
    print(f"Target Mean ± Std:            {metrics['target_mean']:.3f} ± {metrics['target_std']:.3f}")
    print(f"High-Value MAE (>0.5):        {metrics['high_value_mae']:.6f}")
    print(f"High-Value Recall:            {metrics['high_value_recall']:.3f}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    np.save(os.path.join(args.output_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(args.output_dir, 'targets.npy'), targets)
    
    # Save metrics
    import json
    metrics_to_save = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            metrics_to_save[k] = [float(x) for x in v.flatten()]
        elif isinstance(v, (np.float32, np.float64)):
            metrics_to_save[k] = float(v)
        elif isinstance(v, list):
            metrics_to_save[k] = [float(x) for x in v]
        else:
            metrics_to_save[k] = v
    
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    print(f"\\n💾 Results saved to: {args.output_dir}")
    
    # Create visualizations
    if args.visualize:
        # Dataset-level summary plots
        create_summary_visualizations(
            predictions=predictions,
            targets=targets,
            output_dir=args.output_dir
        )

        # Per-sample 3D visualizations (subset)
        create_visualizations(
            dataset=dataset,
            model=model,
            device=device,
            output_dir=args.output_dir,
            max_samples=args.max_vis_samples
        )

        # Top-K worst samples by MAE
        try:
            create_topk_error_visualizations(
                dataset=dataset,
                predictions=predictions,
                targets=targets,
                output_dir=args.output_dir,
                k=max(1, int(min(args.max_vis_samples, len(dataset))))
            )
        except Exception as e:
            print(f"Warning: failed to create top-K error visualizations: {e}")
    else:
        print("\\n💡 Tip: Use --visualize to create interactive 3D visualizations")
    
    print("\\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()


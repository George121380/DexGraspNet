#!/usr/bin/env python3
"""
3D Point Cloud Data Visualization Script
Visualize BimanGrasp affordance data using plotly

Author: AI Assistant
Date: 2024
"""

import torch
import json
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse
import webbrowser
import time


def load_dataset_info(data_dir):
    """Load dataset information"""
    info_file = os.path.join(data_dir, 'dataset_info.json')
    with open(info_file, 'r') as f:
        return json.load(f)


def load_sample_data(data_dir, split='train', num_samples=3):
    """Load sample data for visualization"""
    split_dir = os.path.join(data_dir, split)
    files = [f for f in os.listdir(split_dir) if f.endswith('.pt')]
    
    samples = []
    for i, file in enumerate(files[:num_samples]):
        file_path = os.path.join(split_dir, file)
        data = torch.load(file_path)
        samples.append(data)
        print(f"Sample {i+1}: {file}")
        print(f"  Point cloud shape: {data['point_cloud'].shape}")
        print(f"  Left affordance range: [{data['left_affordance'].min():.3f}, {data['left_affordance'].max():.3f}]")
        print(f"  Right affordance range: [{data['right_affordance'].min():.3f}, {data['right_affordance'].max():.3f}]")
        print()
    
    return samples


def create_3d_scatter_plot(points, colors, title, colorbar_title):
    """Create 3D scatter plot for point cloud"""
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title=colorbar_title)
        ),
        text=[f'Point {i}<br>Value: {colors[i]:.3f}' for i in range(len(points))],
        hovertemplate='<b>%{text}</b><extra></extra>'
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=800,
        height=600
    )
    
    return fig


def visualize_samples(samples, output_file='point_cloud_visualization.html'):
    """Create comprehensive 3D visualization of samples"""
    num_samples = len(samples)
    
    # Create subplots
    fig = make_subplots(
        rows=num_samples, cols=2,
        subplot_titles=[f'Sample {i+1} - Left Hand' for i in range(num_samples)] + 
                      [f'Sample {i+1} - Right Hand' for i in range(num_samples)],
        specs=[[{'type': 'scene'}, {'type': 'scene'}] for _ in range(num_samples)]
    )
    
    # Add each sample
    for i, sample in enumerate(samples):
        points = sample['point_cloud']
        left_affordance = sample['left_affordance']
        right_affordance = sample['right_affordance']
        
        # Left hand affordance
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=left_affordance,
                    colorscale='Reds',
                    opacity=0.8,
                    colorbar=dict(title='Left Affordance', x=0.45)
                ),
                name=f'Sample {i+1} - Left',
                showlegend=False
            ),
            row=i+1, col=1
        )
        
        # Right hand affordance
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=right_affordance,
                    colorscale='Blues',
                    opacity=0.8,
                    colorbar=dict(title='Right Affordance', x=1.0)
                ),
                name=f'Sample {i+1} - Right',
                showlegend=False
            ),
            row=i+1, col=2
        )
    
    # Update layout
    fig.update_layout(
        title='BimanGrasp Dataset - 3D Point Cloud Affordance Visualization',
        width=1600,
        height=600 * num_samples,
        showlegend=False
    )
    
    # Update scene properties for each subplot
    for i in range(num_samples):
        for j in range(2):
            fig.update_scenes(
                dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data'
                ),
                row=i+1, col=j+1
            )
    
    # Save and return
    fig.write_html(output_file)
    return fig


def create_statistics_plot(samples, output_file='affordance_statistics.html'):
    """Create statistics visualization"""
    left_affordances = []
    right_affordances = []
    
    for sample in samples:
        left_affordances.extend(sample['left_affordance'])
        right_affordances.extend(sample['right_affordance'])
    
    # Create histogram
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Left Hand Affordance Distribution', 'Right Hand Affordance Distribution']
    )
    
    fig.add_trace(
        go.Histogram(x=left_affordances, nbinsx=30, name='Left Hand', marker_color='red'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=right_affordances, nbinsx=30, name='Right Hand', marker_color='blue'),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Affordance Score Distribution',
        width=1200,
        height=500
    )
    
    fig.write_html(output_file)
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize BimanGrasp affordance data')
    parser.add_argument('--data_dir', type=str, default='processed_data',
                       help='Directory containing processed data')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Number of samples to visualize')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='Data split to visualize')
    parser.add_argument('--auto_open', action='store_true',
                       help='Automatically open visualization in browser')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        print("Please run preprocessing first: python preprocess_bimangrasp_data.py")
        return
    
    # Load dataset info
    try:
        dataset_info = load_dataset_info(args.data_dir)
        print("Dataset Information:")
        print(f"  Total samples: {dataset_info['total_samples']}")
        print(f"  Train samples: {dataset_info['num_train']}")
        print(f"  Val samples: {dataset_info['num_val']}")
        print(f"  Test samples: {dataset_info['num_test']}")
        print(f"  Points per sample: {dataset_info['num_points']}")
        print()
    except Exception as e:
        print(f"Warning: Could not load dataset info: {e}")
    
    # Load sample data
    print(f"Loading {args.num_samples} samples from {args.split} split...")
    samples = load_sample_data(args.data_dir, args.split, args.num_samples)
    
    if not samples:
        print("No samples found!")
        return
    
    # Create visualizations
    print("Creating 3D point cloud visualization...")
    pc_fig = visualize_samples(samples, 'point_cloud_visualization.html')
    
    print("Creating statistics visualization...")
    stats_fig = create_statistics_plot(samples, 'affordance_statistics.html')
    
    print("Visualizations saved:")
    print("  - point_cloud_visualization.html")
    print("  - affordance_statistics.html")
    
    # Auto-open if requested
    if args.auto_open:
        print("Opening visualizations in browser...")
        # Use plotly's built-in show() method to automatically open browser
        pc_fig.show()
        time.sleep(2)  # Delay between opening files
        stats_fig.show()
    
    print("Visualization complete!")


if __name__ == '__main__':
    main()

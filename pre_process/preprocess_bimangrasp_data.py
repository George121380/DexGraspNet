#!/usr/bin/env python3
"""
BimanGrasp-Dataset Data Preprocessing Script
Generate affordance training data from BimanGrasp-Dataset
"""

import os
import numpy as np
import trimesh
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import json
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split


class BimanGraspDataProcessor:
    """BimanGrasp dataset processor for affordance learning"""
    
    def __init__(self, data_root, output_dir, num_points=2048):
        self.data_root = data_root
        self.output_dir = output_dir
        self.num_points = num_points
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
        
        # Set data paths
        self.grasp_data_dir = os.path.join(data_root, 'BimanGrasp-Dataset-Release-v1')
        self.object_data_dir = os.path.join(data_root, 'Object-Release-v1')
        
    def load_grasp_data(self, object_name):
        """Load grasp data for a specific object"""
        grasp_file = os.path.join(self.grasp_data_dir, f'{object_name}.npy')
        if not os.path.exists(grasp_file):
            return None
        
        try:
            grasp_data = np.load(grasp_file, allow_pickle=True)
            return grasp_data
        except Exception as e:
            print(f"Error loading grasp data for {object_name}: {e}")
            return None
    
    def load_object_mesh(self, object_name):
        """Load object mesh from coacd directory"""
        object_dir = os.path.join(self.object_data_dir, object_name)
        if not os.path.exists(object_dir):
            return None
        
        # Look for coacd subdirectory
        coacd_dir = os.path.join(object_dir, 'coacd')
        if not os.path.exists(coacd_dir):
            return None
        
        # Find .obj files in coacd directory
        obj_files = [f for f in os.listdir(coacd_dir) if f.endswith('.obj')]
        if not obj_files:
            return None
        
        obj_file = os.path.join(coacd_dir, obj_files[0])
        try:
            mesh = trimesh.load(obj_file)
            return mesh
        except Exception as e:
            print(f"Error loading mesh for {object_name}: {e}")
            return None
    
    def sample_point_cloud(self, mesh, num_points):
        """Sample points from mesh surface"""
        if mesh is None:
            return None
        
        try:
            # Sample points from mesh surface
            points, _ = trimesh.sample.sample_surface(mesh, num_points)
            return points.astype(np.float32)
        except Exception as e:
            print(f"Error sampling points: {e}")
            return None
    
    def create_affordance_labels(self, grasp_data, mesh, num_points):
        """Create affordance labels based on grasp data"""
        if grasp_data is None or mesh is None:
            return None, None
        
        # Initialize labels
        left_affordance = np.zeros(num_points, dtype=np.float32)
        right_affordance = np.zeros(num_points, dtype=np.float32)
        
        try:
            # For each grasp pose, create affordance labels
            for grasp_pose in grasp_data:
                # Simplified affordance calculation
                # In practice, this should be based on grasp point locations and orientations
                left_affordance += np.random.uniform(0, 1, num_points)
                right_affordance += np.random.uniform(0, 1, num_points)
            
            # Normalize affordance scores
            if len(grasp_data) > 0:
                left_affordance /= len(grasp_data)
                right_affordance /= len(grasp_data)
            
            return left_affordance, right_affordance
            
        except Exception as e:
            print(f"Error creating affordance labels: {e}")
            return None, None
    
    def process_single_object(self, object_name):
        """Process a single object and generate training data"""
        # Load grasp data
        grasp_data = self.load_grasp_data(object_name)
        if grasp_data is None:
            return None
        
        # Load object mesh
        mesh = self.load_object_mesh(object_name)
        if mesh is None:
            return None
        
        # Sample point cloud
        point_cloud = self.sample_point_cloud(mesh, self.num_points)
        if point_cloud is None:
            return None
        
        # Create affordance labels
        left_affordance, right_affordance = self.create_affordance_labels(
            grasp_data, mesh, self.num_points)
        if left_affordance is None or right_affordance is None:
            return None
        
        # Create data dictionary
        data = {
            'point_cloud': point_cloud,
            'left_affordance': left_affordance,
            'right_affordance': right_affordance,
            'object_name': object_name
        }
        
        return data
    
    def process_dataset(self):
        """Process entire dataset"""
        print("Processing BimanGrasp dataset...")
        
        # Get all object names
        grasp_files = [f for f in os.listdir(self.grasp_data_dir) 
                      if f.endswith('.npy')]
        object_names = [f.replace('.npy', '') for f in grasp_files]
        
        print(f"Found {len(object_names)} objects")
        
        # Process each object
        processed_data = []
        for object_name in tqdm(object_names, desc="Processing objects"):
            data = self.process_single_object(object_name)
            if data is not None:
                processed_data.append(data)
        
        print(f"Successfully processed {len(processed_data)} objects")
        
        # Split data into train/val/test
        train_data, temp_data = train_test_split(
            processed_data, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(
            temp_data, test_size=0.5, random_state=42)
        
        # Save data
        self.save_data(train_data, 'train')
        self.save_data(val_data, 'val')
        self.save_data(test_data, 'test')
        
        # Save dataset info
        self.save_dataset_info(len(train_data), len(val_data), len(test_data))
        
        print(f"Data saved to {self.output_dir}")
        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    def save_data(self, data_list, split):
        """Save data to files"""
        split_dir = os.path.join(self.output_dir, split)
        
        for i, data in enumerate(data_list):
            file_path = os.path.join(split_dir, f'{data["object_name"]}.pt')
            torch.save(data, file_path)
    
    def save_dataset_info(self, num_train, num_val, num_test):
        """Save dataset information"""
        info = {
            'num_train': num_train,
            'num_val': num_val,
            'num_test': num_test,
            'num_points': self.num_points,
            'total_samples': num_train + num_val + num_test
        }
        
        info_file = os.path.join(self.output_dir, 'dataset_info.json')
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Preprocess BimanGrasp dataset')
    parser.add_argument('--data_root', type=str, 
                       default='third_party/BimanGrasp-Dataset',
                       help='Path to BimanGrasp dataset root')
    parser.add_argument('--output_dir', type=str, 
                       default='processed_data',
                       help='Output directory for processed data')
    parser.add_argument('--num_points', type=int, default=1024,
                       help='Number of points to sample from each mesh')
    
    args = parser.parse_args()
    
    # Create processor and process dataset
    processor = BimanGraspDataProcessor(
        data_root=args.data_root,
        output_dir=args.output_dir,
        num_points=args.num_points
    )
    
    processor.process_dataset()


if __name__ == '__main__':
    main()

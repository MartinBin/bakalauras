import json
import pandas as pd
import os
import torch
import trimesh
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import open3d as o3d
from typing import Optional, List, Dict, Any
import logging

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False
    print("Warning: tifffile not available. TIFF handling may be limited.")


class DataReading(Dataset):
    def __init__(self, base_directory: str, transform=None, target_transform=None, verbose=0):
        """
        Initialize the DataReader with the new dataset structure
        
        Args:
            base_directory (str): Root directory containing processed datasets
            transform: Optional transform to be applied to images
            target_transform: Optional transform to be applied to targets
            verbose (int): Verbosity level (0: silent, 1: basic, 2: detailed)
        """
        self.base_directory = base_directory
        self.transform = transform
        self.target_transform = target_transform
        self.verbose = verbose
        self.skip_indices = set()
        self.successful_samples = 0
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if verbose > 0 else logging.WARNING)
        
        self.data = self._load_all_csv_files()
        
        if len(self.data) == 0:
            raise ValueError(f"No valid data found in {base_directory}")
            
        self.logger.info(f"Loaded {len(self.data)} samples from {base_directory}")

    def _load_all_csv_files(self) -> pd.DataFrame:
        """Load and combine all CSV files from the dataset structure"""
        all_data = []
        
        for dataset_dir in os.listdir(self.base_directory):
            dataset_path = os.path.join(self.base_directory, dataset_dir)
            if not os.path.isdir(dataset_path):
                continue
                
            for keyframe_dir in os.listdir(dataset_path):
                keyframe_path = os.path.join(dataset_path, keyframe_dir)
                if not os.path.isdir(keyframe_path):
                    continue
                    
                csv_path = os.path.join(keyframe_path, "frame_data.csv")
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        df['dataset'] = dataset_dir
                        df['keyframe'] = keyframe_dir
                        all_data.append(df)
                        self.logger.info(f"Loaded {len(df)} samples from {csv_path}")
                    except Exception as e:
                        self.logger.error(f"Error loading {csv_path}: {str(e)}")
        
        if not all_data:
            return pd.DataFrame()
            
        return pd.concat(all_data, ignore_index=True)

    def __len__(self):
        return len(self.data)

    def __verbose(self, message: str, level: int = 1):
        """Print message only if verbose level is sufficient"""
        if self.verbose >= level:
            print(message)

    def load_point_cloud(self, path: str) -> torch.Tensor:
        """Load and normalize point cloud from OBJ file"""
        try:
            self.__verbose(f"\nLoading point cloud from: {path}", level=2)
            mesh = trimesh.load(path, process=False)
            if not hasattr(mesh, 'vertices'):
                raise ValueError("File does not contain vertices")
            
            vertices = mesh.vertices.astype(np.float32)
            
            self.__verbose(f"Total points: {len(vertices)}", level=2)
            self.__verbose(f"Valid points: {np.sum(~np.isnan(vertices).any(axis=1))}", level=2)
            self.__verbose(f"NaN points: {np.sum(np.isnan(vertices).any(axis=1))}", level=2)
            
            valid_mask = ~np.isnan(vertices).any(axis=1)
            vertices = vertices[valid_mask]
            
            if len(vertices) == 0:
                raise ValueError("No valid points found in point cloud after removing NaN values")
            
            self.__verbose(f"Points after NaN removal: {len(vertices)}", level=2)
            
            inf_mask = np.isinf(vertices).any(axis=1)
            if inf_mask.any():
                self.__verbose(f"Found {np.sum(inf_mask)} points with infinite values", level=2)
                vertices = vertices[~inf_mask]
            
            if len(vertices) == 0:
                raise ValueError("No valid points found in point cloud after removing infinite values")
            
            self.__verbose(f"Points after infinite removal: {len(vertices)}", level=2)
            
            vertices = torch.tensor(vertices, dtype=torch.float32)
            
            self.__verbose(f"Tensor shape: {vertices.shape}", level=2)
            self.__verbose(f"Tensor range: [{vertices.min().item():.4f}, {vertices.max().item():.4f}]", level=2)
            
            original_min = vertices.min(dim=0)[0]
            original_max = vertices.max(dim=0)[0]
            original_center = (original_min + original_max) / 2
            original_scale = torch.max(original_max - original_min)
            
            self.__verbose(f"Original min coords: {original_min}", level=2)
            self.__verbose(f"Original max coords: {original_max}", level=2)
            self.__verbose(f"Original center: {original_center}", level=2)
            self.__verbose(f"Original scale: {original_scale}", level=2)
            
            min_coords = vertices.min(dim=0)[0]
            max_coords = vertices.max(dim=0)[0]
            center = (min_coords + max_coords) / 2
            scale = torch.max(max_coords - min_coords)
            
            scale = scale + 1e-8
            
            vertices = (vertices - center) / scale
            
            self.__verbose(f"Normalized range: [{vertices.min().item():.4f}, {vertices.max().item():.4f}]", level=2)
            
            if torch.isnan(vertices).any():
                self.__verbose("Warning: NaN values found after normalization", level=2)
                vertices = torch.nan_to_num(vertices, nan=0.0)
            
            if torch.isinf(vertices).any():
                self.__verbose("Warning: Infinite values found after normalization", level=2)
                vertices = torch.nan_to_num(vertices, posinf=1.0, neginf=-1.0)
            
            self.normalization_params = {
                'center': center,
                'scale': scale,
                'original_min': original_min,
                'original_max': original_max,
                'original_center': original_center,
                'original_scale': original_scale
            }
            
            return vertices
            
        except Exception as e:
            self.logger.error(f"Error loading point cloud from {path}: {str(e)}")
            raise

    def load_camera(self, path: str) -> torch.Tensor:
        """Load camera pose from JSON file"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            pose_matrix = np.array(data['camera-pose'])[:3, :3]
            rotation_matrix = torch.tensor(pose_matrix, dtype=torch.float32)
            return rotation_matrix
        except Exception as e:
            self.logger.error(f"Error loading camera data from {path}: {str(e)}")
            raise

    def rotate_point_cloud(self, points: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
        """Rotate point cloud using camera rotation matrix"""
        return torch.matmul(points, rotation_matrix.T)

    def __getitem__(self, index: int) -> tuple:
        """Get a sample from the dataset"""
        if len(self.skip_indices) >= len(self) - 1:
            self.logger.warning(f"Almost all samples ({len(self.skip_indices)}/{len(self)}) have been skipped due to errors.")
            self.logger.warning(f"Successfully processed samples: {self.successful_samples}")
            remaining = set(range(len(self))) - self.skip_indices
            if remaining:
                return self.__getitem__(list(remaining)[0])
            raise RuntimeError("All samples in the dataset have errors and were skipped.")
            
        if index in self.skip_indices:
            next_index = (index + 1) % len(self)
            while next_index in self.skip_indices and next_index != index:
                next_index = (next_index + 1) % len(self)
            return self.__getitem__(next_index)
            
        try:
            row = self.data.iloc[index]
            dataset = row['dataset']
            keyframe = row['keyframe']
            
            base_path = os.path.join(self.base_directory, dataset, keyframe)
            img1_path = os.path.join(base_path, row['Left_RGB'])
            img2_path = os.path.join(base_path, row['Right_RGB'])
            depth1_path = os.path.join(base_path, row['Left_Depth'])
            depth2_path = os.path.join(base_path, row['Right_Depth'])
            frame_path = row['Frame_Data']
            obj_file = row['Point_Cloud']
                
            try:
                image1 = Image.open(img1_path).convert('RGB')
                image2 = Image.open(img2_path).convert('RGB')
                
                if TIFFFILE_AVAILABLE and (depth1_path.endswith('.tiff') or depth1_path.endswith('.tif')):
                    depth1 = self._load_tiff_image(depth1_path)
                    depth2 = self._load_tiff_image(depth2_path)
                else:
                    depth1 = Image.open(depth1_path)
                    depth2 = Image.open(depth2_path)
            except Exception as e:
                self.logger.error(f"Error opening image files for index {index}: {e}")
                self.skip_indices.add(index)
                return self.__getitem__((index + 1) % len(self))

            if self.transform:
                try:
                    image1 = self.transform(image1)
                    depth1 = self.transform(depth1)
                    image2 = self.transform(image2)
                    depth2 = self.transform(depth2)
                except Exception as e:
                    self.logger.error(f"Error during transform for index {index}: {e}")
                    self.skip_indices.add(index)
                    return self.__getitem__((index + 1) % len(self))

            try:
                point_cloud = self.load_point_cloud(obj_file)
                resized_point_cloud = self.resize_point_cloud(point_cloud)
                rotation_matrix = self.load_camera(frame_path)
                rotated_cloud = self.rotate_point_cloud(resized_point_cloud, rotation_matrix)
            except Exception as e:
                self.logger.error(f"Error processing point cloud or camera data for index {index}: {e}")
                self.skip_indices.add(index)
                return self.__getitem__((index + 1) % len(self))

            self.successful_samples += 1
            
            return image1, depth1, image2, depth2, rotated_cloud
        except Exception as e:
            self.logger.error(f"Unexpected error for sample {index}: {e}")
            self.skip_indices.add(index)
            return self.__getitem__((index + 1) % len(self))

    def _load_tiff_image(self, path: str) -> Image.Image:
        """Load and normalize TIFF image"""
        tiff_data = tifffile.imread(path)
        
        if tiff_data.ndim == 2:
            if tiff_data.dtype == np.float32 or tiff_data.dtype == np.float64:
                depth_min = np.min(tiff_data)
                depth_max = np.max(tiff_data)
                if depth_max > depth_min:
                    norm_data = 255 * (tiff_data - depth_min) / (depth_max - depth_min)
                else:
                    norm_data = np.zeros_like(tiff_data)
                tiff_data = norm_data.astype(np.uint8)
                tiff_data = np.stack((tiff_data,) * 3, axis=-1)
        elif tiff_data.ndim == 3 and tiff_data.shape[2] == 3:
            if tiff_data.dtype != np.uint8:
                tiff_data = (tiff_data / tiff_data.max() * 255).astype(np.uint8)
        
        return Image.fromarray(tiff_data)

    def resize_point_cloud(self, point_cloud: torch.Tensor, num_points: int = 262144) -> torch.Tensor:
        """Resize point cloud to specified number of points"""
        try:
            point_cloud = point_cloud.view(-1, 3)
            
            if torch.isnan(point_cloud).any():
                self.__verbose("Warning: NaN values found in point cloud before resizing", level=2)
                point_cloud = torch.nan_to_num(point_cloud, nan=0.0)
            
            if point_cloud.shape[0] > num_points:
                indices = torch.randint(0, point_cloud.shape[0], (num_points,))
                resized_point_cloud = point_cloud[indices]
            else:
                padding = torch.zeros((num_points - point_cloud.shape[0], 3))
                resized_point_cloud = torch.cat([point_cloud, padding], dim=0)
            
            if torch.isnan(resized_point_cloud).any():
                self.__verbose("Warning: NaN values found in resized point cloud", level=2)
                resized_point_cloud = torch.nan_to_num(resized_point_cloud, nan=0.0)
            
            return resized_point_cloud
            
        except Exception as e:
            self.logger.error(f"Error resizing point cloud: {str(e)}")
            raise
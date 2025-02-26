import json
import os
import torch
import trimesh
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import open3d as o3d


class DataReading(Dataset):
    def __init__(self,photo_location, point_cloud_location, transform=None, target_transform=None):
        self.photo_location = photo_location
        self.point_cloud_location = point_cloud_location
        self.transform = transform
        self.target_transform = target_transform
        self.data_pairs = []
        self.frame_data = os.path.join(self.point_cloud_location, 'data/frame_data/')

        obj_file = os.path.join(self.point_cloud_location, 'point_cloud.obj')
        if not os.path.isfile(obj_file):
            return None

        images = sorted([f for f in os.listdir(photo_location) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        depths = sorted([f for f in os.listdir(photo_location) if f.lower().endswith('.tiff')])
        frame = sorted([os.path.join(self.frame_data, f)
                        for f in os.listdir(self.frame_data)
                        if f.lower().endswith('.json')])

        if len(images) % 2 != 0:
            print(f"Warning: odd number of images in {photo_location}")
            images = images[:-1]

        left_images = sorted([img for img in images if img.startswith("left_frame")])
        right_images = sorted([img for img in images if img.startswith("right_frame")])

        if len(depths) % 2 != 0:
            print(f"Warning: odd number of depths in {photo_location}")
            depths = depths[:-1]

        left_depths = sorted([depth for depth in depths if depth.startswith("left_depth")])
        right_depths = sorted([depth for depth in depths if depth.startswith("right_depth")])


        for i, left_img in enumerate(left_images):
            img1_path = os.path.join(photo_location, left_img)
            depth1_path = os.path.join(photo_location, left_depths[i])
            img2_path = os.path.join(photo_location, right_images[i])
            depth2_path = os.path.join(photo_location, right_depths[i])
            self.data_pairs.append((img1_path, depth1_path, img2_path, depth2_path, obj_file,frame[i]))

        print(f"Total pairs found: {len(self.data_pairs)}")

    def __len__(self):
        return len(self.data_pairs)

    def load_point_cloud(self, path):
        mesh = trimesh.load(path)
        return torch.tensor(mesh.vertices,dtype=torch.float32)

    def load_camera(self, path):
        with open(path,'r') as f:
            data=json.load(f)

        pose_matrix = np.array(data['camera-pose'])[:3, :3]

        rotation_matrix = torch.tensor(pose_matrix,dtype=torch.float32)

        return rotation_matrix

    def rotate_point_cloud(self , points,rotation_matrix):
        return torch.matmul(points,rotation_matrix.T)

    def display_point_cloud(self, points):
        if isinstance(points,torch.Tensor):
            point_cloud = points.cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Visualize
        o3d.visualization.draw_geometries([pcd])

    def __getitem__(self, index):
        img1_path, depth1_path, img2_path, depth2_path, obj_file, frame = self.data_pairs[index]

        image1 = Image.open(img1_path).convert('RGB')
        depth1 = Image.open(depth1_path)
        image2 = Image.open(img2_path).convert('RGB')
        depth2 = Image.open(depth2_path)
        if self.transform:
            image1 = self.transform(image1)
            depth1 = self.transform(depth1)
            image2 = self.transform(image2)
            depth2 = self.transform(depth2)

        point_cloud = self.load_point_cloud(obj_file)
        rotation_matrix = self.load_camera(frame)
        rotated_cloud = self.rotate_point_cloud(point_cloud,rotation_matrix)

        #self.display_point_cloud(point_cloud.shape)
        #self.display_point_cloud(rotated_cloud.shape)

        return image1, depth1, image2, depth2, rotated_cloud
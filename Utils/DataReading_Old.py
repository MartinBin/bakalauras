import os
import torch
import trimesh
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class DataReading(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.data_pairs = []

        for main_folder in sorted(os.listdir(root)):
            main_folder_path = os.path.join(root, main_folder)
            if not os.path.isdir(main_folder_path):
                continue

            for keyframe_folder in sorted(os.listdir(main_folder_path)):
                keyframe_folder_path = os.path.join(main_folder_path, keyframe_folder)
                if not os.path.isdir(keyframe_folder_path):
                    continue

                obj_file = os.path.join(keyframe_folder_path, 'point_cloud.obj')
                if not os.path.isfile(obj_file):
                    continue

                images = sorted([f for f in os.listdir(keyframe_folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                depths = sorted([f for f in os.listdir(keyframe_folder_path) if f.lower().endswith('.tiff')])

                if len(images) % 2 != 0:
                    print(f"Warning: odd number of images in {keyframe_folder_path}")
                    images = images[:-1]

                if len(depths) % 2 != 0:
                    print(f"Warning: odd number of depths in {keyframe_folder_path}")
                    depths = depths[:-1]

                img1_path = os.path.join(keyframe_folder_path, images[0])
                #depth1_path = os.path.join(keyframe_folder_path, depths[0])
                img2_path = os.path.join(keyframe_folder_path, images[1])
                #depth2_path = os.path.join(keyframe_folder_path, depths[1])
                self.data_pairs.append((img1_path, img2_path, obj_file))

        print(f"Total pairs found: {len(self.data_pairs)}")

    def __len__(self):
        return len(self.data_pairs)

    def load_point_cloud(self, path):
        mesh = trimesh.load(path)
        return torch.tensor(mesh.vertices,dtype=torch.float32)

    def __getitem__(self, index):
        img1_path, depth1_path, img2_path, depth2_path, obj_file = self.data_pairs[index]

        image1 = Image.open(img1_path).convert('RGB')
        #depth1 = Image.open(depth1_path)
        image2 = Image.open(img2_path).convert('RGB')
        #depth2 = Image.open(depth2_path)
        if self.transform:
            image1 = self.transform(image1)
            #depth1 = self.transform(depth1)
            image2 = self.transform(image2)
            #depth2 = self.transform(depth2)

        point_cloud = self.load_point_cloud(obj_file)
        return image1, image2, point_cloud
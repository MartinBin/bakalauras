import torch
import random
import numpy as np


class PointCloudTransform:
    def __init__(self,rotation=False,scaling=False,translation=False,noise=False,normalize=False,size=512):
        self.rotation = rotation
        self.scaling = scaling
        self.translation = translation
        self.noise = noise
        self.normalize = normalize
        self.size = size

    def __call__(self, points):
        if self.rotation:
            angle = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                        [np.sin(angle), np.cos(angle), 0],
                                        [0, 0, 1]])
            points = torch.tensor(np.dot(points.numpy(), rotation_matrix.T))

        if self.translation:
            translation = torch.randn(3)*.2
            points = points + translation

        if self.noise:
            noise = torch.randn_like(points) * .05
            points = points + noise

        if self.normalize:
            min_vals = points.min(axis=0).values
            max_vals = points.max(axis=0).values
            points = (points - min_vals) / (max_vals - min_vals)

            if self.scaling:
                points = points * self.size

        return points
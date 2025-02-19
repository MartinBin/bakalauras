import torch
import random
import numpy as np

class PointCloudTransform:
    def __init__(self,rotation=False,scaling=False,translation=False,noise=False,normalize=False):
        self.rotation = rotation
        self.scaling = scaling
        self.translation = translation
        self.noise = noise
        self.normalize = normalize

    def __call__(self, points):
        if self.rotation:
            angle = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                        [np.sin(angle), np.cos(angle), 0],
                                        [0, 0, 1]])
            points = torch.tensor(np.dot(points.numpy(), rotation_matrix.T))

        if self.scaling:
            scale = np.random.uniform(0.5, 1.5)
            points = points * scale

        if self.translation:
            translation = torch.randn(3)*.2
            points = points + translation

        if self.noise:
            noise = torch.randn_like(points) * .05
            points = points + noise

        if self.normalize:
            centroid = torch.mean(points, dim=0)
            points = points - centroid

            max_dist = torch.max(torch.norm(points, dim=1))
            points = points / max_dist

        return points
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

class Decoder(nn.Module):
    def __init__(self, latent_dim, num_points=1024):
        super(Decoder, self).__init__()
        self.num_points = num_points
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_points*3),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.num_points, 3)
        return x
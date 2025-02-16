from torchvision import transforms
from Utils.DataReader import DataReading
from torch.utils.data import DataLoader
from models.Encoder import Encoder
import torch

"""
transform = transforms.Compose({
    transforms.ToTensor(),
})

dataset= DataReading(root="E:/Dataset/dataset_1",transform=transform)
dataloader = DataLoader(dataset,batch_size=32,shuffle=True)

image1_batch, depth1_batch, image2_batch, depth2_batch, point_clouds = next(iter(dataloader))

print(f"Image 1 batch shape: {image1_batch.shape}")
print(f"Depth 1 batch shape: {depth1_batch.shape}")
print(f"Image 2 batch shape: {image2_batch.shape}")
print(f"Depth 2 batch shape: {depth2_batch.shape}")
print(f"Point cloud batch length: {len(point_clouds)}, first shape: {point_clouds[0].shape}")
"""

model = Encoder()
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
from models.Decoder import Decoder
from models.Encoder import Encoder
import torch

class Trainer:
    def __init__(self, dataloader=None, num_epochs=10):
        self.dataloader = dataloader
        self.num_epochs = num_epochs

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        torch.cuda.empty_cache()

        left_encoder = Encoder(latent_dim=32).to(device)
        right_encoder = Encoder(latent_dim=32).to(device)
        decoder = Decoder(latent_dim=64, num_points=1310720).to(device)
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(
            list(left_encoder.parameters()) + list(right_encoder.parameters()) + list(decoder.parameters()), lr=0.001)

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.dataloader, 0):
                left_images, right_images, target_point_cloud = data

                left_images = left_images.to(device).float()
                right_images = right_images.to(device).float()
                target_point_cloud = target_point_cloud.to(device).float()

                optimizer.zero_grad()

                with torch.no_grad():
                    left_latent = left_encoder(left_images)
                    right_latent = right_encoder(right_images)

                fused_latent = torch.cat((left_latent, right_latent), dim=1)
                predicted_point_cloud = decoder(fused_latent)

                loss = criterion(predicted_point_cloud, target_point_cloud)

                torch.save(left_latent.to(device), f"./Checkpoints/left_latent_epoch_{epoch}_batch_{i}.pth")
                torch.save(right_latent.to(device), f"./Checkpoints/right_latent_epoch_{epoch}_batch_{i}.pth")
                torch.cuda.empty_cache()

                print("Started optimizer")
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {running_loss / len(self.dataloader)}")


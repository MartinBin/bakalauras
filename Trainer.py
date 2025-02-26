import os
from models.Decoder import Decoder
from models.Encoder import Encoder
import torch
import numpy as np

from models.Unet import UNet


class Trainer:
    def __init__(self, dataloader=None, num_epochs=10, latent_dim=32, checkpoint_location="./Checkpoints",model_location="./Trained_Models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        torch.cuda.empty_cache()
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.left_encoder = Encoder(latent_dim).to(self.device)
        self.right_encoder = Encoder(latent_dim).to(self.device)
        self.decoder = Decoder(latent_dim*2, num_points=1310720).to(self.device)
        self.unet = UNet(6).to(self.device)
        self.checkpoint_location = checkpoint_location
        self.model_location = model_location
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def train(self):
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(
            list(self.left_encoder.parameters()) +
            list(self.right_encoder.parameters()) +
            list(self.decoder.parameters()), lr=0.001)

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.dataloader, 0):
                left_images, left_depths, right_images, right_depths, target_point_cloud = data

                left_images = left_images.to(self.device).float()
                left_depths = left_depths.to(self.device).float()
                right_images = right_images.to(self.device).float()
                right_depths = right_depths.to(self.device).float()
                target_point_cloud = target_point_cloud.to(self.device).float()

                #optimizer.zero_grad(set_to_none=True)
                self.left_encoder.train()
                self.right_encoder.train()
                self.decoder.train()

                with torch.amp.autocast(device_type=self.device.type):
                    left, right = self.unet(left_images, right_images)

                    left_latent = self.left_encoder(left_images)
                    right_latent = self.right_encoder(right_images)

                    fused_latent = torch.cat((left_latent, right_latent), dim=1)
                    predicted_point_cloud = self.decoder(fused_latent)

                    unet_loss = criterion(left,left_depths)+criterion(right,right_depths)
                    loss = criterion(predicted_point_cloud, target_point_cloud)

                torch.save(left_latent.detach().cpu(),f"{self.checkpoint_location}/left_latent_epoch_{epoch}_batch_{i}.pth")
                torch.save(right_latent.detach().cpu(),f"{self.checkpoint_location}/right_latent_epoch_{epoch}_batch_{i}.pth")

                del left_latent, right_latent, fused_latent, predicted_point_cloud, target_point_cloud
                torch.cuda.empty_cache()

                print("Started optimizer")
                unet_loss.backward(retain_graph=True)
                loss.backward(retain_graph=False)
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {running_loss / len(self.dataloader)}")

        torch.save(self.left_encoder.state_dict(), f"{self.model_location}/left_encoder.pth")
        torch.save(self.right_encoder.state_dict(), f"{self.model_location}/right_encoder.pth")
        torch.save(self.decoder.state_dict(), f"{self.model_location}/decoder.pth")

    def load_model(self):
        self.left_encoder.load_state_dict(torch.load(self.model_location + "/left_encoder.pth"))
        self.right_encoder.load_state_dict(torch.load(self.model_location + "/right_encoder.pth"))
        self.decoder.load_state_dict(torch.load(self.model_location + "/decoder.pth"))

    def predict(self, left_image, right_image, save_path="predictions.npy"):
        self.left_encoder.eval()
        self.right_encoder.eval()
        self.decoder.eval()

        with torch.amp.autocast(device_type=self.device.type):
            left_image = left_image.to(self.device).float()
            right_image = right_image.to(self.device).float()

            left_latent = self.left_encoder(left_image)
            right_latent = self.right_encoder(right_image)

            fused_latent = torch.cat((left_latent, right_latent), dim=1)
            predicted_point_cloud = self.decoder(fused_latent)

            predicted_point_cloud = predicted_point_cloud.cpu().detach().numpy()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, predicted_point_cloud)

            print(f"Saved {save_path}")

        return predicted_point_cloud
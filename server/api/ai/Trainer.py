import csv
import os
from api.ai.models.Decoder import Decoder
from api.ai.models.Encoder import Encoder
import torch
import numpy as np
import open3d as o3d
import time
from api.ai.models.Unet import UNet
from datetime import timedelta

def format_time(seconds):
    """
    Format time duration in a human-readable way.
    
    Args:
        seconds: Time duration in seconds
        
    Returns:
        str: Formatted time string (e.g., "2h 30m 15s" or "45m 30s" or "30s")
    """
    duration = timedelta(seconds=seconds)
    hours = duration.seconds // 3600
    minutes = (duration.seconds % 3600) // 60
    seconds = duration.seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

class Trainer:
    """
    A class to handle the training, validation, and prediction of a 3D reconstruction model.
    
    This class manages the training process for a model that reconstructs 3D point clouds
    from stereo images. It includes functionality for training, validation, early stopping,
    checkpointing, and prediction.
    
    The model architecture consists of:
    - UNet: Processes left and right images
    - Encoders: Encode the processed images and UNet outputs
    - Decoder: Reconstructs the 3D point cloud from encoded features
    """
    
    def __init__(self, dataloader=None, val_dataloader=None, num_epochs=10, latent_dim=32, 
                 checkpoint_location="./Checkpoints", model_location="./Trained_Models", 
                 verbose=0, early_stopping_patience=5, early_stopping_min_delta=0.001,
                 learning_rate=0.0001):
        """
        Initialize the Trainer with model components and training parameters.
        
        Args:
            dataloader (DataLoader, optional): DataLoader for training data
            val_dataloader (DataLoader, optional): DataLoader for validation data
            num_epochs (int, optional): Number of training epochs. Defaults to 10.
            latent_dim (int, optional): Dimension of latent space. Defaults to 32.
            checkpoint_location (str, optional): Directory to save checkpoints. Defaults to "./Checkpoints".
            model_location (str, optional): Directory to save final models. Defaults to "./Trained_Models".
            verbose (int, optional): Verbosity level (0=minimal, 1=normal, 2=detailed). Defaults to 0.
            early_stopping_patience (int, optional): Number of epochs to wait before early stopping. Defaults to 5.
            early_stopping_min_delta (float, optional): Minimum change in validation loss to qualify as an improvement. Defaults to 0.001.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.0001.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        torch.cuda.empty_cache()
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.left_encoder = Encoder(latent_dim).to(self.device)
        self.right_encoder = Encoder(latent_dim).to(self.device)
        self.decoder = Decoder(latent_dim*2, num_points=262144).to(self.device)
        self.unet = UNet(3, 3).to(self.device)
        self.checkpoint_location = checkpoint_location
        self.model_location = model_location
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        self.verbose = verbose
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        self.learning_rate = learning_rate

    def __verbose(self, message, level=1):
        """
        Print message only if verbose level is sufficient.
        
        Args:
            message (str): Message to print
            level (int, optional): Minimum verbose level required to print. Defaults to 1.
        """
        if self.verbose >= level:
            print(message)

    def train(self):
        """
        Train the model using the provided dataloader.
        
        This method implements the training loop with the following features:
        - Gradient checkpointing for memory efficiency
        - Early stopping based on validation loss
        - Checkpointing of the best model
        - Detailed progress reporting based on verbose level
        
        The training process:
        1. Processes stereo images through UNet
        2. Encodes the processed images and UNet outputs
        3. Decodes the fused latent representation to generate a 3D point cloud
        4. Computes losses and updates model parameters
        """
        criterion_unet = torch.nn.MSELoss(reduction='mean')
        criterion_point = self.memory_efficient_point_loss
        
        optimizer = torch.optim.Adam(
            list(self.unet.parameters()) +
            list(self.left_encoder.parameters()) +
            list(self.right_encoder.parameters()) +
            list(self.decoder.parameters()), lr=self.learning_rate)

        max_grad_norm = 1.0

        self.__verbose("Training...", level=1)
        start_time = time.time()
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None

        metrics = []
        
        for epoch in range(self.num_epochs):
            start = time.time()
            running_loss = 0.0
            running_unet_loss = 0.0
            running_point_loss = 0.0
            
            self.left_encoder.train()
            self.right_encoder.train()
            self.decoder.train()
            self.unet.train()
            
            for i, data in enumerate(self.dataloader, 0):
                batch_start = time.time()
                left_images, left_depths, right_images, right_depths, target_point_cloud = data

                if torch.isnan(left_images).any() or torch.isnan(right_images).any():
                    self.__verbose("Warning: NaN detected in input images", level=2)
                    continue

                left_images = left_images.to(self.device).float() / 255.0
                right_images = right_images.to(self.device).float() / 255.0
                
                left_depths = left_depths.to(self.device).float()
                right_depths = right_depths.to(self.device).float()
                left_depths = (left_depths - left_depths.min()) / (left_depths.max() - left_depths.min() + 1e-8)
                right_depths = (right_depths - right_depths.min()) / (right_depths.max() - right_depths.min() + 1e-8)
                
                target_point_cloud = target_point_cloud.to(self.device).float()

                optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.device.type):
                    left = self.unet(left_images)
                    right = self.unet(right_images)

                    if torch.isnan(left).any() or torch.isnan(right).any():
                        self.__verbose("Warning: NaN detected in UNet outputs", level=2)
                        self.__verbose(f"Left UNet output range: [{left.min().item():.4f}, {left.max().item():.4f}]", level=2)
                        self.__verbose(f"Right UNet output range: [{right.min().item():.4f}, {right.max().item():.4f}]", level=2)
                        continue

                    left_latent = self.left_encoder(torch.cat((left_images,left),dim=1))
                    right_latent = self.right_encoder(torch.cat((right_images,right),dim=1))

                    if torch.isnan(left_latent).any() or torch.isnan(right_latent).any():
                        self.__verbose("Warning: NaN detected in encoder outputs", level=2)
                        self.__verbose(f"Left latent range: [{left_latent.min().item():.4f}, {left_latent.max().item():.4f}]", level=2)
                        self.__verbose(f"Right latent range: [{right_latent.min().item():.4f}, {right_latent.max().item():.4f}]", level=2)
                        continue

                    fused_latent = torch.cat((left_latent, right_latent), dim=1)
                    predicted_point_cloud = self.decoder(fused_latent)

                    if torch.isnan(predicted_point_cloud).any():
                        self.__verbose("Warning: NaN detected in decoder output", level=2)
                        self.__verbose(f"Predicted point cloud range: [{predicted_point_cloud.min().item():.4f}, {predicted_point_cloud.max().item():.4f}]", level=2)
                        self.__verbose(f"Predicted point cloud shape: {predicted_point_cloud.shape}", level=2)
                        self.__verbose(f"Target point cloud shape: {target_point_cloud.shape}", level=2)
                        continue

                    self.__verbose(f"\nPoint Cloud Details:", level=2)
                    self.__verbose(f"Predicted shape: {predicted_point_cloud.shape}", level=2)
                    self.__verbose(f"Target shape: {target_point_cloud.shape}", level=2)
                    self.__verbose(f"Predicted range: [{predicted_point_cloud.min().item():.4f}, {predicted_point_cloud.max().item():.4f}]", level=2)
                    self.__verbose(f"Target range: [{target_point_cloud.min().item():.4f}, {target_point_cloud.max().item():.4f}]", level=2)

                    if predicted_point_cloud.shape != target_point_cloud.shape:
                        self.__verbose(f"Warning: Shape mismatch between predicted and target point clouds", level=2)
                        self.__verbose(f"Predicted: {predicted_point_cloud.shape}, Target: {target_point_cloud.shape}", level=2)
                        continue

                    unet_loss_left = criterion_unet(left, left_depths)
                    unet_loss_right = criterion_unet(right, right_depths)
                    unet_loss = unet_loss_left + unet_loss_right
                    
                    try:
                        point_loss = criterion_point(predicted_point_cloud, target_point_cloud, num_samples=1000)
                        
                    except Exception as e:
                        self.__verbose(f"Error calculating point cloud loss: {str(e)}", level=2)
                        self.__verbose(f"Predicted shape: {predicted_point_cloud.shape}", level=2)
                        self.__verbose(f"Target shape: {target_point_cloud.shape}", level=2)
                        continue

                    self.__verbose(f"\nLoss Details:", level=2)
                    self.__verbose(f"UNet Left Loss: {unet_loss_left.item():.6f}", level=2)
                    self.__verbose(f"UNet Right Loss: {unet_loss_right.item():.6f}", level=2)
                    self.__verbose(f"UNet Total Loss: {unet_loss.item():.6f}", level=2)
                    self.__verbose(f"Point Cloud Loss: {point_loss.item():.6f}", level=2)

                    if torch.isnan(unet_loss) or torch.isnan(point_loss):
                        self.__verbose("Warning: NaN detected in loss calculation", level=2)
                        self.__verbose(f"UNet Loss NaN: {torch.isnan(unet_loss)}", level=2)
                        self.__verbose(f"Point Loss NaN: {torch.isnan(point_loss)}", level=2)
                        continue

                    total_loss = unet_loss + point_loss

                if torch.isnan(total_loss):
                    self.__verbose("Warning: NaN detected in total loss", level=2)
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.unet.parameters()) +
                    list(self.left_encoder.parameters()) +
                    list(self.right_encoder.parameters()) +
                    list(self.decoder.parameters()),
                    max_grad_norm
                )
                optimizer.step()

                running_loss += total_loss.item()
                running_unet_loss += unet_loss.item()
                running_point_loss += point_loss.item()

                batch_number = i + 1
                total_batches = len(self.dataloader)
                batch_end = time.time()
                
                if batch_number % 10 == 0:  
                    self.__verbose(f"\nBatch {batch_number}/{total_batches}", level=1)
                    self.__verbose(f"UNet Loss: {unet_loss.item():.6f}", level=1)
                    self.__verbose(f"Point Cloud Loss: {point_loss.item():.6f}", level=1)
                    self.__verbose(f"Total Loss: {total_loss.item():.6f}", level=1)
                    self.__verbose(f"Time: {format_time(batch_end-batch_start)}", level=1)

            end = time.time()
            avg_loss = running_loss / len(self.dataloader)
            avg_unet_loss = running_unet_loss / len(self.dataloader)
            avg_point_loss = running_point_loss / len(self.dataloader)
            self.__verbose(f"\nEpoch [{epoch + 1}/{self.num_epochs}]", level=1)
            self.__verbose(f"Average UNet Loss: {avg_unet_loss:.6f}", level=1)
            self.__verbose(f"Average Point Cloud Loss: {avg_point_loss:.6f}", level=1)
            self.__verbose(f"Average Total Loss: {avg_loss:.6f}", level=1)
            self.__verbose(f"Time: {format_time(end - start)}\n", level=1)
            
            if self.val_dataloader is not None:
                val_loss = self._validate()
                self.__verbose(f"Validation Loss: {val_loss:.6f}", level=1)
                
                if val_loss < self.best_val_loss - self.early_stopping_min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    
                    self.best_model_state = {
                        'unet': self.unet.state_dict(),
                        'left_encoder': self.left_encoder.state_dict(),
                        'right_encoder': self.right_encoder.state_dict(),
                        'decoder': self.decoder.state_dict()
                    }
                    
                    if not os.path.exists(self.checkpoint_location):
                        os.makedirs(self.checkpoint_location)
                    
                    torch.save(self.best_model_state, f"{self.checkpoint_location}/best_model.pth")
                    self.__verbose(f"New best model saved with validation loss: {val_loss:.6f}", level=1)

                    metrics.append({
                    'epoch': epoch + 1,
                    'avg_unet_loss': avg_unet_loss,
                    'avg_point_loss': avg_point_loss,
                    'avg_total_loss': avg_loss,
                    'val_loss': val_loss if self.val_dataloader is not None else None
                    })
                    
                else:
                    self.patience_counter += 1
                    self.__verbose(f"No improvement for {self.patience_counter} epochs", level=1)
                    
                    if self.patience_counter >= self.early_stopping_patience:
                        self.__verbose(f"Early stopping triggered after {epoch + 1} epochs", level=1)
                        break
            else:
                if not os.path.exists(self.checkpoint_location):
                    os.makedirs(self.checkpoint_location)
                
                torch.save({
                    'unet': self.unet.state_dict(),
                    'left_encoder': self.left_encoder.state_dict(),
                    'right_encoder': self.right_encoder.state_dict(),
                    'decoder': self.decoder.state_dict()
                }, f"{self.checkpoint_location}/epoch_{epoch+1}.pth")

        end_time = time.time()
        self.__verbose(f"Training taken: {format_time(end_time - start_time)}", level=1)

        if self.best_model_state is not None:
            self.__verbose("Loading best model from validation", level=1)
            self.unet.load_state_dict(self.best_model_state['unet'])
            self.left_encoder.load_state_dict(self.best_model_state['left_encoder'])
            self.right_encoder.load_state_dict(self.best_model_state['right_encoder'])
            self.decoder.load_state_dict(self.best_model_state['decoder'])
        
        if not os.path.exists(self.model_location):
            os.makedirs(self.model_location)

        with open('epoch_metrics.csv', mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['epoch', 'avg_unet_loss', 'avg_point_loss', 'avg_total_loss', 'val_loss'])
            writer.writeheader()
            writer.writerows(metrics)

        torch.save(self.unet.state_dict(), f"{self.model_location}/unet.pth")
        torch.save(self.left_encoder.state_dict(), f"{self.model_location}/left_encoder.pth")
        torch.save(self.right_encoder.state_dict(), f"{self.model_location}/right_encoder.pth")
        torch.save(self.decoder.state_dict(), f"{self.model_location}/decoder.pth")
        self.__verbose("Models saved successfully", level=1)
        
    def _validate(self):
        """
        Run validation on the validation dataset.
        
        This method evaluates the model on the validation dataset and returns the validation loss.
        It uses the same loss functions as the training process but without gradient computation.
        
        Returns:
            float: Average validation loss across all batches
        """
        self.left_encoder.eval()
        self.right_encoder.eval()
        self.decoder.eval()
        self.unet.eval()
        
        criterion_unet = torch.nn.MSELoss(reduction='mean')
        criterion_point = self.memory_efficient_point_loss
        
        val_loss = 0.0
        
        with torch.no_grad():
            for data in self.val_dataloader:
                left_images, left_depths, right_images, right_depths, target_point_cloud = data
                
                left_images = left_images.to(self.device).float() / 255.0
                right_images = right_images.to(self.device).float() / 255.0
                
                left_depths = left_depths.to(self.device).float()
                right_depths = right_depths.to(self.device).float()
                left_depths = (left_depths - left_depths.min()) / (left_depths.max() - left_depths.min() + 1e-8)
                right_depths = (right_depths - right_depths.min()) / (right_depths.max() - right_depths.min() + 1e-8)
                
                target_point_cloud = target_point_cloud.to(self.device).float()
                
                with torch.amp.autocast(device_type=self.device.type):
                    left = self.unet(left_images)
                    right = self.unet(right_images)
                    
                    left_latent = self.left_encoder(torch.cat((left_images,left),dim=1))
                    right_latent = self.right_encoder(torch.cat((right_images,right),dim=1))
                    
                    fused_latent = torch.cat((left_latent, right_latent), dim=1)
                    predicted_point_cloud = self.decoder(fused_latent)
                    
                    unet_loss_left = criterion_unet(left, left_depths)
                    unet_loss_right = criterion_unet(right, right_depths)
                    unet_loss = unet_loss_left + unet_loss_right
                    
                    try:
                        point_loss = criterion_point(predicted_point_cloud, target_point_cloud, num_samples=1000)
                    except Exception as e:
                        self.__verbose(f"Error calculating validation point cloud loss: {str(e)}", level=2)
                        continue
                    
                    total_loss = unet_loss + point_loss
                    val_loss += total_loss.item()
        
        return val_loss / len(self.val_dataloader)

    def load_model(self):
        """
        Load the trained model weights from the model location.
        
        This method loads the saved weights for all model components (UNet, encoders, and decoder)
        from the specified model location directory.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading models to device: {device}")
        
        self.unet.load_state_dict(torch.load(self.model_location+"/unet.pth", map_location=device))
        self.left_encoder.load_state_dict(torch.load(self.model_location + "/left_encoder.pth", map_location=device))
        self.right_encoder.load_state_dict(torch.load(self.model_location + "/right_encoder.pth", map_location=device))
        self.decoder.load_state_dict(torch.load(self.model_location + "/decoder.pth", map_location=device))
        
    def memory_efficient_point_loss(self, pred, target, num_samples=1000):
        """
        Memory-efficient point cloud loss using a combination of sampling strategies.
        Designed to work with limited GPU memory (12GB) while still providing meaningful metrics.
        
        This loss function combines several metrics:
        - Chamfer distance (bidirectional point-to-point distance)
        - Histogram-based distribution comparison
        - Volume and center comparison
        
        Args:
            pred (torch.Tensor): predicted point cloud (B, N, 3) or (N, 3)
            target (torch.Tensor): target point cloud (B, M, 3) or (M, 3)
            num_samples (int, optional): number of points to sample from each cloud. Defaults to 1000.
            
        Returns:
            torch.Tensor: Loss value that better reflects point cloud quality while being memory-efficient
        """
        if len(pred.shape) == 2:
            pred = pred.unsqueeze(0)
        if len(target.shape) == 2:
            target = target.unsqueeze(0)
            
        B, N, _ = pred.shape
        _, M, _ = target.shape
        
        sample_size = min(num_samples, 500)
        
        if N > sample_size:
            indices = torch.randperm(N, device=pred.device)[:sample_size]
            pred = pred[:, indices, :]
            N = sample_size
            
        if M > sample_size:
            indices = torch.randperm(M, device=target.device)[:sample_size]
            target = target[:, indices, :]
            M = sample_size
        
        chamfer_loss = torch.tensor(0.0, device=pred.device)
        
        batch_size = 100
        
        for i in range(0, N, batch_size):
            end_idx = min(i + batch_size, N)
            pred_batch = pred[:, i:end_idx, :]
            
            pred_expanded = pred_batch.unsqueeze(2)
            target_expanded = target.unsqueeze(1)
            
            distances = torch.sum((pred_expanded - target_expanded) ** 2, dim=3)
            
            min_distances = torch.min(distances, dim=2)[0]
            
            chamfer_loss += torch.sum(min_distances)
        
        chamfer_loss = chamfer_loss / N
        
        target_to_pred_loss = torch.tensor(0.0, device=pred.device)
        
        for i in range(0, M, batch_size):
            end_idx = min(i + batch_size, M)
            target_batch = target[:, i:end_idx, :]
            
            target_expanded = target_batch.unsqueeze(2)
            pred_expanded = pred.unsqueeze(1)
            
            distances = torch.sum((target_expanded - pred_expanded) ** 2, dim=3)
            
            min_distances = torch.min(distances, dim=2)[0]
            
            target_to_pred_loss += torch.sum(min_distances)
        
        target_to_pred_loss = target_to_pred_loss / M
        
        chamfer_loss = chamfer_loss + target_to_pred_loss
        
        pred_2d = pred[:, :, :2]
        target_2d = target[:, :, :2]
        
        num_bins = 5
        
        x_min = min(pred_2d[:, :, 0].min(), target_2d[:, :, 0].min())
        x_max = max(pred_2d[:, :, 0].max(), target_2d[:, :, 0].max())
        y_min = min(pred_2d[:, :, 1].min(), target_2d[:, :, 1].min())
        y_max = max(pred_2d[:, :, 1].max(), target_2d[:, :, 1].max())
        
        x_min -= 1e-6
        x_max += 1e-6
        y_min -= 1e-6
        y_max += 1e-6
        
        pred_hist = torch.zeros((B, num_bins, num_bins), device=pred.device)
        target_hist = torch.zeros((B, num_bins, num_bins), device=pred.device)
        
        x_bin_size = (x_max - x_min) / num_bins
        y_bin_size = (y_max - y_min) / num_bins
        
        for b in range(B):
            for i in range(N):
                x_idx = min(int((pred_2d[b, i, 0] - x_min) / x_bin_size), num_bins - 1)
                y_idx = min(int((pred_2d[b, i, 1] - y_min) / y_bin_size), num_bins - 1)
                pred_hist[b, x_idx, y_idx] += 1
                
            for i in range(M):
                x_idx = min(int((target_2d[b, i, 0] - x_min) / x_bin_size), num_bins - 1)
                y_idx = min(int((target_2d[b, i, 1] - y_min) / y_bin_size), num_bins - 1)
                target_hist[b, x_idx, y_idx] += 1
        
        pred_hist = pred_hist / (pred_hist.sum(dim=(1, 2), keepdim=True) + 1e-8)
        target_hist = target_hist / (target_hist.sum(dim=(1, 2), keepdim=True) + 1e-8)
        
        hist_loss = torch.mean((pred_hist - target_hist) ** 2)
        
        pred_min = pred.min(dim=1)[0]
        pred_max = pred.max(dim=1)[0]
        target_min = target.min(dim=1)[0]
        target_max = target.max(dim=1)[0]
        
        pred_volume = torch.prod(pred_max - pred_min, dim=1)
        target_volume = torch.prod(target_max - target_min, dim=1)
        
        pred_center = (pred_max + pred_min) / 2
        target_center = (target_max + target_min) / 2
        
        volume_loss = torch.mean((pred_volume - target_volume) ** 2)
        center_loss = torch.mean(torch.sum((pred_center - target_center) ** 2, dim=1))
        
        combined_loss = chamfer_loss + 0.1 * hist_loss + 0.05 * volume_loss + 0.05 * center_loss
        
        scaled_loss = combined_loss * 10.0
        
        return scaled_loss

    def predict(self, left_image, right_image, target_point_cloud=None, save_path="./predictions"):
        """
        Generate a 3D point cloud prediction from stereo images.
        
        This method takes left and right stereo images, processes them through the model,
        and generates a 3D point cloud prediction. Optionally, it can compare the prediction
        with a target point cloud and save both to files.
        
        Args:
            left_image (torch.Tensor): Left stereo image
            right_image (torch.Tensor): Right stereo image
            target_point_cloud (torch.Tensor, optional): Target point cloud for comparison. Defaults to None.
            save_path (str, optional): Path to save prediction results. Defaults to "./predictions".
            
        Returns:
            tuple: (predicted_point_cloud, metrics) where metrics is a dictionary of evaluation metrics
                  or None if target_point_cloud is not provided
        """
        self.unet.eval()
        self.left_encoder.eval()
        self.right_encoder.eval()
        self.decoder.eval()

        with torch.amp.autocast(device_type=self.device.type):
            left_image = left_image.to(self.device).float() / 255.0
            right_image = right_image.to(self.device).float() / 255.0

            if torch.isnan(left_image).any() or torch.isnan(right_image).any():
                self.__verbose("Warning: NaN detected in input images", level=2)
                return None, None

            left = self.unet(left_image)
            right = self.unet(right_image)

            if torch.isnan(left).any() or torch.isnan(right).any():
                self.__verbose("Warning: NaN detected in UNet outputs", level=2)
                self.__verbose(f"Left UNet output range: [{left.min().item():.4f}, {left.max().item():.4f}]", level=2)
                self.__verbose(f"Right UNet output range: [{right.min().item():.4f}, {right.max().item():.4f}]", level=2)
                return None, None

            left_latent = self.left_encoder(torch.cat((left_image,left),dim=1))
            right_latent = self.right_encoder(torch.cat((right_image,right),dim=1))

            if torch.isnan(left_latent).any() or torch.isnan(right_latent).any():
                self.__verbose("Warning: NaN detected in encoder outputs", level=2)
                self.__verbose(f"Left latent range: [{left_latent.min().item():.4f}, {left_latent.max().item():.4f}]", level=2)
                self.__verbose(f"Right latent range: [{right_latent.min().item():.4f}, {right_latent.max().item():.4f}]", level=2)
                return None, None

            fused_latent = torch.cat((left_latent, right_latent), dim=1)

            self.__verbose("Starting decoder",level=1)
            predicted_point_cloud = self.decoder(fused_latent)

            if torch.isnan(predicted_point_cloud).any():
                self.__verbose("Warning: NaN detected in decoder output", level=2)
                self.__verbose(f"Predicted point cloud range: [{predicted_point_cloud.min().item():.4f}, {predicted_point_cloud.max().item():.4f}]", level=2)
                self.__verbose(f"Predicted point cloud shape: {predicted_point_cloud.shape}", level=2)
                return None, None

            self.__verbose(f"Predicted point cloud shape: {predicted_point_cloud.shape}", level=1)
            
            if target_point_cloud is not None:
                target_point_cloud = target_point_cloud.to(self.device)
                self.__verbose(f"Target point cloud shape: {target_point_cloud.shape}", level=1)
                
                if predicted_point_cloud.shape != target_point_cloud.shape:
                    self.__verbose(f"Warning: Shape mismatch between predicted and target point clouds", level=2)
                    self.__verbose(f"Predicted: {predicted_point_cloud.shape}, Target: {target_point_cloud.shape}", level=2)
                    return predicted_point_cloud, None
                
                try:
                    mse_loss = torch.nn.MSELoss()(predicted_point_cloud, target_point_cloud)
                    mae_loss = torch.nn.L1Loss()(predicted_point_cloud, target_point_cloud)
                    
                    chamfer_dist = self.memory_efficient_point_loss(predicted_point_cloud, target_point_cloud, num_samples=1000)
                    
                    if torch.isnan(mse_loss) or torch.isnan(mae_loss) or torch.isnan(chamfer_dist):
                        self.__verbose("Warning: NaN detected in loss calculation", level=2)
                        self.__verbose(f"MSE Loss NaN: {torch.isnan(mse_loss)}", level=2)
                        self.__verbose(f"MAE Loss NaN: {torch.isnan(mae_loss)}", level=2)
                        self.__verbose(f"Chamfer Distance NaN: {torch.isnan(chamfer_dist)}", level=2)
                        return predicted_point_cloud, None
                    
                    self.__verbose(f"MSE Loss: {mse_loss.item():.6f}", level=1)
                    self.__verbose(f"MAE Loss: {mae_loss.item():.6f}", level=1)
                    self.__verbose(f"Chamfer Distance: {chamfer_dist.item():.6f}", level=1)
                except Exception as e:
                    self.__verbose(f"Error calculating loss: {str(e)}", level=2)
                    import traceback
                    self.__verbose(f"Traceback: {traceback.format_exc()}", level=2)
                    return predicted_point_cloud, None
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                pred_filename = f"{save_path}_predicted.ply"
                target_filename = f"{save_path}_target.ply"
                
                self.save_point_cloud_as_ply(predicted_point_cloud, pred_filename)
                self.save_point_cloud_as_ply(target_point_cloud, target_filename)
                
                self.__verbose(f"Saved predicted point cloud to {pred_filename}", level=1)
                self.__verbose(f"Saved target point cloud to {target_filename}", level=1)
                
                return predicted_point_cloud, {
                    'mse': mse_loss.item(),
                    'mae': mae_loss.item(),
                    'chamfer': chamfer_dist.item(),
                }
            else:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                output_filename = f"{save_path}_predicted.ply"
                self.save_point_cloud_as_ply(predicted_point_cloud, output_filename)
                self.__verbose(f"Saved predicted point cloud to {output_filename}", level=1)
                
                return predicted_point_cloud, None

    def save_point_cloud_as_ply(self, point_cloud, output_filename):
        """
        Save a point cloud to a PLY file with proper coordinates.
        
        This method converts a point cloud tensor to a PLY file format that can be
        opened in 3D visualization software. It handles normalization and denormalization
        of the point cloud coordinates.
        
        Args:
            point_cloud (torch.Tensor): Point cloud to save
            output_filename (str): Path to save the PLY file
        """
        try:
            point_cloud = point_cloud.reshape(-1, 3)
            
            if torch.is_tensor(point_cloud):
                points_np = point_cloud.detach().cpu().numpy()
            else:
                points_np = point_cloud
            
            if np.isnan(points_np).any() or np.isinf(points_np).any():
                self.__verbose("Warning: NaN or infinite values found in point cloud before saving", level=2)
                points_np = np.nan_to_num(points_np, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if hasattr(self, 'dataloader') and self.dataloader is not None and hasattr(self.dataloader.dataset, 'normalization_params'):
                params = self.dataloader.dataset.normalization_params

                points_np = points_np * params['scale'].cpu().numpy() + params['center'].cpu().numpy()
                self.__verbose("Point cloud denormalized using stored parameters", level=2)
            else:
                points_np = points_np * 2.0 - 1.0
                self.__verbose("Point cloud normalized using default parameters", level=2)
            
            center = np.mean(points_np, axis=0)
            points_np = points_np - center
            
            max_dist = np.max(np.abs(points_np))
            if max_dist > 0:
                scale = 1.0 / max_dist
                points_np = points_np * scale
            
            pcd = o3d.geometry.PointCloud()
            
            pcd.points = o3d.utility.Vector3dVector(points_np)
            
            self.__verbose(f"Point cloud shape: {points_np.shape}", level=2)
            self.__verbose(f"Point cloud range: [{points_np.min()}, {points_np.max()}]", level=2)
            self.__verbose(f"Point cloud mean: {points_np.mean(axis=0)}", level=2)
            
            o3d.io.write_point_cloud(output_filename, pcd, write_ascii=True)
            self.__verbose(f"Saved point cloud to {output_filename}", level=1)
            
        except Exception as e:
            self.__verbose(f"Error saving point cloud: {str(e)}", level=1)
            import traceback
            self.__verbose(f"Traceback: {traceback.format_exc()}", level=1)
            raise

    def getUnetOutput(self,left,right):
        """
        Get the UNet output for left and right images.
        
        This method processes left and right images through the UNet model and returns
        the processed outputs. It's useful for visualizing the intermediate results
        of the UNet processing.
        
        Args:
            left (torch.Tensor): Left image
            right (torch.Tensor): Right image
            
        Returns:
            tuple: (left_unet_output, right_unet_output)
        """
        self.unet.eval()
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type):
                left_images = left.to(self.device).float() / 255.0
                right_images = right.to(self.device).float() / 255.0
                left_output = self.unet(left_images)
                right_output = self.unet(right_images)
                return left_output, right_output, left_output, right_output

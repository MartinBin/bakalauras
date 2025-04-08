import os
from models.Decoder import Decoder
from models.Encoder import Encoder
import torch
import numpy as np
import torch.utils.checkpoint as cp
import open3d as o3d
import time
from models.Unet import UNet
import cv2
from datetime import datetime, timedelta

def format_time(seconds):
    """Format time duration in a human-readable way"""
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
    def __init__(self, dataloader=None, num_epochs=10, latent_dim=32, checkpoint_location="./Checkpoints",model_location="./Trained_Models",verbose=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        torch.cuda.empty_cache()
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.left_encoder = Encoder(latent_dim).to(self.device)
        self.right_encoder = Encoder(latent_dim).to(self.device)
        self.decoder = Decoder(latent_dim*2, num_points=262144).to(self.device)
        self.unet = UNet(3, 3).to(self.device)
        self.checkpoint_location = checkpoint_location
        self.model_location = model_location
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        self.verbose = verbose

    def __verbose(self, message, level=1):
        """Print message only if verbose level is sufficient"""
        if self.verbose >= level:
            print(message)

    def train(self):
        criterion_unet = torch.nn.MSELoss(reduction='mean')
        criterion_point = self.memory_efficient_point_loss
        
        optimizer = torch.optim.Adam(
            list(self.unet.parameters()) +
            list(self.left_encoder.parameters()) +
            list(self.right_encoder.parameters()) +
            list(self.decoder.parameters()), lr=0.0001)

        max_grad_norm = 1.0

        self.__verbose("Training...", level=1)
        start_time = time.time()
        for epoch in range(self.num_epochs):
            start = time.time()
            running_loss = 0.0
            running_unet_loss = 0.0
            running_point_loss = 0.0
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
                self.left_encoder.train()
                self.right_encoder.train()
                self.decoder.train()
                self.unet.train()

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

                    # Calculate UNet loss (for depth maps)
                    unet_loss_left = criterion_unet(left, left_depths)
                    unet_loss_right = criterion_unet(right, right_depths)
                    unet_loss = unet_loss_left + unet_loss_right
                    
                    # Calculate point cloud loss using memory-efficient approach
                    try:
                        # Use memory-efficient point loss
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

                    # Combine UNet and point cloud losses
                    # You can adjust the weights to prioritize one over the other
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

        end_time = time.time()
        self.__verbose(f"Training taken: {format_time(end_time - start_time)}", level=1)

        if not os.path.exists(self.model_location):
            os.makedirs(self.model_location)

        torch.save(self.unet.state_dict(), f"{self.model_location}/unet.pth")
        torch.save(self.left_encoder.state_dict(), f"{self.model_location}/left_encoder.pth")
        torch.save(self.right_encoder.state_dict(), f"{self.model_location}/right_encoder.pth")
        torch.save(self.decoder.state_dict(), f"{self.model_location}/decoder.pth")
        self.__verbose("Models saved successfully", level=1)

    def load_model(self):
        self.unet.load_state_dict(torch.load(self.model_location+"/unet.pth"))
        self.left_encoder.load_state_dict(torch.load(self.model_location + "/left_encoder.pth"))
        self.right_encoder.load_state_dict(torch.load(self.model_location + "/right_encoder.pth"))
        self.decoder.load_state_dict(torch.load(self.model_location + "/decoder.pth"))
        
    def memory_efficient_point_loss(self, pred, target, num_samples=1000):
        """
        Memory-efficient point cloud loss using a combination of sampling strategies.
        Designed to work with limited GPU memory (12GB) while still providing meaningful metrics.
        
        Args:
            pred: predicted point cloud (B, N, 3) or (N, 3)
            target: target point cloud (B, M, 3) or (M, 3)
            num_samples: number of points to sample from each cloud
            
        Returns:
            Loss value that better reflects point cloud quality while being memory-efficient
        """
        # Ensure inputs are 3D tensors
        if len(pred.shape) == 2:
            pred = pred.unsqueeze(0)
        if len(target.shape) == 2:
            target = target.unsqueeze(0)
            
        # Get dimensions
        B, N, _ = pred.shape
        _, M, _ = target.shape
        
        # Use a smaller number of samples for memory efficiency
        sample_size = min(num_samples, 500)  # Limit to 500 points for memory efficiency
        
        # Sample points if needed
        if N > sample_size:
            indices = torch.randperm(N, device=pred.device)[:sample_size]
            pred = pred[:, indices, :]
            N = sample_size
            
        if M > sample_size:
            indices = torch.randperm(M, device=target.device)[:sample_size]
            target = target[:, indices, :]
            M = sample_size
        
        # 1. Calculate a memory-efficient approximation of Chamfer distance
        # Instead of computing all pairwise distances, use a batched approach
        
        # Initialize loss components
        chamfer_loss = torch.tensor(0.0, device=pred.device)
        
        # Process in smaller batches to save memory
        batch_size = 100  # Process 100 points at a time
        
        # pred -> target direction
        for i in range(0, N, batch_size):
            end_idx = min(i + batch_size, N)
            pred_batch = pred[:, i:end_idx, :]  # (B, batch_size, 3)
            
            # Calculate distances to all target points
            # This is still memory-intensive but we're limiting the batch size
            pred_expanded = pred_batch.unsqueeze(2)  # (B, batch_size, 1, 3)
            target_expanded = target.unsqueeze(1)  # (B, 1, M, 3)
            
            # Calculate squared distances
            distances = torch.sum((pred_expanded - target_expanded) ** 2, dim=3)  # (B, batch_size, M)
            
            # Find minimum distance for each predicted point
            min_distances = torch.min(distances, dim=2)[0]  # (B, batch_size)
            
            # Add to chamfer loss
            chamfer_loss += torch.sum(min_distances)
        
        # Normalize by number of points
        chamfer_loss = chamfer_loss / N
        
        # target -> pred direction (similar approach)
        target_to_pred_loss = torch.tensor(0.0, device=pred.device)
        
        for i in range(0, M, batch_size):
            end_idx = min(i + batch_size, M)
            target_batch = target[:, i:end_idx, :]  # (B, batch_size, 3)
            
            # Calculate distances to all predicted points
            target_expanded = target_batch.unsqueeze(2)  # (B, batch_size, 1, 3)
            pred_expanded = pred.unsqueeze(1)  # (B, 1, N, 3)
            
            # Calculate squared distances
            distances = torch.sum((target_expanded - pred_expanded) ** 2, dim=3)  # (B, batch_size, N)
            
            # Find minimum distance for each target point
            min_distances = torch.min(distances, dim=2)[0]  # (B, batch_size)
            
            # Add to target->pred loss
            target_to_pred_loss += torch.sum(min_distances)
        
        # Normalize by number of points
        target_to_pred_loss = target_to_pred_loss / M
        
        # Combine both directions for full Chamfer distance
        chamfer_loss = chamfer_loss + target_to_pred_loss
        
        # 2. Calculate a simple density loss using histograms
        # This is more memory-efficient than the previous approach
        
        # Project points onto a 2D plane and create histograms
        pred_2d = pred[:, :, :2]  # (B, N, 2)
        target_2d = target[:, :, :2]  # (B, M, 2)
        
        # Create histograms with fewer bins to save memory
        num_bins = 5  # Reduced from 10 to save memory
        
        # Calculate bin edges
        x_min = min(pred_2d[:, :, 0].min(), target_2d[:, :, 0].min())
        x_max = max(pred_2d[:, :, 0].max(), target_2d[:, :, 0].max())
        y_min = min(pred_2d[:, :, 1].min(), target_2d[:, :, 1].min())
        y_max = max(pred_2d[:, :, 1].max(), target_2d[:, :, 1].max())
        
        # Add small epsilon to avoid edge cases
        x_min -= 1e-6
        x_max += 1e-6
        y_min -= 1e-6
        y_max += 1e-6
        
        # Create histograms
        pred_hist = torch.zeros((B, num_bins, num_bins), device=pred.device)
        target_hist = torch.zeros((B, num_bins, num_bins), device=pred.device)
        
        # Calculate bin indices
        x_bin_size = (x_max - x_min) / num_bins
        y_bin_size = (y_max - y_min) / num_bins
        
        # Fill histograms
        for b in range(B):
            for i in range(N):
                x_idx = min(int((pred_2d[b, i, 0] - x_min) / x_bin_size), num_bins - 1)
                y_idx = min(int((pred_2d[b, i, 1] - y_min) / y_bin_size), num_bins - 1)
                pred_hist[b, x_idx, y_idx] += 1
                
            for i in range(M):
                x_idx = min(int((target_2d[b, i, 0] - x_min) / x_bin_size), num_bins - 1)
                y_idx = min(int((target_2d[b, i, 1] - y_min) / y_bin_size), num_bins - 1)
                target_hist[b, x_idx, y_idx] += 1
        
        # Normalize histograms
        pred_hist = pred_hist / (pred_hist.sum(dim=(1, 2), keepdim=True) + 1e-8)
        target_hist = target_hist / (target_hist.sum(dim=(1, 2), keepdim=True) + 1e-8)
        
        # Calculate histogram loss
        hist_loss = torch.mean((pred_hist - target_hist) ** 2)
        
        # 3. Calculate a simple shape loss using bounding boxes
        # This helps ensure the overall shape is similar
        
        # Calculate bounding boxes
        pred_min = pred.min(dim=1)[0]  # (B, 3)
        pred_max = pred.max(dim=1)[0]  # (B, 3)
        target_min = target.min(dim=1)[0]  # (B, 3)
        target_max = target.max(dim=1)[0]  # (B, 3)
        
        # Calculate volume and center differences
        pred_volume = torch.prod(pred_max - pred_min, dim=1)  # (B,)
        target_volume = torch.prod(target_max - target_min, dim=1)  # (B,)
        
        pred_center = (pred_max + pred_min) / 2  # (B, 3)
        target_center = (target_max + target_min) / 2  # (B, 3)
        
        # Calculate volume and center loss
        volume_loss = torch.mean((pred_volume - target_volume) ** 2)
        center_loss = torch.mean(torch.sum((pred_center - target_center) ** 2, dim=1))
        
        # Combine all losses with appropriate weights
        # Chamfer distance is the primary metric
        combined_loss = chamfer_loss + 0.1 * hist_loss + 0.05 * volume_loss + 0.05 * center_loss
        
        # Scale the loss to a reasonable range
        scaled_loss = combined_loss * 10.0
        
        return scaled_loss

    def predict(self, left_image, right_image, target_point_cloud=None, save_path="./predictions"):
        self.unet.eval()
        self.left_encoder.eval()
        self.right_encoder.eval()
        self.decoder.eval()

        with torch.amp.autocast(device_type=self.device.type):
            # Normalize inputs the same way as in training
            left_image = left_image.to(self.device).float() / 255.0
            right_image = right_image.to(self.device).float() / 255.0

            # Check for NaN in inputs
            if torch.isnan(left_image).any() or torch.isnan(right_image).any():
                self.__verbose("Warning: NaN detected in input images", level=2)
                return None, None

            # Get UNet outputs
            left = self.unet(left_image)
            right = self.unet(right_image)

            # Check for NaN in UNet outputs
            if torch.isnan(left).any() or torch.isnan(right).any():
                self.__verbose("Warning: NaN detected in UNet outputs", level=2)
                self.__verbose(f"Left UNet output range: [{left.min().item():.4f}, {left.max().item():.4f}]", level=2)
                self.__verbose(f"Right UNet output range: [{right.min().item():.4f}, {right.max().item():.4f}]", level=2)
                return None, None

            left_latent = self.left_encoder(torch.cat((left_image,left),dim=1))
            right_latent = self.right_encoder(torch.cat((right_image,right),dim=1))

            # Check for NaN in encoder outputs
            if torch.isnan(left_latent).any() or torch.isnan(right_latent).any():
                self.__verbose("Warning: NaN detected in encoder outputs", level=2)
                self.__verbose(f"Left latent range: [{left_latent.min().item():.4f}, {left_latent.max().item():.4f}]", level=2)
                self.__verbose(f"Right latent range: [{right_latent.min().item():.4f}, {right_latent.max().item():.4f}]", level=2)
                return None, None

            fused_latent = torch.cat((left_latent, right_latent), dim=1)
            predicted_point_cloud = self.decoder(fused_latent)

            # Check for NaN in decoder output
            if torch.isnan(predicted_point_cloud).any():
                self.__verbose("Warning: NaN detected in decoder output", level=2)
                self.__verbose(f"Predicted point cloud range: [{predicted_point_cloud.min().item():.4f}, {predicted_point_cloud.max().item():.4f}]", level=2)
                self.__verbose(f"Predicted point cloud shape: {predicted_point_cloud.shape}", level=2)
                return None, None

            self.__verbose(f"Predicted point cloud shape: {predicted_point_cloud.shape}", level=1)
            
            # Compare with target point cloud if provided
            if target_point_cloud is not None:
                target_point_cloud = target_point_cloud.to(self.device)
                self.__verbose(f"Target point cloud shape: {target_point_cloud.shape}", level=1)
                
                # Check for shape mismatch
                if predicted_point_cloud.shape != target_point_cloud.shape:
                    self.__verbose(f"Warning: Shape mismatch between predicted and target point clouds", level=2)
                    self.__verbose(f"Predicted: {predicted_point_cloud.shape}, Target: {target_point_cloud.shape}", level=2)
                    return predicted_point_cloud, None
                
                # Calculate metrics
                try:
                    # Traditional MSE and MAE
                    mse_loss = torch.nn.MSELoss()(predicted_point_cloud, target_point_cloud)
                    mae_loss = torch.nn.L1Loss()(predicted_point_cloud, target_point_cloud)
                    
                    # Point cloud specific metrics
                    chamfer_dist = self.memory_efficient_point_loss(predicted_point_cloud, target_point_cloud, num_samples=1000)
                    
                    # Check for NaN in loss
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
                # If no target is provided, just save the predicted point cloud
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                output_filename = f"{save_path}_predicted.ply"
                self.save_point_cloud_as_ply(predicted_point_cloud, output_filename)
                self.__verbose(f"Saved predicted point cloud to {output_filename}", level=1)
                
                return predicted_point_cloud, None

    def save_point_cloud_as_ply(self, point_cloud, output_filename):
        """Save point cloud to PLY file with proper coordinates"""
        try:
            # Ensure point cloud is in the correct format (N x 3)
            point_cloud = point_cloud.reshape(-1, 3)
            
            # Convert to numpy if it's a PyTorch tensor
            if torch.is_tensor(point_cloud):
                points_np = point_cloud.detach().cpu().numpy()
            else:
                points_np = point_cloud
            
            # Check for NaN or infinite values
            if np.isnan(points_np).any() or np.isinf(points_np).any():
                self.__verbose("Warning: NaN or infinite values found in point cloud before saving", level=2)
                points_np = np.nan_to_num(points_np, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Denormalize the point cloud if normalization parameters are available
            if hasattr(self, 'dataloader') and hasattr(self.dataloader.dataset, 'normalization_params'):
                params = self.dataloader.dataset.normalization_params
                # Denormalize: point * scale + center
                points_np = points_np * params['scale'].cpu().numpy() + params['center'].cpu().numpy()
                self.__verbose("Point cloud denormalized using stored parameters", level=2)
            
            # Create Open3D point cloud object
            pcd = o3d.geometry.PointCloud()
            
            # Set points directly from numpy array
            pcd.points = o3d.utility.Vector3dVector(points_np)
            
            # Print point cloud statistics for debugging
            self.__verbose(f"Point cloud shape: {points_np.shape}", level=2)
            self.__verbose(f"Point cloud range: [{points_np.min()}, {points_np.max()}]", level=2)
            self.__verbose(f"Point cloud mean: {points_np.mean(axis=0)}", level=2)
            
            # Save to file
            o3d.io.write_point_cloud(output_filename, pcd)
            self.__verbose(f"Saved point cloud to {output_filename}", level=1)
            
        except Exception as e:
            print(f"Error saving point cloud: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def get_unet_output(self, image):
        """
        Get the UNet output for a single image
        
        Args:
            image: Input image tensor of shape [1, 3, H, W]
            
        Returns:
            UNet output tensor of shape [1, 3, H, W]
        """
        self.unet.eval()
        with torch.no_grad():
            # Ensure image is on the correct device
            image = image.to(self.device)
            
            # Get UNet output
            unet_output = self.unet(image)
            
            # Normalize output to [0, 1] range for visualization
            unet_output = torch.clamp(unet_output, 0, 1)
            
            return unet_output